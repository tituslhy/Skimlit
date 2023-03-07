from collections import defaultdict
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, mixed_precision, Sequential
from tensorflow.keras.layers import TextVectorization

import tensorflow_hub as hub

from spacy.lang.en import English
import docx

import os
import tempfile
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','config'))
from config import *

import warnings
warnings.filterwarnings("ignore")

def preprocess_sentence(abstract: str) -> list:
    """Takes the summary of interest and returns a list of parsed sentence

    Args:
        Abstract (str): The input furnished from the user

    Returns:
        list: List of parsed sentences
        
    Reference: 
    https://spacy.io/usage/linguistic-features#sbd%20from%20spacy.lang.en%20import%20English
    """
    nlp = English()
    nlp.add_pipe('sentencizer')
    
    #Creates 'doc' of parsed sequences
    doc = nlp(abstract)
    
    return [str(sent) for sent in list(doc.sents)]

def preprocess_position(abstract_lines: list,
                        line_number_depth: int = LINE_DEPTH,
                        total_lines_depth: int = TOTAL_LINE_DEPTH) -> tf.Tensor:
    """
    The function develops postional encoding tensors for input to the model by:
    1. Generating a list of dictionaries from list of parsed sentences. Each dictionary includes the following field:
        a. text - this is the parsed sentence
        b. line_number - this is the parsed sentence's position in the entire abstract
        c. total_lines - this is the total number of lines in the abstract
    
    2. Using tensorflow's one-hot encoding function for positional encoding.

    Args:
        abstract_lines (str): list of parsed sentences from the preprocess_sentence function

    Returns:
        abstract_line_numbers_one_hot (tf.Tensor): One-hot encoded tensor of sentence positions in abstract
        abstract_total_lines_one_hot (tf.Tensor): One-hot encoded tensor of total lines in abstract
    """
    total_lines_in_sample = len(abstract_lines)
    
    # Assemble list of dictonaries
    sample_lines = []
    for line_number, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict['text'] = line
        sample_dict['line_number'] = line_number
        sample_dict['total_lines'] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
    
    # Assemble positional encoding
    abstract_line_numbers = [line['line_number'] for line in sample_lines]
    abstract_line_numbers_one_hot = tf.one_hot(abstract_line_numbers, depth = line_number_depth)
    
    abstract_total_lines = [line['total_lines'] for line in sample_lines]
    abstract_total_lines_one_hot = tf.one_hot(abstract_total_lines, depth = total_lines_depth)
    
    return abstract_line_numbers_one_hot, abstract_total_lines_one_hot

def split_to_chars(text: str) -> str:
    """Splits a sentence of words to a sentence of characters

    Args:
        text (str): Sentence of interest

    Returns:
        str: Sentence of characters
    """
    return " ".join(list(text))

def preprocess_chars(abstract_lines: list) -> list:
    """Returns list of sentences of characters from parsed sentences of abstract

    Args:
        abstract_lines (list): list of parsed sentences from the preprocess_sentence function

    Returns:
        list: list of sentences of characters from abstract
    """
    return [split_to_chars(sentence) for sentence in abstract_lines]

def build_model(num_char_tokens : int = NUM_CHAR_TOKENS,
                output_seq_char_len: int = output_seq_char_len,
                len_char_vocab: int = LEN_CHAR_VOCAB) -> tf.keras.Model: 
    """Instantiates the tribrid model architecture for character, sentence and positional embeddings

    Args:
        num_char_tokens (int): Number of character tokens in vocabulary. Defaults to NUM_CHAR_TOKENS.
        output_seq_char_len (int): Maximum number of characters in output sequence. Defaults to output_seq_char_len.
        len_char_vocab (int): Number of characters in vocabulary. Defaults to LEN_CHAR_VOCAB.

    Returns:
        model: A compiled tensorflow model.
    """
    
    # Token model
    sentence_embedding_layer = hub.KerasLayer(
    'https://tfhub.dev/google/universal-sentence-encoder/4',
    trainable = False,
    name = 'universal_sentence_encoder')
    
    token_inputs = layers.Input(shape=[],
                                dtype="string",
                                name="token_inputs")
    token_embeddings = sentence_embedding_layer(token_inputs)
    token_outputs = layers.Dense(128, activation='relu')(token_embeddings)
    
    token_model = tf.keras.Model(inputs=token_inputs,
                                 outputs=token_outputs)

    # Char model
    char_vectorizer = TextVectorization(
        max_tokens = num_char_tokens,
        output_sequence_length = output_seq_char_len,
        standardize='lower_and_strip_punctuation',
        name = "char_vectorizer")
    
    char_embed = layers.Embedding(
        input_dim = num_char_tokens,
        output_dim = len_char_vocab,
        mask_zero = False,
        name = 'char_embed')
    
    char_inputs = layers.Input(shape=(1,),
                                dtype="string",
                                name="char_inputs")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(
        layers.LSTM(32)
    )(char_embeddings)
    char_model = tf.keras.Model(inputs = char_inputs,
                                outputs = char_bi_lstm)

    # line_numbers position encoding model
    line_number_inputs = layers.Input(
        shape = (15,), #matches the depth
        dtype = tf.int32,
        name = 'line_number_input'
    )
    x = layers.Dense(32, activation='relu')(line_number_inputs)
    line_number_model = tf.keras.Model(inputs = line_number_inputs,
                                        outputs = x)

    # total_line_numbers position enocding model
    total_line_inputs = layers.Input(
        shape = (20,),
        dtype = tf.int32,
        name = 'total_line_inputs'
    )
    y = layers.Dense(32, activation='relu')(total_line_inputs)
    total_line_model = tf.keras.Model(inputs = total_line_inputs,
                                      outputs = y)
    
    # Combining embeddings
    combined_embeddings = layers.Concatenate(
        name="token_char_hybrid_embedding"
        )([token_model.output, char_model.output])

    # Sentence embeddings are passed through a dropout
    z = layers.Dense(256, activation = 'relu')(combined_embeddings)
    z = layers.Dropout(0.5)(z) #following the paper

    # Combine z with position embeddings
    tribrid_embeddings = layers.Concatenate(
        name = 'char_token_position_embedding'
    )([line_number_model.output,
        total_line_model.output,
        z])

    output_layer = layers.Dense(
        5,
        activation = 'softmax',
        name = 'output_layer'
    )(tribrid_embeddings)

    model = tf.keras.Model(inputs = [line_number_model.input,
                                    total_line_model.input,
                                    token_model.input,
                                    char_model.input],
                           outputs = output_layer,
                           name = 'model_tribrid_model')
    
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
        )

    return model

def run_inference(model: tf.keras.Model,
                  abstract_line_numbers_one_hot: tf.Tensor,
                  abstract_total_lines_one_hot: tf.Tensor,
                  abstract_lines: list,
                  abstract_chars: list,
                  labels : list =  CLASSES):
    """Generates model predictions

    Args:
        model (tf.keras.Model): model object with weights instantiated
        abstract_line_numbers_one_hot (tf.Tensor): tensor output from positional encoding preprocessing
        abstract_total_lines_one_hot (tf.Tensor): tensor output from positional encoding preprocessing
        abstract_lines (list): list of parsed sentences from sentence preprocessing
        abstract_chars (list): list of character sentences from character preprocessing
        labels (CLASSES): class labels

    Returns:
        list: list of labels for each line.
    """
    
    try:
        pred_probabilities = model.predict(x=(abstract_line_numbers_one_hot,
                                          abstract_total_lines_one_hot,
                                          tf.constant(abstract_lines),
                                          tf.constant(abstract_chars)))
    except:
        return tensorflow_prediction_exception()
    predictions = tf.argmax(pred_probabilities, axis=1)
    return [labels[i] for i in predictions]

def return_text(sentence_labels: list,
                abstract_lines: list):
    """Returns a skimmable summary

    Args:
        sentence_labels (list): List of model predicted labels for each sentence
        abstract_lines (list): List of parsed sentences from abstract
    """
    compile = defaultdict(list)
    for idx, sentence in enumerate(abstract_lines):
        compile[sentence_labels[idx]].append(sentence)
    for label in compile:
        compile[label] = ' '.join(compile[label])
    text = ''
    for summary in compile:
        if len(text)==0:
            text += '__'+ summary + '__ : ' + compile[summary] + '  '
        else:
            text += '\n__'+ summary + '__ : ' + compile[summary] + '  '
    return text
    