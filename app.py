import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf

from config.config import *
from utils.utils import *
from utils.exceptions import *

app = FastAPI()

origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Abstract (BaseModel):
    Abstract: str
    Checkpoint_dir: str
    class Config:
        schema_extra = {
            "example": {
                "Abstract": "The wheels of the bus go round and round",
                "Checkpoint_dir": '../training/cp.ckpt'
            }
        }
    
@app.get('/')
async def root():
    return {'message': 'Welcome to the Skimlit API!'}

@app.get('/Abstract')
async def get_skimmable_abstract(abstract: Abstract,):
    
    abstract_lines = preprocess_sentence(abstract.Abstract)
    abstract_line_numbers_one_hot, abstract_total_lines_one_hot = preprocess_position(abstract_lines)
    abstract_chars = preprocess_chars(abstract_lines)
    
    model = build_model()
    model.load_weights(abstract.checkpoint_dir)
    labelled_sentences = run_inference(model,
                                       abstract_line_numbers_one_hot,
                                       abstract_total_lines_one_hot,
                                       abstract_lines,
                                       abstract_chars)