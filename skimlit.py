import json
import requests
import streamlit as st
from PIL import Image

st.title('SkimLit - making summaries skimmable')
st.header('This app helps make unreadable summaries easily skimmable - give it a go!')

if 'visibility' not in st.session_state:
    st.session_state.visibility = 'visible'
    st.session_state.disabled = False

image = Image.open('./Images/skimlit.png')
st.image(
    image,
    caption = "Here's an illustration of what the Skimlit app does"
)

text_input = st.text_area(
    "Enter the 'unskimmable' abstract here 👇",
    height = 100,
    label_visibility = st.session_state.visibility,
    disabled = st.session_state.disabled,
    placeholder = '''
    This RCT examined the efficacy of a manualized social intervention for children with HFASDs. 
    Participants were randomly assigned to treatment or wait-list conditions. 
    Treatment included instruction and therapeutic activities targeting social skills, face-emotion recognition, interest expansion, and interpretation of non-literal language. 
    A response-cost program was applied to reduce problem behaviors and foster skills acquisition. 
    Significant treatment effects were found for five of seven primary outcome measures (parent ratings and direct child measures). 
    Secondary measures based on staff ratings (treatment group only) corroborated gains reported by parents. 
    High levels of parent, child and staff satisfaction were reported, along with high levels of treatment fidelity. 
    Standardized effect size estimates were primarily in the medium and large ranges and favored the treatment group.
    '''
)

if st.button('Skim it'):
    with st.spinner(
        text = 'Skimlit is working its magic: Skimmable summaries here we go...'
    ):
        res = requests.post(
            url = 'https://http://127.0.0.1:8000/',
            data = json.dumps(text_input)
        )
    
    st.write(
        f'''{res.text}'''
    )
                
with st.expander("About Skimlit"):

    about = """
    **[Skmlit](https://github.com/tituslhy/Skimlit)** uses a language model trained on
    character, word and sentence position encodings from the paper 
    [PubMed 200k RCT: a Dataset for Sequenctial Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071) 
    to classify sentences in an abstract. The sentences of the abstract are then classified and returned to the user 
    in a skimmable format.
    
    The model is trained using TensorFlow on Colab. (**Thank you Google for the free GPUs** ❤)
    
    **Created by:**
    * Titus LIM Hsien Yong
    """
    st.write(about)
    st.write("")
    model_text = """
    The Model is trained on ~20,000 abstracts on just 3 epochs, yielding an overall test accuracy of 85%.
    """
    st.markdown(model_text)