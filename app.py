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
    
@app.get('/')
async def root():
    return {'message': 'Welcome to the Skimlit API!'}