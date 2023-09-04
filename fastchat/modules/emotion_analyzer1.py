import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
from pysentimiento import create_analyzer

# Verify if GPU is being used (optional, for testing)
#import torch
#print(f"Is PyTorch using a GPU?: {torch.cuda.is_available()}")

#import tensorflow as tf
#print(f"Is TensorFlow using a GPU?: {tf.config.list_physical_devices('GPU')}")

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, db=0)

class Text(BaseModel):
    text: str

sentiment_analyzer = create_analyzer(task="sentiment", lang="en")
irony_analyzer = create_analyzer(task="irony", lang="en")
emotion_analyzer = create_analyzer(task="emotion", lang="en")

def emotion_analyzer_function(text: str):
    sentiment_result = sentiment_analyzer.predict(text)
    irony_result = irony_analyzer.predict(text)
    emotion_result = emotion_analyzer.predict(text)
    
    sentiment = sentiment_result.output
    sentiment_prob = sentiment_result.probas
    
    irony = irony_result.output
    irony_prob = irony_result.probas
    
    emotion = emotion_result.output
    emotion_prob = emotion_result.probas
    
    analysis = {
        "sentiment": sentiment,
        "sentiment_prob": sentiment_prob,
        "irony": irony,
        "irony_prob": irony_prob,
        "emotion": emotion,
        "emotion_prob": emotion_prob,
    }
    
    return analysis

def get_cached_analysis(text: str):
    cached_result = r.get(text)
    if cached_result:
        return json.loads(cached_result)
    else:
        analysis_result = emotion_analyzer_function(text)
        r.set(text, json.dumps(analysis_result))
        return analysis_result
