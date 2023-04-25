from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI();

# Set up allowed origins, methods, and headers
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    data: List[int]


with open('linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/ml/predict')
async def predict_endpoint(item: Item):
    data = item.data
    yhat = model.predict(np.array([data]))
    ans = yhat[0]
    #print(yhat)
    return {"Prediction": ans}