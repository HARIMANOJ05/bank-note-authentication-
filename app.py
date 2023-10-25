# import the required libraries 

from typing import Iterable
import uvicorn
from fastapi import  FastAPI
from banknotes import BankNote
import pickle 
import numpy as np
import pandas as pd

app=FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#index route on app @ 8000 port 
@app.get('/')
def index():
    return {'messages':'hello, mr.engineer'}

@app.get('/{name}')
def get_name(name: str):
    return {'welcome to the bank note authentication app.':f'{name}'} 

@app.post('/predict')
def predict_banknote(data:BankNote):
    data=data.dict()
    
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy = data ['entropy']
    
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note "
    else:
        prediction="Iterable s a Bank Note"

    return{
        'prediction':prediction
    }
#run the api with uvicorn 
# will run on http://127.0.01:8000

if __name__ =='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)


















