import uvicorn
from fastapi import FastAPI

import json
import numpy as np
import pickle
import pandas as pd

app=FastAPI()
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

from sklearn.model_selection import train_test_split

import pandas as pd
URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(URL, sep=";")

train, test = train_test_split(df)
test_x = test.drop(["quality"], axis=1)



@app.get("/")
def home():
    return {'Message':'Welcome to mlflow'}

@app.get('/predict')
def predict():
    # data = list(lambda x: int(x), data.split(','))
    prediction=classifier.predict(test_x)

    return {
        'prediction':prediction.tolist()
    }

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)




