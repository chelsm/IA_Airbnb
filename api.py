import io
import os
import pickle
import uuid

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from function import train_model, get_last_model, predict

client = OpenAI(
  organization='org-0CmOzjMuVo2WWrKtcHmU8BSf',
  project='proj_MB8lPmmfaHr67ptQs2BPkOU8',
  api_key= "sk-proj-6FFJRqZbaGA5b2CTM2ZTT3BlbkFJSXUDFKwliJfkvLo5NrjV    "
)

tags_metadata = [
    {
        "name": "Training",
        "description": "Operations with training data.",
    },
    {
        "name": "Predict",
        "description": "Operations with models.",
    },
    {
        "name": "Models",
        "description": "Call to Open IA API.",
    }
]


app = FastAPI(
    title="Airbnb Price Prediction API",
    openapi_tags=tags_metadata,
    description="""
    # Airbnb Price Prediction API
    This is an API for predicting the price of an Airbnb listing.
    """
)

class TrainingData(BaseModel):
    log_price: int
    property_type: int
    room_type: int
    accommodates: int
    bathrooms: int
    bed_type: int
    cancellation_policy: str
    cleaning_fee: bool
    name: str
    city: str
    host_identity_verified: bool
    number_of_reviews: int
    review_scores_rating: int
    zipcode: int
    bedrooms: int
    beds: int


class PredictionData(BaseModel):
    property_type: int
    room_type: int
    bathrooms: int
    accommodates: int
    bedrooms: int
    beds: int


@app.post("/training", tags=["Training"], summary="Entraîner le modèle", description="Endpoint pour entraîner un modèle de machine learning.")
def train_model_endpoint(file_path: str): 
    try:
        df = pd.read_csv(file_path)
        model = train_model(df)
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        model_filename = f"model_{uuid.uuid4()}.pkl"
        model_file_path = os.path.join('trained_models', model_filename)
        with open(model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        return {"message": "Modèle entraîné avec succès et sauvegardé"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", tags=["Predict"], summary="Faire une prédiction", description="Endpoint pour faire une prédiction à partir du modèle entraîné.")
def predict_endpoint(file_path: str):
    try:
        df = pd.read_csv(file_path)
        model_path = get_last_model()
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        prediction = predict(model, df)
        return {"prediction": prediction}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model", tags=["Models"], summary="Obtenir un modèle d'IA", description="Endpoint pour obtenir un modèle d'IA de l'API OpenAI.")
def get_model(prompt: str):
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=7,
            temperature=0
        )
        if response.status == 200:
            model_output = response.choices[0].text
            return {"model_output": model_output}
        else:
            raise HTTPException(status_code=response.status, detail=response.error.message)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))