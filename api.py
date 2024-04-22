import os
import pickle
import uuid

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from function import calculate_mse, train_model, get_last_model, predict

client = OpenAI(
    api_key="sk-proj-YTmJg1pIAgUQpoPAWHpST3BlbkFJodX2M41av2Ib6lO9yAGI",
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
        predictions = predict(model, df)
        
        actual_values = df['log_price'].tolist() 
        mse = calculate_mse(actual_values, predictions)
        
        return {"predicted prices": predictions, "mse": mse}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # Attend les données JSON avec les mêmes caractéristiques que le modèle attend
#     # Préparer les données pour la prédiction
#     features = [[
#         data['property_type'],
#         data['room_type'],
#         data['accommodates'],
#         data['bathrooms'],
#         data['bedrooms'],
#         data['beds']
#     ]]
#     # Faire la prédiction
#     prediction = model.predict(features)
#     # Retourner la prédiction
#     return jsonify({'predicted_price': prediction[0]})

@app.get("/model", tags=["Models"], summary="Obtenir un modèle d'IA", description="Endpoint pour obtenir un modèle d'IA de l'API OpenAI.")
def get_model(prompt: str):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
            ]
        )

        if completion.status == 200:
            model_output = completion.choices[0].text
            return {"model_output": model_output}
        else:
            raise HTTPException(status_code=completion.status, detail=completion.error.message)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))