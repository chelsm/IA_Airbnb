
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_model(df):
    X = df[['property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
    y = df['log_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def get_last_model():
    trained_models_dir = './trained_models'
    if not os.path.exists(trained_models_dir):
        raise FileNotFoundError("Le dossier trained_models n'existe pas.")
    
    model_files = os.listdir(trained_models_dir)
    
    pkl_files = []
    for file in model_files:
        if file.endswith('.pkl'):
            pkl_files.append(file)
    
    if not pkl_files:
        raise FileNotFoundError("Aucun fichier .pkl trouvé dans le dossier trained_models.")
    
    pkl_files.sort(key=lambda x: os.path.getmtime(os.path.join(trained_models_dir, x)), reverse=True)
    
    return os.path.join(trained_models_dir, pkl_files[0])


def predict(model, df):
    X_pred = df[['property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds']].values.tolist()
    predictions = model.predict(X_pred)
    return predictions.tolist()