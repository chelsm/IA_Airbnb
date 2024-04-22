import os
import requests
import streamlit as st
import pandas as pd


def set_page_config():
    st.set_page_config(
        page_title="Streamlit Airbnb App",
        page_icon="https://cdn-icons-png.flaticon.com/512/2111/2111320.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data

@st.cache_data
def load_data(file_path):
    return load_data_from_csv(file_path)

def display_data(data, title):
    st.subheader(title)

    toggle_button = st.checkbox(f"Display data from {title}")

    if toggle_button:
        st.write(f"Here are the {title} data to display:")
        st.write(data)

        selected_columns = st.multiselect(f"Select columns to display for {title}:", data.columns)
        if selected_columns:
            st.write(data[selected_columns])

        if st.checkbox(f"Display descriptive statistics for {title}"):
            st.write(f"Descriptive statistics for {title}:")
            st.write(data.describe())

def inverse_categorical(column):
    df = pd.read_csv("data.csv")
    original_values = df[column].unique()
    mapping = {str(i): value for i, value in enumerate(original_values)}
    return mapping

def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  

def train_model_json():
    data_df = pd.read_csv("data_clean.csv")
    
    property_types_mapping = inverse_categorical("property_type")
    room_types_mapping = inverse_categorical("room_type")

    print('property_types_mapping', property_types_mapping)

    property_types = list(property_types_mapping.values())
    room_types = list(room_types_mapping.values())
    max_bathrooms = int(data_df["bathrooms"].max())
    max_accommodates = int(data_df["accommodates"].max())
    max_bedrooms = int(data_df["bedrooms"].max())
    max_beds = int(data_df["beds"].max())

    property_type = st.selectbox("Property type", property_types)
    room_type = st.selectbox("Room type", room_types)
    bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=max_bathrooms, step=1)
    accommodates = st.number_input("Number of accommodates", min_value=1, max_value=max_accommodates, step=1)
    bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=max_bedrooms, step=1)
    beds = st.number_input("Number of beds", min_value=1, max_value=max_beds, step=1)

    property_type_key = get_key_from_value(property_types_mapping, property_type)
    room_type_key = get_key_from_value(room_types_mapping, room_type)

    data = {
        "property_type": property_type_key,
        "room_type": room_type_key,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
    }

    
    if st.button("Train Model"):
        response = requests.post("http://localhost:8000/predict-json", json=data)
        if response.status_code == 200:
            st.success("Model trained successfully and saved.")
            st.write("Predicted prices:", response.json()["predicted_prices"])
        else:
            st.error("An error occurred while training the model.")


def home_page():
    data_csv = load_data("data.csv")
    display_data(data_csv, "Airbnb Data")
    try:
        data_clean_csv = load_data("data_clean.csv")
        if data_clean_csv is not None and not data_clean_csv.empty:
            display_data(data_clean_csv, "Clean Airbnb Data")
        else:
            st.write("No data available.")
    except FileNotFoundError:
        st.write("The file 'data_clean.csv' was not found. Please execute the data cleaning script first.")


def swagger_page():
    st.write('------------------------------------------------------------------------------------------------------------------')
    st.subheader("Swagger Documentation")
    st.write("Swagger documentation:")
    st.components.v1.iframe("http://localhost:8000/docs", height=1000, scrolling=True)

def predict_page():
    st.write('------------------------------------------------------------------------------------------------------------------')
    st.subheader("Predict Prices")
    st.write("Predict prices using the trained model & JSON data.")
    train_model_json()


def main():
    set_page_config()
    st.title("Airbnb App")
    st.write("This is a app that predicts prices using the Airbnb API.")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Swagger Documentation", "Predict Prices"])

    if selection == "Home":
        home_page()
    elif selection == "Swagger Documentation":
        swagger_page()
    elif selection == "Predict Prices":
        predict_page()



if __name__ == "__main__":
    main()