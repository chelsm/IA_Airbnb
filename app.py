import os
import pickle
import tempfile
import requests
import streamlit as st
import pandas as pd

from api import train_model_endpoint, get_model, predict_endpoint

has_data_clean = False

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

def inverse_categorical(df, column):
    original_values = df[column].unique()
    mapping = {str(i): value for i, value in enumerate(original_values)}
    return mapping

def train_model():
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            cwd = os.getcwd()
            file_path = os.path.join(cwd, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            print('Uploaded file', file_path)
            response = requests.post("http://localhost:8000/training", data={"file_path": file_path})
            if response.status_code == 200:
                st.success("Model trained successfully and saved.")
            else:
                st.error("An error occurred while training the model.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    set_page_config()
    st.title("Airbnb App")
    st.write("This is a simple app that uses the Airbnb API to search for listings in a given city.")

    st.write('------------------------------------------------------------------------------------------------------------------')

    data_csv = load_data("data.csv")
    display_data(data_csv, "Airbnb Data")

    try:
        data_clean_csv = load_data("data_clean.csv")

        if data_clean_csv is not None and not data_clean_csv.empty:
            display_data(data_clean_csv, "Clean Airbnb Data")
            
            st.write('------------------------------------------------------------------------------------------------------------------')

            st.subheader("Train Model")

            train_model()
               


           
        else:
            st.write("No data available.")
    except FileNotFoundError:
        st.write("The file 'data_clean.csv' was not found. Please execute the data cleaning script first.")


if __name__ == "__main__":
    main()
