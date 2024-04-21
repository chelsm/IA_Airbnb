import streamlit as st
import requests

st.set_page_config(
    page_title="Streamlit Airbnb App",
    page_icon=":home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Airbnb Price Prediction App")

if st.checkbox("Show data"):
    st.subheader("Data")
    response = requests.get("http://localhost:8000/city")
    data = response.json()
    st.write(data)

