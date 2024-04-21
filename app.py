import streamlit as st
import pandas as pd

has_data_clean = False

def set_page_config():
    st.set_page_config(
        page_title="Streamlit Airbnb App",
        page_icon=":home:",
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

            st.subheader("Get Airbnb Listings from API")

            city = st.text_input("Enter a city name:", "NYC")
            submit = st.button("Search")

            if submit:
                filtered_data = data_clean_csv[data_clean_csv['city'] == city]
                if not filtered_data.empty:
                    st.write("**Here are the top 10 listings:**")
                    for index, row in filtered_data.head(10).iterrows():
                        property_type = inverse_categorical(data_csv, 'property_type').get(str(row['property_type']), 'Unknown')
                        name = row['name']
                        price = row['log_price']
                        currency = '$'
                        st.write(f"**Name:** {name}")
                        st.write(f"- Type: {property_type}")
                        st.write(f"- Price: {price} {currency}")
                else:
                    st.write("No listings found for the specified city.")
        else:
            st.write("No data available.")
    except FileNotFoundError:
        st.write("The file 'data_clean.csv' was not found. Please execute the data cleaning script first.")


if __name__ == "__main__":
    main()
