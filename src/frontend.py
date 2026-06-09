import streamlit as st
import requests

st.title("Amazon Product Recommender")

product_title = st.text_input("Enter product name")

if st.button("Recommend"):

    params = {
        "product_title": product_title,
        "top_k": 10
    }

    st.write("Sending:", params)

    response = requests.get(
        "http://backend:8000/recommend",
        params=params
    )

    st.write("Status Code:", response.status_code)
    st.write("Response:", response.text)

    if response.status_code == 200:
        st.json(response.json())