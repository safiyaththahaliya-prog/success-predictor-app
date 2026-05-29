import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load("success_model.pkl")

st.title("🎬 Success Predictor")

rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0)

votes = st.number_input("Number of Votes", min_value=0)

if st.button("Predict"):

    data = np.array([[rating, votes]])

    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")
