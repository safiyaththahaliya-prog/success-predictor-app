import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("success_model.pkl")

st.title("🎬 Success Predictor")

rating = st.slider("Movie Rating", 0.0, 10.0, 5.0)

votes = st.number_input("Votes", min_value=0)

if st.button("Predict"):

    data = np.array([[rating, votes]])

    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")
