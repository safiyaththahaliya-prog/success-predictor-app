import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("success_model.pkl")

st.title("Success Predictor App")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict"):

    
    prediction = model.predict([[feature1, feature2, feature3, feature4]])
    st.success(f"Prediction: {prediction[0]}")
