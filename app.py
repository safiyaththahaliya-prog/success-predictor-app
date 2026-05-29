import streamlit as st
import joblib
import numpy as np

model = joblib.load("success_model.pkl")

st.title("Movie Success Predictor")

feature1 = st.number_input("Enter startYear")

if st.button("Predict"):
    prediction = model.predict([[feature1]])
    st.success(f"Predicted Rating: {prediction[0]}")
