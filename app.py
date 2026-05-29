import streamlit as st
import joblib

model = joblib.load("success_model.pkl")

st.title("🎬 Movie Success Predictor")

st.write("Predict movie rating based on movie details.")

startYear = st.number_input("Start Year", 1900, 2030, 2020)
runtimeMinutes = st.number_input("Runtime Minutes", 60, 300, 120)
numVotes = st.number_input("Number of Votes", 0, 1000000, 1000)
rank = st.number_input("Movie Rank", 1, 100, 50)

if st.button("Predict"):

    prediction = model.predict([[startYear]])

    st.success(f"Predicted Rating: {prediction[0]:.2f}")
