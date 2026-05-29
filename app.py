import streamlit as st

st.title("🎬 Success Predictor")

rating = st.slider("Movie Rating", 0.0, 10.0, 5.0)

votes = st.number_input("Votes", min_value=0)

if st.button("Predict"):
    st.success("App is working successfully!")
