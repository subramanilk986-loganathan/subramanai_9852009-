# Simple Streamlit app for prediction. Run: streamlit run app/main_streamlit.py
import streamlit as st
from src.predict import predict_heart_disease
import json

st.title('Heart Disease Prediction Demo')

st.markdown('Enter patient features as JSON or leave blank fields to use 0. Example: {"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":1}')


user_input = st.text_area('Patient features (JSON/dict)', height=150)
if st.button('Predict'):
    try:
        if not user_input.strip():
            st.warning('Please enter input JSON/dict.')
        else:
            data = json.loads(user_input)
            res = predict_heart_disease(data)
            st.success(f"Prediction: {res['prediction']} | Probability (pos class): {res['probability']}")
            st.json(res)
    except Exception as e:
        st.error(f'Error: {e}')
