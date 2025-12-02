import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = load_model('battery_life_model.h5')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
        
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# App Title
st.title("EV Battery Life Predictor using ANN")

# User Inputs
st.header("Enter Battery Parameters")

discharge_type = st.selectbox(
    "Discharge Type",
    ('charge', 'discharge', 'impedance')
)

capacity = st.number_input("Capacity", value=0.0, format="%.6f")
re = st.number_input("Re (Impedance Real Part)", value=0.0, format="%.6e")
rct = st.number_input("Rct (Charge Transfer Resistance)", value=0.0, format="%.6e")

# Prediction Logic
if st.button("Predict"):
    try:
        # Encode categorical input
        type_encoded = label_encoder.transform([discharge_type])[0]
        
        # Create input array
        input_data = np.array([[type_encoded, capacity, re, rct]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Display result
        st.success(f"Predicted Battery Life (Ambient Temperature): {prediction[0][0]:.4f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
