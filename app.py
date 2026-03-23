import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# -----------------------------
# Page Title
# -----------------------------
st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate.")

# -----------------------------
# Load model artifacts
# -----------------------------
model = tf.keras.models.load_model("artifacts/housing_model.h5", compile=False)
scaler = joblib.load("artifacts/scaler.pkl")
features = joblib.load("artifacts/feature_names.pkl")

# -----------------------------
# User Inputs
# -----------------------------
acres = st.number_input(
    "Land area (acres)", min_value=0.01, max_value=20.0, value=0.25, step=0.01
)

land_value = st.number_input(
    "Land value ($)", min_value=1000, max_value=1000000, value=50000, step=1000
)

build_value = st.number_input(
    "Building value ($)", min_value=1000, max_value=2000000, value=200000, step=5000
)

yard_value = st.number_input(
    "Yard items value ($)", min_value=0, max_value=500000, value=0, step=1000
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame({
        "CALC_ACRES": [acres],
        "LAND_VALUE": [land_value],
        "BUILD_VALUE": [build_value],
        "YARDITEMS_VALUE": [yard_value]
    })

    input_scaled = scaler.transform(input_df)

    pred_value = model.predict(input_scaled, verbose=0)[0][0]

    pred_value = pred_value * 10000
    
    st.success(f"Estimated appraised value: ${pred_value:,.0f}")
