# streamlit_app.py
import streamlit as st
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "wine_cultivar_model.pkl")

st.title("Wine Cultivar Predictor")

st.write("Enter the chemical properties of the wine sample:")

# The six features we used
FEATURES = ['alcohol','malic_acid','ash','alcalinity_of_ash','total_phenols','flavanoids']

# make inputs
inputs = {}
for feat in FEATURES:
    # default values chosen small; user can change
    inputs[feat] = st.number_input(feat.replace('_', ' ').title(), format="%.6f", step=0.1)

# Show debugging if needed
if st.checkbox("Show model path and files (debug)"):
    st.write("Model path:", MODEL_PATH)
    st.write("Files in repo dir:", os.listdir(BASE_DIR))
    model_dir = os.path.join(BASE_DIR, "model")
    st.write("Files in model dir:", os.listdir(model_dir) if os.path.exists(model_dir) else "model dir not found")

if st.button("Predict"):
    # Load model safely with helpful error
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Make sure model/wine_cultivar_model.pkl exists in the repo and is committed.")
    else:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed loading model: {e}")
        else:
            arr = np.array([inputs[f] for f in FEATURES], dtype=float).reshape(1, -1)
            pred = model.predict(arr)[0] + 1  # map 0->1 etc
            st.success(f"Predicted cultivar: Cultivar {pred}")
