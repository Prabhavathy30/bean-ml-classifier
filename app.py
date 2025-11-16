import streamlit as st
import numpy as np
import pickle

st.title("🌱 Bean Type Classifier")
st.write("This app predicts the type of bean based on the input features.")

# ------------------------------
# Load the saved model
# ------------------------------
@st.cache_resource
def load_model():
    with open("bean_model_ml.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model

model = load_model()

# ------------------------------
# Feature Inputs
# ------------------------------

st.header("Enter Bean Features")

# ⚠️ Replace these with your dataset's actual column names
feature_names = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
    "Eccentricity",
    "Solidity",
    "Extent",
    "AspectRatio",
    "Roundness"
]

inputs = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

input_array = np.array(inputs).reshape(1, -1)

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    st.success(f"### ✅ Predicted Bean Type: **{prediction}**")
