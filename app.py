import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('soil_model_clean.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("Soil Quality Prediction")

# Input fields
N = st.number_input("Nitrogen", 0.0)
P = st.number_input("Phosphorus", 0.0)
K = st.number_input("Potassium", 0.0)
pH = st.number_input("pH", 0.0)
EC = st.number_input("Electrical Conductivity", 0.0)
OC = st.number_input("Organic Carbon", 0.0)
Zn = st.number_input("Zinc", 0.0)
Fe = st.number_input("Iron", 0.0)
Cu = st.number_input("Copper", 0.0)
Mn = st.number_input("Manganese", 0.0)
B = st.number_input("Boron", 0.0)
S = st.number_input("Sulphur", 0.0)

if st.button("Predict"):
    input_data = np.array([[N, P, K, pH, EC, OC, Zn, Fe, Cu, Mn, B, S]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    label = encoder.inverse_transform(prediction)
    st.success(f"Soil Quality: {label[0]}")
