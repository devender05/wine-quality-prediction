import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------
# Load Best Saved Model
# -----------------------
MODEL_PATH = "XGBoost.pkl"  # change to your best model

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("🍷 Wine Quality Prediction App (Good or Bad)")
st.write("Enter wine chemical properties to predict if the wine is **Good (1)** or **Bad (0)**")

st.markdown("---")

# -----------------------
# UI Input Fields
# -----------------------
fixed_acidity = st.number_input("Fixed Acidity", value=7.4, format="%.3f")
volatile_acidity = st.number_input("Volatile Acidity", value=0.70, format="%.3f")
citric_acid = st.number_input("Citric Acid", value=0.00, format="%.3f")
residual_sugar = st.number_input("Residual Sugar", value=1.9, format="%.3f")
chlorides = st.number_input("Chlorides", value=0.076, format="%.3f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0, format="%.1f")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0, format="%.1f")
density = st.number_input("Density", value=0.9978, format="%.5f")
pH = st.number_input("pH Level", value=3.51, format="%.2f")
sulphates = st.number_input("Sulphates", value=0.56, format="%.2f")
alcohol = st.number_input("Alcohol", value=9.4, format="%.2f")

# Collect inputs in correct order
sample = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol
}

df_input = pd.DataFrame([sample])

# -----------------------
# Predict Button
# -----------------------
if st.button("Predict Quality"):
    prob = model.predict_proba(df_input)[0][1]
    pred = int(prob >= 0.5)

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    if pred == 1:
        st.success(f"**Good Wine!** 🍷 (Probability = {prob:.4f})")
    else:
        st.error(f"**Bad Wine!** ❌ (Probability = {prob:.4f})")

    st.markdown("---")
    st.write("### Input Data")
    st.table(df_input)

