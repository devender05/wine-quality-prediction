import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
# UI Input Fields (Single Prediction)
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

# Collect single inputs
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

df_single = pd.DataFrame([sample])

# -----------------------
# CSV Upload for Batch Prediction
# -----------------------
uploaded_file = st.file_uploader("Or upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    expected_cols = list(sample.keys())
    if not all(col in df_input.columns for col in expected_cols):
        st.error(f"CSV must have the following columns: {', '.join(expected_cols)}")
        st.stop()
else:
    df_input = df_single

# -----------------------
# Graph of Inputted CSV Data Only
# -----------------------
st.markdown("---")
st.subheader("📊 Graph of Inputted CSV Data")

# Transpose for easier plotting (samples as x, features as lines)
df_plot = df_input.T.reset_index()
df_plot.columns = ['Feature'] + [f"Sample_{i+1}" for i in range(df_plot.shape[1]-1)]
df_plot = df_plot[1:].reset_index(drop=True)  # Skip index row

fig, ax = plt.subplots(figsize=(10, 6))
df_plot.plot(x='Feature', ax=ax)
ax.set_title('Wine Chemical Properties Across Samples')
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Samples')
st.pyplot(fig)

# Optional: Correlation Heatmap for multi-feature insight
if len(df_input) > 1:
    st.write("### Correlation Heatmap of Features")
    corr = df_input.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax2)
    st.pyplot(fig2)

# -----------------------
# Predict Button (Kept for completeness, but graph is independent)
# -----------------------
if st.button("Predict Quality"):
    predictions = model.predict(df_input)
    probabilities = model.predict_proba(df_input)[:, 1]
    
    df_input = df_input.copy()
    df_input['prediction'] = predictions
    df_input['probability'] = probabilities

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    if len(df_input) == 1:
        pred = predictions[0]
        prob = probabilities[0]
        if pred == 1:
            st.success(f"**Good Wine!** 🍷 (Probability = {prob:.4f})")
        else:
            st.error(f"**Bad Wine!** ❌ (Probability = {prob:.4f})")
    else:
        good_count = (predictions == 1).sum()
        total_count = len(df_input)
        st.info(f"Out of {total_count} wines, {good_count} are predicted as **Good** and {total_count - good_count} as **Bad**.")

    st.markdown("---")
    st.write("### Input Data and Predictions")
    st.dataframe(df_input)

    if len(df_input) > 1:
        st.write("### Prediction Distribution")
        st.bar_chart(df_input['prediction'].value_counts().sort_index())

