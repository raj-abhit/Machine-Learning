import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

# ------------------------------------------------------------
# 🧩 Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Coca-Cola Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Coca-Cola Stock Analysis & Prediction")
st.markdown("""
This interactive app uses a trained machine learning model to predict the **next-day closing price**  
of Coca-Cola's stock based on recent historical and technical indicators.
""")

# ------------------------------------------------------------
# 📦 Load Model & Feature Info
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        pipe = joblib.load("best_model_pipeline.pkl")
        with open("feature_order.json", "r") as f:
            features = json.load(f)
        return pipe, features
    except Exception as e:
        st.error("⚠️ Model files not found. Please train and save the model first.")
        st.stop()

pipe, FEATURES = load_model()

# ------------------------------------------------------------
# 📊 User Input Section
# ------------------------------------------------------------
st.subheader("🔧 Input Features")

# Generate input widgets dynamically
user_data = {}
for feature in FEATURES:
    if "lag" in feature or "Return" in feature or "Volatility" in feature:
        val = st.number_input(f"{feature}", value=0.0, step=0.01)
    elif "MA" in feature or "EMA" in feature:
        val = st.number_input(f"{feature}", value=0.0, step=0.1)
    elif feature.lower() == "volume":
        val = st.number_input(f"{feature}", value=10000000)
    else:
        val = st.number_input(f"{feature}", value=60.0, step=0.1)
    user_data[feature] = val

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# ------------------------------------------------------------
# 🚀 Prediction
# ------------------------------------------------------------
if st.button("Predict Next-Day Closing Price"):
    try:
        prediction = pipe.predict(input_df)[0]
        st.success(f"📈 Predicted Next-Day Closing Price: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# ------------------------------------------------------------
# 📉 Optional Section: Insights
# ------------------------------------------------------------
st.markdown("---")
st.subheader("💡 Model Info")
st.write(f"**Total features used:** {len(FEATURES)}")
st.json(FEATURES)
st.caption("Model: trained using RandomForest/GradientBoosting pipelines (best selected automatically).")

# ------------------------------------------------------------
# 🕒 Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Developed by **Abhit Raj**"
)
