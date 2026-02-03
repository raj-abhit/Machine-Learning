# app.py

import streamlit as st
import pandas as pd
import pickle

# --- Load Model and Features ---
try:
    with open('dt_satisfaction_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('model_features.pkl', 'rb') as file:
        model_columns = pickle.load(file)

except FileNotFoundError:
    st.error("Model files not found. Please ensure 'dt_satisfaction_model.pkl' and 'model_features.pkl' are in the same directory.")
    st.stop()

# --- Streamlit Application Interface ---
st.title("💡 Customer Satisfaction Prediction App")
st.markdown("Predict the likelihood of a ticket leading to **High** or **Low** Customer Satisfaction based on key features.")

st.header("Ticket Feature Input")

# Options based on your dataset (use all unique values from your EDA)
products = ['GoPro Hero', 'LG Smart TV', 'Dell XPS', 'Microsoft Office', 'Xbox', 'PlayStation', 'Dyson Vacuum Cleaner', 'Other/Unlisted']
ticket_types = ['Technical issue', 'Billing inquiry', 'Product inquiry', 'Refund request', 'Cancellation request']
channels = ['Social media', 'Chat', 'Email', 'Phone']

# Create input widgets
product_purchased = st.selectbox("Product Purchased:", options=products)
ticket_type = st.selectbox("Ticket Type:", options=ticket_types)
ticket_channel = st.selectbox("Ticket Channel:", options=channels)

# --- Prediction Logic ---

def predict_satisfaction(product, ticket_type, channel):
    # 1. Create a dummy DataFrame matching the model's features (all zeros)
    data = pd.DataFrame(0, index=[0], columns=model_columns)

    # 2. Set the corresponding one-hot encoded columns to 1
    
    # Product Purchased
    product_col = f'Product Purchased_{product}'
    if product_col in data.columns:
        data[product_col] = 1

    # Ticket Type
    type_col = f'Ticket Type_{ticket_type}'
    if type_col in data.columns:
        data[type_col] = 1

    # Ticket Channel
    channel_col = f'Ticket Channel_{channel}'
    if channel_col in data.columns:
        data[channel_col] = 1

    # 3. Make Prediction
    prediction = model.predict(data)[0]
    
    return prediction

if st.button('Predict Satisfaction'):
    result = predict_satisfaction(product_purchased, ticket_type, ticket_channel)
    
    # Display the result
    if result == 1:
        st.success('**Prediction: HIGH Customer Satisfaction! (Rating 4 or 5) 🎉**')
    else:
        st.warning('**Prediction: LOW Customer Satisfaction. (Rating 1, 2, or 3) 😞**')
        st.caption("This combination of features suggests a higher risk of dissatisfaction based on model insights.")