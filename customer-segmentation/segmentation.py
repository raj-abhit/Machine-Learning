import streamlit as st
import pandas as pd
import numpy as np
import joblib
kmeans = joblib.load('kmeans_model.pkl')
scaler=joblib.load('scaler.pkl')
st.title('Customer Segmentation App')
st.write('Enter customer details to predict the')

age = st.number_input('Age',min_value = 18,max_value = 35)
income =st.number_input('Income',min_value = 0,max_value = 200000,value = 50000)
total_spending = st.number_input('Total Spending(sum of putchases)',min_value = 0,max_value = 5000,value = 1000)
num_web_purchases = st.number_input('Number of Web Purchases',min_value = 0 , max_value = 100,value = 10)
num_store_purchases = st.number_input('Number of Store Purchases',min_value = 0,max_value = 100,value = 10)
num_web_visits = st.number_input('Number of Web Visits',min_value = 0 ,max_value = 100,value = 10)
recency = st.number_input('Recency(days since last purchase)',min_value = 0 ,max_value =365,value = 30)


input_data = pd.DataFrame({
    "Age":[age],
    "Income":[income],
    "Total_Spending":[ total_spending],
    "NumWebPurchases":[num_web_purchases],
    "NumStorePurchases":[num_store_purchases],
    "NumWebVisitsMonth":[num_web_visits],
    "Recency":[recency]
})

input_scaled = scaler.transform(input_data)

if st.button("predict Segment"):

    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment:Cluster {cluster}")

    st.write("""
**Cluster Descriptions**

- Cluster 0: Older customers with high income and high spending, prefer store shopping over web, low web activity  
- Cluster 1: Middle-aged, low income and low spending, high recency, frequent web visits  
- Cluster 2: Youngest and wealthiest group, very high spending, prefers stores, minimal web visits  
- Cluster 3: Younger, low income and low spending, very recent purchases, mostly web-focused  
- Cluster 4: Mid-age, moderate income and spending, highest web purchases and visits, balanced shopper  
""")

   
    

                                