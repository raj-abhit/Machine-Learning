# 🧠 Customer Segmentation using K-Means Clustering

# 📘 Introduction
In an increasingly data-driven business environment, understanding customer behavior is critical to delivering personalized services and optimizing marketing strategies. Raw customer data often contains hidden patterns that, when uncovered, can significantly enhance business decision-making.

This project tackles this challenge by applying Unsupervised Machine Learning techniques—specifically K-Means Clustering—to segment customers based on their purchasing behaviors and web activity. Through this segmentation, businesses can target specific customer groups more effectively, improve customer engagement, and increase overall profitability.

# 🎯 Objectives
The main objectives of this project are:

Segment customers into meaningful clusters based on their behavioral and transactional data.

Identify high-value customers, frequent web visitors, or customers with high spending potential.

Use visualization and dimensionality reduction (PCA) to interpret and analyze the clusters.

Provide actionable insights based on customer segmentation for marketing and business strategy.

# ⚙️ Technologies & Libraries
This project leverages the following technologies:

Python 3.x

Pandas – Data manipulation and preprocessing

NumPy – Numerical computations

Matplotlib & Seaborn – Data visualization

Scikit-learn – KMeans clustering, PCA, preprocessing, model evaluation

Streamlit – Web application framework for interactive dashboard

Joblib – Saving/loading the trained clustering model and pipeline

# 🧪 Methodology
Data Preprocessing:

Handle missing values, normalization, and encoding.

Standardize data to ensure fair clustering.

K-Means Clustering:

Optimal number of clusters determined using the Elbow Method (WCSS).

Apply K-Means to segment the dataset into distinct groups.

Dimensionality Reduction:

Use Principal Component Analysis (PCA) for 2D visualization of clusters.

Model Deployment:

A Streamlit-based GUI is built for interactive analysis and visualization of clusters.

# 📊 Cluster Evaluation
Each customer segment is labeled and interpreted based on feature patterns:

Cluster 0: High budget, frequent web visitors

Cluster 1: High spending customers

Cluster 2: Budget-conscious, web-active users

Cluster 3: Infrequent buyers with low web interaction

Cluster 4: Moderate spenders, occasional web users

These interpretations help businesses tailor marketing strategies and customer engagement plans.

# 📈 Visualization
Elbow plot to identify optimal number of clusters

PCA scatter plots for visualizing clusters in reduced dimensions

Streamlit-based interactive dashboard for real-time analysis

# 📂 Files & Structure
segmentation.py: Main application file with clustering and Streamlit code

customer_segmentation.ipynb: Data Analysis and Machine Learning code

customer_data.csv: Dataset used for clustering

model.pkl: Saved KMeans model and preprocessing pipeline

README.md: Project overview and usage guide

