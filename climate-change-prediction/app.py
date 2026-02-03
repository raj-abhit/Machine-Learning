import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# --- 1. Helper Functions ---

# Function to clean text (must match the cleaning done during training)
def simple_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|@\w+|[^a-z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Function to inverse transform log-transformed prediction
def expm1_predict(y_pred):
    return np.maximum(np.expm1(y_pred), 0)

# --- 2. Model Loading ---

try:
    # Load all saved artifacts
    preprocessor = joblib.load('full_preprocessor_pipeline.pkl')
    regressor = joblib.load('gbr_engagement_regressor.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    lda = joblib.load('lda_model.pkl')
    
    NUM_TOPICS = lda.n_components
    
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'full_preprocessor_pipeline.pkl', 'gbr_engagement_regressor.pkl', 'tfidf_vectorizer.pkl', and 'lda_model.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --- 3. Streamlit Interface (FIXED: SET WIDE LAYOUT) ---

# Set wide layout option to utilize full screen width
st.set_page_config(page_title="Climate Comment Engagement Predictor", layout="wide")

st.title("🌍 Climate Change Comment Engagement Model")
st.markdown("Predict the potential **LikesCount** of a new NASA climate post comment using the trained Gradient Boosting Model. Factors include linguistic, temporal, and topic features.")

# Use containers for clean layout
col_left, col_right = st.columns([1, 2])

# --- User Inputs (Sidebar or Left Column) ---
# We'll keep inputs in the sidebar for cleaner structure

st.sidebar.header("Input New Comment Data")

comment_text = st.sidebar.text_area("1. Comment Text", "The new satellite data confirms the rapid acceleration of sea level rise in the past decade.", height=150)

cols1 = st.sidebar.columns(2)
comments_count = cols1[0].number_input("2. Current Comments Count", min_value=0, value=5, step=1)
hour_of_day = cols1[1].slider("3. Hour of Day (0-23)", min_value=0, max_value=23, value=14)

day_name = st.sidebar.selectbox(
    "4. Day of Week",
    options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    index=2
)

# --- 4. Prediction Logic and Display ---

if st.sidebar.button("Predict Engagement (LikesCount) 🚀"):
    
    # 1. Create Base DataFrame for Structured Features
    input_data = pd.DataFrame({
        'commentsCount': [comments_count],
        'text_length': [len(comment_text)],
        'has_question': [1 if '?' in comment_text else 0],
        'has_exclamation': [1 if '!' in comment_text else 0],
        'hour_of_day': [hour_of_day],
        'date_index': [0], # Placeholder
        'day_name': [day_name]
    })
    
    # 2. Process Text for Topic Features
    cleaned_text = simple_clean_text(comment_text)
    text_vectorized = tfidf.transform([cleaned_text])
    lda_output = lda.transform(text_vectorized)
    topic_cols = [f'Topic_{i+1}_Prob' for i in range(NUM_TOPICS)]
    
    for i, col in enumerate(topic_cols):
        input_data[col] = lda_output[0][i]
        
    # Define feature order as it was during training
    FEATURES_ORDER = ['commentsCount', 'text_length', 'has_question', 'has_exclamation', 'hour_of_day', 'date_index', 'day_name'] + topic_cols
    input_df_ordered = input_data[FEATURES_ORDER]
    
    # 3. Apply Preprocessing Pipeline
    X_processed = preprocessor.transform(input_df_ordered)
    
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out(input_df_ordered.columns)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        else:
            X_processed_df = pd.DataFrame(X_processed)
    except Exception:
        X_processed
    # 4. Make Final Prediction
    log_prediction = regressor.predict(X_processed)[0]
    log_prediction = regressor.predict(X_processed_df)[0]
    
    
    # --- 5. Display Results (Moved to main page for better visibility) ---
    
    st.header("Prediction Results")
    
    # Display the primary result with a large metric
    st.metric(
        label="Predicted Likes Count (Average)", 
        value=f"{final_prediction:.1f} Likes", 
        delta=None
    )
    
    st.subheader("Model Insights")
    
    # Display the most probable topic
    most_likely_topic_index = np.argmax(lda_output[0])
    
    st.markdown(
        f"""
        **Dominant Topic Detected:** **Topic {most_likely_topic_index + 1}** (Probability: {lda_output[0][most_likely_topic_index]:.2f})
        """
    )
    
    st.info("The prediction is based on the influence of the comment's content (Topic), time of post (Hour/Day), and its length.")
    
    # Display the raw input data used
    st.subheader("Raw Input Summary")
    st.dataframe(
        pd.DataFrame({
            'Text Length': [len(comment_text)],
            'Has Question': ['Yes' if '?' in comment_text else 'No'],
            'Posted Hour': [f'{hour_of_day}:00'],
            'Posted Day': [day_name]
        }),
        hide_index=True
    )
else:
    st.info("👈 Enter the comment details in the sidebar and click the 'Predict' button to see the results.")