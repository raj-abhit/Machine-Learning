import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Helper Functions (Must match the training script logic) ---

# Function to inverse transform log-transformed prediction
def expm1_predict(y_pred_log):
    return np.maximum(np.expm1(y_pred_log), 0)

# Function to simplify OS (Cell 2, step 1)
def simplify_os(os_name):
    if 'Windows' in os_name: return 'Windows'
    elif 'macOS' in os_name or 'Mac OS' in os_name: return 'Mac'
    elif 'Linux' in os_name: return 'Linux'
    else: return 'Other/No OS'

# Function to extract CPU Type (Cell 2, step 2)
def extract_cpu_type(model):
    if 'i7' in model: return 'i7'
    elif 'i5' in model: return 'i5'
    elif 'i3' in model: return 'i3'
    elif 'Celeron' in model or 'Pentium' in model or 'AMD' in model: return 'Low-End'
    else: return 'Other'

# --- 2. Model Loading (Caching for performance) ---

@st.cache_resource
def load_models():
    """Loads the preprocessor and the Gradient Boosting Regressor model."""
    try:
        preprocessor = joblib.load('full_preprocessor_laptop.pkl')
        regressor = joblib.load('gbr_laptop_price_regressor.pkl')
        
        # Get feature names from the preprocessor to ensure input DataFrame matches
        categorical_features_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
            ['Company', 'TypeName', 'Screen', 'Touchscreen', 'IPSpanel', 
             'RetinaDisplay', 'CPU_company', 'PrimaryStorageType', 'OS_simplified', 'CPU_Type']
        )
        
        # Define the final feature order (MUST MATCH X_final.columns from training)
        features_order = [
            'Inches', 'Ram', 'Weight', 'CPU_freq', 'Total_Storage_GB'
        ] + list(categorical_features_names) + [
            'High_Res'
        ]
        
        return preprocessor, regressor, features_order
    except FileNotFoundError:
        st.error("Error: Model or Preprocessor files ('full_preprocessor_laptop.pkl' or 'gbr_laptop_price_regressor.pkl') not found. Please ensure they are in the same directory.")
        return None, None, None

preprocessor, regressor, FEATURES_ORDER = load_models()

# --- 3. Streamlit UI and Input Collection ---

def main():
    st.set_page_config(
        page_title="Laptop Price Predictor",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("💻 Laptop Price Prediction App")
    st.markdown("Use the best performing Gradient Boosting Model to predict laptop prices in Euros.")

    if preprocessor is None or regressor is None:
        return

    # --- Sidebar for Input ---
    with st.sidebar:
        st.header("Laptop Specifications")
        
        # CATEGORICAL FEATURES
        company = st.selectbox("Company", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Samsung', 'Razer', 'Other'], index=0)
        type_name = st.selectbox("Type Name", ['Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'], index=0)
        ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
        os_input = st.selectbox("Operating System", ['Windows 10', 'macOS', 'Linux', 'No OS', 'Android', 'Chrome OS', 'Windows 7'], index=0)
        
        screen_type = st.selectbox("Screen Type", ['Standard', 'Full HD', '4K'], index=1)
        touchscreen = st.checkbox("Touchscreen", False)
        ips_panel = st.checkbox("IPS Panel", False)
        retina_display = st.checkbox("Retina Display", False)
        
        # CPU Info (Simpler inputs needed for feature engineering)
        cpu_model_input = st.selectbox("CPU Family", ['Core i7', 'Core i5', 'Core i3', 'Celeron', 'AMD A-Series', 'Other Intel/AMD'], index=0)
        cpu_company = st.selectbox("CPU Company", ['Intel', 'AMD', 'Samsung'], index=0)
        
        # STORAGE
        primary_storage = st.number_input("Primary Storage (GB)", min_value=32, max_value=2048, value=256, step=32)
        secondary_storage = st.number_input("Secondary Storage (GB, often 0)", min_value=0, max_value=2048, value=0, step=32)
        primary_storage_type = st.selectbox("Primary Storage Type", ['SSD', 'HDD', 'Flash Storage', 'Hybrid'], index=0)


    # --- Main Area for NUMERICAL Input ---
    st.header("Hardware Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.86, step=0.01)
    with col3:
        cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=1.0, max_value=4.0, value=2.5, step=0.1)
    
    # --- Screen Resolution Input for 'High_Res' Feature ---
    st.subheader("Screen Resolution")
    col4, col5 = st.columns(2)
    with col4:
        screen_w = st.number_input("Screen Width (pixels)", min_value=1024, max_value=3840, value=1920, step=100)
    with col5:
        screen_h = st.number_input("Screen Height (pixels)", min_value=768, max_value=2400, value=1080, step=100)
        
    
    # --- Prediction Button ---
    if st.button("Predict Price", type="primary"):
        
        # --- 4. Feature Engineering (Replicating Cell 2) ---
        
        # 4.1 Create a raw input DataFrame (matching the structure BEFORE preprocessing)
        raw_data = {
            'Company': company,
            'TypeName': type_name,
            'Inches': inches,
            'Ram': ram,
            'OS': os_input, # Used for simplification
            'Weight': weight,
            'Screen': screen_type,
            'ScreenW': screen_w,
            'ScreenH': screen_h,
            'Touchscreen': 'Yes' if touchscreen else 'No',
            'IPSpanel': 'Yes' if ips_panel else 'No',
            'RetinaDisplay': 'Yes' if retina_display else 'No',
            'CPU_company': cpu_company,
            'CPU_freq': cpu_freq,
            'CPU_model': cpu_model_input, # Used for type extraction
            'PrimaryStorage': primary_storage,
            'SecondaryStorage': secondary_storage,
            'PrimaryStorageType': primary_storage_type
            # Dropped: 'GPU_model', 'Product', 'SecondaryStorageType', 'GPU_company'
        }
        
        input_df = pd.DataFrame([raw_data])
        
        # 4.2 Apply Feature Engineering steps
        input_df['OS_simplified'] = input_df['OS'].apply(simplify_os)
        input_df['CPU_Type'] = input_df['CPU_model'].apply(extract_cpu_type)
        input_df['Total_Storage_GB'] = input_df['PrimaryStorage'] + input_df['SecondaryStorage']
        input_df['High_Res'] = ((input_df['ScreenW'] >= 1920) | (input_df['ScreenH'] >= 1080)).astype(int)

        # 4.3 Select only the columns the preprocessor expects (matching X in Cell 4)
        X_test_raw = input_df[['Inches', 'Ram', 'Weight', 'CPU_freq', 'Total_Storage_GB',
                               'Company', 'TypeName', 'Screen', 'Touchscreen', 'IPSpanel', 
                               'RetinaDisplay', 'CPU_company', 'PrimaryStorageType', 
                               'OS_simplified', 'CPU_Type', 'High_Res']]
        
        # 4.4 Apply Preprocessor
        # Note: The preprocessor handles the remaining numerical/scaling/OHE internally
        
        try:
            X_processed = preprocessor.transform(X_test_raw)
        except ValueError as e:
            st.error(f"Preprocessing Error: {e}. The input data structure might not match the trained model's expected features.")
            st.dataframe(X_test_raw)
            return

        
        # --- 5. Make Final Prediction ---
        log_prediction = regressor.predict(X_processed)[0]
        final_prediction = expm1_predict(log_prediction)
        
        
        # --- 6. Display Results ---
        
        st.success("Prediction Complete!")
        
        st.metric(
            label="Predicted Laptop Price", 
            value=f"€{final_prediction:,.2f}", 
            delta=None,
            help="The prediction is based on the Gradient Boosting Regressor, inverse-transformed from the log scale."
        )

        st.info(f"R-squared of the model: 0.87. Mean Absolute Error: €188.42.")
        
        st.markdown("""
        ### Feature Summary
        - **RAM:** The most important feature for price. You selected **{ram}GB**.
        - **CPU Type:** You selected **{cpu_type}**.
        - **Total Storage:** You selected **{total_storage}GB** ({primary_type}).
        - **High Resolution:** {high_res_status}.
        """.format(
            ram=ram,
            cpu_type=input_df['CPU_Type'].iloc[0],
            total_storage=input_df['Total_Storage_GB'].iloc[0],
            primary_type=primary_storage_type,
            high_res_status="Yes" if input_df['High_Res'].iloc[0] == 1 else "No"
        ))


if __name__ == "__main__":
    main()
