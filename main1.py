import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any, Tuple, Union

# Suppress warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- CONFIGURATION CONSTANTS ---
MODEL_PATH = '7_logistic_model.pkl'
DATA_PATH = '7_churn.csv'

# Define the columns (SeniorCitizen handled separately below for robustness)
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'Contract'
]
NUMERICAL_COLS = ['tenure', 'TotalCharges']
FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'Contract', 'TotalCharges'
]

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(
    page_title="Awesome Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. ASSET LOADING & PREPROCESSOR FITTING (CRITICAL FIX) ---
@st.cache_resource(show_spinner="Initializing Model & Preprocessors...")
def load_and_fit_preprocessors():
    """Loads the model and fits the LabelEncoders/StandardScaler on the training data."""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Data Cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True) 
        
        # 1. Fit Encoders
        encoders = {}
        # Include SeniorCitizen here to catch both '0'/'1' or 'Yes'/'No' if it was treated categorically
        for col in CATEGORICAL_COLS + ['SeniorCitizen']:
            le = LabelEncoder()
            # Convert to string to ensure consistency for LabelEncoder fitting
            df[col] = df[col].astype(str)
            le.fit(df[col])
            encoders[col] = le

        # 2. Fit Scaler
        scaler = StandardScaler()
        scaler.fit(df[NUMERICAL_COLS])

        # 3. Load Model
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
            
        return model, encoders, scaler
        
    except Exception as e:
        st.error(f"Error loading required assets ({MODEL_PATH} or {DATA_PATH}): {e}")
        st.stop()

# Load assets globally
try:
    model, encoders, scaler = load_and_fit_preprocessors()
except:
    st.stop()


# --- 2. PREDICTION FUNCTION (FIXED) ---
def get_prediction(user_input_data: Dict[str, Any]) -> Tuple[Union[int, None], Union[np.ndarray, None]]:
    """Processes input and returns the prediction and probability."""
    
    df_input = pd.DataFrame([user_input_data])

    # 1. Transform Categorical Features
    for col in CATEGORICAL_COLS + ['SeniorCitizen']:
        try:
            # Convert to string to match the fitted encoder's data type
            df_input[col] = df_input[col].astype(str)
            # Use .transform() only, NOT .fit_transform()
            df_input[col] = encoders[col].transform(df_input[col])
        except Exception:
            st.error(f"Error: Unknown value found in '{col}'. Check your input options.")
            return None, None
        
    # 2. Scale Numerical Features
    df_input[NUMERICAL_COLS] = scaler.transform(df_input[NUMERICAL_COLS])
    
    # 3. Ensure correct feature order
    df_input = df_input[FEATURE_ORDER]

    # 4. Predict
    prediction_result = model.predict(df_input)[0]
    # Predict_proba returns [prob_stay, prob_churn]
    probability = model.predict_proba(df_input)[0] 
    
    return int(prediction_result), probability


# --- 3. UI/UX FUNCTIONS ---

# Tips data moved outside the function for cleanliness
CHURN_TIPS = [
    "1. Offer a **Two-Year Contract** incentive to lock in loyalty.",
    "2. **Personalized Check-in:** Call to address specific pain points.",
    "3. **Loyalty Bonus:** Offer a bill credit or free service upgrade.",
    "4. **Monitor Engagement:** Flag for immediate follow-up by retention team."
]

RETENTION_TIPS = [
    "1. **Reward Loyalty:** Send a 'Thank You' or bonus on their anniversary.",
    "2. **Proactive Check-in:** Ensure ongoing satisfaction and offer early access.",
    "3. **Simplify Billing:** Make account management seamless and easy.",
    "4. **Value Confirmation:** Highlight the benefits they currently receive."
]

def build_sidebar():
    """Gathers user input in a clean, scrollable sidebar using tabs."""
    st.sidebar.title("⚙️ Customer Data Input")
    st.sidebar.markdown("---")

    tab_profile, tab_service = st.sidebar.tabs(["👤 Profile & Loyalty", "📞 Service Details"])
    
    with tab_profile:
        st.subheader("Customer Profile")
        gender = st.selectbox("🚻 Gender", options=['Female', 'Male'])
        # SeniorCitizen is handled here
        SeniorCitizen = st.selectbox("👴 Senior Citizen", options=['0', '1'], format_func=lambda x: 'Yes' if x == '1' else 'No')
        Partner = st.selectbox("❤️ Partner Status", options=['Yes', 'No'])
        Dependents = st.selectbox("👨‍👩‍👧 Dependents Status", options=['Yes', 'No'])
        
        tenure = st.slider("🗓️ Tenure (months)", min_value=0, max_value=72, value=12)
        TotalCharges = st.number_input("💰 Total Charges ($)", min_value=0.0, value=1000.0, step=50.0)

    with tab_service:
        st.subheader("Contract & Service")
        Contract = st.selectbox("📄 Contract Type", options=['Month-to-month', 'One year', 'Two year'], help="Month-to-month plans carry the highest risk.")
        PhoneService = st.selectbox("☎️ Phone Service", options=['Yes', 'No'])
        MultipleLines = st.selectbox("📞 Multiple Lines", options=['Yes', 'No', 'No phone service'])

    user_data = {
        'gender': gender, 
        'SeniorCitizen': SeniorCitizen, # Passes '0' or '1'
        'Partner': Partner, 
        'Dependents': Dependents,
        'tenure': tenure, 
        'PhoneService': PhoneService, 
        'MultipleLines': MultipleLines, 
        'Contract': Contract, 
        'TotalCharges': TotalCharges
    }
    
    return user_data

def display_results(result, proba):
    """Displays highly visual and actionable prediction results."""
    
    churn_probability = proba[1] * 100
    
    st.header("⚡ Prediction & Risk Analysis")
    st.markdown("---")

    col_status, col_gauge = st.columns([1, 2])

    with col_status:
        if result == 1:
            st.error(f"### 🚨 HIGH RISK: CHURN")
            st.metric(label="Predicted Status", value="Likely to Leave", delta=f"{churn_probability:.1f}% Risk", delta_color="inverse")
        else:
            st.success(f"### ✅ LOW RISK: RETAINED")
            st.metric(label="Predicted Status", value="Likely to Stay", delta=f"{100-churn_probability:.1f}% Certainty", delta_color="normal")
            
    with col_gauge:
        st.markdown(f"**Overall Churn Risk Score: {churn_probability:.1f}%**")
        st.progress(churn_probability / 100)
        st.markdown(f"*The model predicts a **{churn_probability:.1f}% chance** of the customer canceling.*")

    st.markdown("---")

    # Actionable Tips Section
    st.subheader("🛠️ Recommended Action Plan")
    
    if result == 1:
        with st.expander("View Churn Prevention Strategies", expanded=True):
            st.dataframe(pd.DataFrame({"Action Plan": CHURN_TIPS}), hide_index=True, use_container_width=True)
    else:
        with st.expander("View Customer Retention Strategies"):
            st.dataframe(pd.DataFrame({"Retention Strategy": RETENTION_TIPS}), hide_index=True, use_container_width=True)


# --- 4. MAIN APP EXECUTION ---
def main():
    
    # Custom CSS for awesome header
    st.markdown("""
        <style>
        .awesome-header {
            font-size: 36px;
            color: #FF4B4B; /* Streamlit Primary Red */
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
    st.markdown('<p class="awesome-header">🔮 Customer Churn Risk Analyzer</p>', unsafe_allow_html=True)
    st.markdown("Predict the likelihood of a customer leaving using their profile and service details entered in the sidebar.")
    st.markdown("---")

    # Get User Inputs
    user_data = build_sidebar()

    # Prediction Button
    if st.sidebar.button("✨ RUN CHURN ANALYSIS", type="primary", use_container_width=True):
        
        # Input Validation
        if user_data['tenure'] == 0 and user_data['TotalCharges'] > 0.0:
            st.warning("⚠️ Input Alert: A new customer (Tenure 0) must have $0 Total Charges.")
        else:
            with st.spinner('Calculating probabilities and risk score...'):
                prediction_result, probability_array = get_prediction(user_data)

            if prediction_result is not None:
                display_results(prediction_result, probability_array)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("App built by Anmol | Model: Logistic Regression")

if __name__ == "__main__":
    main()