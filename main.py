# --- Importing ToolKits ---
import re
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as html
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from fpdf import FPDF
import openai
import warnings

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("HR Analytics Dashboard")
    page = option_menu(
        menu_title=None,
        options=['Home', 'Attrition Prediction', 'DEI Metrics', 'HR Chatbot', 'Generate Report'],
        icons=['house-fill', 'bar-chart-line-fill', "globe-americas", "chat-text-fill", "file-earmark-text-fill"],
        menu_icon="cast",
        default_index=0
    )

# --- Load Dataset ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(" ", "_").str.replace(".", "")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data("HR_comma_sep.csv")

# --- Train Model (Ensuring Correct Feature Count) ---
@st.cache_data
def train_model(df):
    df = df.copy()
    
    # Encode categorical variables
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    
    # Selecting numeric data only
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)

    # Define X (Features) & Y (Target)
    X = df.drop(columns=["left"], errors="ignore")
    y = df["left"].astype(int)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, X.columns.tolist()

model, X_train, X_test, y_train, y_test, feature_names = train_model(df)

# --- Attrition Prediction Page ---
if page == "Attrition Prediction":
    st.header("🔍 Predict Employee Attrition")

    # Capture user inputs
    with st.form("Prediction_Form"):
        satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.slider('Last Evaluation Score', 0.0, 1.0, 0.6)
        number_project = st.slider("Number of Projects", 1, 10, 4)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
        time_spend_company = st.slider("Years at Company", 1, 20, 5)
        work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

        # Submit button
        predict_button = st.form_submit_button("Predict Attrition")

    if predict_button:
        try:
            # Encode Salary to match training data
            salary_encoded = {"low": 0, "medium": 1, "high": 2}[salary]

            # Construct Feature Input (Ensure Feature Order Matches Training)
            input_features = np.array([
                satisfaction_level, last_evaluation, number_project,
                average_montly_hours, time_spend_company, work_accident,
                promotion_last_5years, salary_encoded  # Matches training data (8 features)
            ]).reshape(1, -1)

            # Make Prediction
            prediction_result = model.predict(input_features)[0]
            prediction_prob = model.predict_proba(input_features)[0]

            # Display Result
            if prediction_result == 1:
                st.error(f"⚠️ High Risk: Employee is likely to leave! (Probability: {prediction_prob[1]*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk: Employee is likely to stay. (Probability: {prediction_prob[0]*100:.2f}%)")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
