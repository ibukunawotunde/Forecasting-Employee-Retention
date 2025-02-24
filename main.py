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
import openai  # For AI-powered chatbot
import warnings

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Custom Styling for Modern UI ---
st.markdown(
    """
    <style>
        .st-emotion-cache-16txtl3 h1 {
            font: bold 30px Arial;
            text-align: center;
            margin-bottom: 15px;
            color: #005DAA;
        }
        div[data-testid=stSidebarContent] {
            background-color: #E6EEF8;
            border-right: 4px solid #005DAA;
            padding: 15px!important;
        }
        .main-container {
            background-color: #F8FAFC;
            padding: 20px;
            border-radius: 12px;
        }
        div[data-testid=stFormSubmitButton]> button {
            width: 100%;
            background: linear-gradient(90deg, #005DAA, #0073E6);
            border: none;
            padding: 14px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            transition: 0.3s ease-in-out;
        }
        div[data-testid=stFormSubmitButton]> button:hover {
            background: linear-gradient(90deg, #0073E6, #005DAA);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("HR Analytics Dashboard")
    page = option_menu(
        menu_title=None,
        options=['Home', 'Attrition Prediction', 'Feature Importance', 'DEI Metrics', 'HR Chatbot', 'Generate Report'],
        icons=['house-fill', 'bar-chart-line-fill', "graph-up-arrow", "globe-americas", "chat-text-fill", "file-earmark-text-fill"],
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

# --- Train Model ---
@st.cache_data
def train_model(df):
    df = df.copy()
    df["Salary"] = df["Salary"].map({"low": 0, "medium": 1, "high": 2})
    
    # Selecting numeric data only
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)

    X = df.drop("left", axis=1, errors="ignore")
    y = df["left"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model(df)

# --- Home Page ---
if page == "Home":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header('HR Analytics Dashboard')
    st.subheader("🔹 Welcome to the HR Analytics Platform!")
    st.write("This platform helps HR professionals analyze attrition trends, predict employee turnover, and gain insights into workplace diversity and inclusion.")

    # Quick Metrics
    st.subheader("📊 Quick HR Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Rate", "23%", "⬆ 5% from last year")
    col2.metric("Avg. Satisfaction Score", "72%", "⬆ 3%")
    col3.metric("Top Attrition Factor", "Lack of Career Growth", "🔍 Insights Available")

    # Interactive Attrition Trends Chart
    st.subheader("📉 Attrition Trends Over Time")
    fig = px.line(df, x="time_spend_company", y="left", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Attrition Prediction Page ---
if page == "Attrition Prediction":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🔍 Predict Employee Attrition")

    # Form for user input
    with st.form("Prediction_Form"):
        satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.slider('Last Evaluation Score', 0.0, 1.0, 0.6)
        number_project = st.slider("Number of Projects", 1, 10, 4)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
        time_spend_company = st.slider("Years at Company", 1, 20, 5)
        work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        department = st.selectbox("Department", df["Department"].unique())
        salary = st.selectbox("Salary Level", df["Salary"].unique())

        # Submit button
        predict_button = st.form_submit_button("Predict Attrition")

    if predict_button:
        # Encoding Categorical Variables
        department_encoded = [1 if department == dept else 0 for dept in df["Department"].unique()]
        salary_encoded = [1 if salary == sal else 0 for sal in df["Salary"].unique()]

        # Constructing Feature Input
        input_features = [
            satisfaction_level, last_evaluation, number_project,
            average_montly_hours, time_spend_company, work_accident,
            promotion_last_5years
        ] + department_encoded + salary_encoded

        # Make Prediction
        try:
            prediction_result = model.predict([input_features])[0]
            prediction_prob = model.predict_proba([input_features])[0]

            # Display Result
            if prediction_result == 1:
                st.error(f"⚠️ High Risk: Employee is likely to leave! (Probability: {prediction_prob[1]*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk: Employee is likely to stay. (Probability: {prediction_prob[0]*100:.2f}%)")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
