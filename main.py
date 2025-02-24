# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from streamlit_option_menu import option_menu

# --- Page Configuration ---
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS Styling for a Modern Look ---
st.markdown(
    """
    <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(to right, #0072C6, #004A8F);
            color: white;
        }

        /* Sidebar Customization */
        div[data-testid="stSidebar"] {
            background-color: #1E1E1E;
            padding: 20px;
        }
        div[data-testid="stSidebarContent"] {
            color: white;
            font-size: 18px;
        }

        /* Sidebar Menu Text Styling */
        .css-1v0mbdj p, .css-1v0mbdj a {
            font-size: 18px;
            font-weight: bold;
        }

        /* Centered Titles */
        .st-emotion-cache-1h1ov1w h1 {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
        }

        /* Buttons & Forms */
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            background-color: #0072C6;
            border: none;
            padding: 12px;
            border-radius: 8px;
            color: white;
            font-size: 16px;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #004A8F;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Menu (Updated without Missing Logo) ---
with st.sidebar:
    st.title(":blue[HR Analytics Dashboard]")

    # Sidebar Navigation Menu
    page = option_menu(
        menu_title=None,
        options=['Home', 'Attrition Prediction', 'Feature Importance', 'DEI Metrics', 'HR Chatbot', 'Generate Report'],
        icons=['house-fill', 'bar-chart-line-fill', "graph-up-arrow", "globe-americas", "chat-text-fill", "file-earmark-text-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#17B794", "font-size": "20px"},
        }
    )

# --- Load Data ---
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
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
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
    st.title("🏠 HR Analytics Dashboard")
    st.write("A modern dashboard to analyze employee attrition and diversity trends.")

    # Show Sample Data
    st.subheader("📋 Dataset Overview")
    st.dataframe(df.head(10))

# --- Attrition Prediction ---
elif page == "Attrition Prediction":
    st.title("📊 Attrition Prediction")

    with st.form("Predict_value"):
        satisfaction_level = st.number_input('Satisfaction Level', min_value=0.05, max_value=1.0, value=0.66)
        last_evaluation = st.number_input('Last Evaluation', min_value=0.05, max_value=1.0, value=0.54)
        avg_monthly_hours = st.number_input('Average Monthly Hours', min_value=50, max_value=320, step=1, value=120)
        time_in_company = st.number_input('Time In Company', min_value=1, max_value=20, step=1, value=5)
        salary_category = st.selectbox("Salary", options=["low", "medium", "high"])

        predict_button = st.form_submit_button(label='Predict')

    if predict_button:
        salary = [0, 0]
        if salary_category == "low":
            salary = [1, 0]
        elif salary_category == "medium":
            salary = [0, 1]

        new_data = [satisfaction_level, last_evaluation, avg_monthly_hours, time_in_company] + salary
        predicted_value = model.predict([new_data])[0]

        if predicted_value == 0:
            st.success("✅ Employee is expected to **STAY**")
        else:
            st.error("🚨 Employee is expected to **LEAVE**")

# --- Feature Importance ---
elif page == "Feature Importance":
    st.title("🔎 Feature Importance")

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

# --- DEI Metrics ---
elif page == "DEI Metrics":
    st.title("🌍 DEI Analytics")

    st.subheader("📌 Attrition by Gender")
    gender_attrition = df.groupby("gender")["left"].mean()
    st.bar_chart(gender_attrition)

    st.subheader("📌 Salary vs. Attrition")
    salary_attrition = df.groupby("salary")["left"].mean()
    st.bar_chart(salary_attrition)

    st.subheader("📌 Attrition by Department")
    dept_attrition = df.groupby("Department")["left"].mean()
    st.bar_chart(dept_attrition)

# --- HR Chatbot ---
elif page == "HR Chatbot":
    st.title("💬 HR Chatbot (Coming Soon)")
    st.write("An AI-powered chatbot for HR insights.")

# --- Generate Report ---
elif page == "Generate Report":
    st.title("📄 Generate HR Report")
    st.write("Create a customized HR and DEI analytics report.")
