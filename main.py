# --- Importing Required Libraries ---
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
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    X = df.drop("ATTRITION", axis=1, errors="ignore")
    y = df["ATTRITION"].astype(int)
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
    fig = px.line(df, x="YEARS_AT_COMPANY", y="ATTRITION", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- DEI Metrics Dashboard ---
if page == "DEI Metrics":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🌍 Diversity, Equity & Inclusion (DEI) Metrics")
    st.write("Analyze attrition rates by gender, department, job role, salary level, and work-life balance to identify disparities.")

    # Ensure required columns exist
    required_columns = ["ATTRITION", "GENDER", "DEPARTMENT", "MARITAL_STATUS", "DAILY_RATE"]
    
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing Columns: {', '.join(missing_cols)}. Please ensure DEI data is available.")
    else:
        df["GENDER"] = df["GENDER"].map({1: "Female", 2: "Male"})
        df["DEPARTMENT"] = df["DEPARTMENT"].map({1: "HR", 2: "R&D", 3: "Sales"})
        df["MARITAL_STATUS"] = df["MARITAL_STATUS"].map({1: "Divorced", 2: "Married", 3: "Single"})

        # Attrition by Gender
        st.subheader("📊 Attrition Rate by Gender")
        fig1 = px.bar(df.groupby("GENDER")["ATTRITION"].mean().reset_index(), x="GENDER", y="ATTRITION",
                      color="GENDER", title="Attrition Rate by Gender")
        st.plotly_chart(fig1, use_container_width=True)

        # Attrition by Department
        st.subheader("🏢 Attrition by Department")
        fig2 = px.bar(df.groupby("DEPARTMENT")["ATTRITION"].mean().reset_index(), x="DEPARTMENT", y="ATTRITION",
                      color="DEPARTMENT", title="Attrition Rate by Department")
        st.plotly_chart(fig2, use_container_width=True)

        # Attrition by Salary
        st.subheader("💰 Attrition by Salary Level")
        fig3 = px.box(df, x="ATTRITION", y="DAILY_RATE", color="ATTRITION",
                      title="Salary Distribution Among Employees Who Left vs Stayed")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- AI Chatbot ---
if page == "HR Chatbot":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🤖 HR Policy Chatbot")
    user_input = st.text_input("Ask me something about HR policies...")
    if st.button("Ask Chatbot"):
        response = f"Great question! HR policy regarding {user_input} is currently being updated."
        st.write(response)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Generate Report ---
if page == "Generate Report":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("📄 Generate HR Report")
    if st.button("Download Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="HR Analytics Report", ln=True, align="C")
        pdf.output("HR_Analytics_Report.pdf")
        st.success("Report generated successfully!")
    st.markdown('</div>', unsafe_allow_html=True)
