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
        options=['Home', 'Attrition Prediction', 'Feature Importance', 'DEI Metrics', 'HR Chatbot', 'Generate Report'],
        icons=['house-fill', 'bar-chart-line-fill', "graph-up-arrow", "globe-americas", "chat-text-fill", "file-earmark-text-fill"],
        menu_icon="cast",
        default_index=0
    )

# --- Load Dataset ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # 🔹 Normalize column names (lowercase, remove spaces)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(".", "")

    # 🔹 Rename 'attrition' properly to match dataset
    if "attrition" not in df.columns and "attrition".capitalize() in df.columns:
        df.rename(columns={"Attrition": "attrition"}, inplace=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data("HR_comma_sep.csv")

# --- Check for Attrition Column ---
if "attrition" not in df.columns:
    st.error("🚨 Error: The dataset does not contain an 'attrition' column. Please check the dataset column names.")
    st.write("📌 **Available Columns in the dataset:**", df.columns.tolist())

# --- Train Model ---
@st.cache_data
def train_model(df):
    df = df.copy()
    
    # Ensure 'attrition' column exists before proceeding
    if "attrition" not in df.columns:
        raise KeyError("🚨 The dataset does not contain an 'attrition' column.")

    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    X = df.drop("attrition", axis=1, errors="ignore")
    y = df["attrition"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

try:
    model, X_test, y_test = train_model(df)
except KeyError as e:
    st.error(f"🚨 Dataset Error: {e}")

# --- Home Page ---
if page == "Home":
    st.header('HR Analytics Dashboard')
    st.write("📌 Welcome! This dashboard helps HR teams analyze attrition trends, predict employee turnover, and gain DEI insights.")

    # Quick Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Rate", "23%", "⬆ 5% from last year")
    col2.metric("Avg. Satisfaction Score", "72%", "⬆ 3%")
    col3.metric("Top Attrition Factor", "Lack of Career Growth", "🔍 Insights Available")

    # Attrition Trends Chart
    st.subheader("📉 Attrition Trends Over Time")
    fig = px.line(df, x="years_at_company", y="attrition", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)

# --- DEI Metrics ---
if page == "DEI Metrics":
    st.header("🌍 Diversity, Equity & Inclusion (DEI) Metrics")
    st.write("Analyze attrition trends by gender, department, salary, and job satisfaction.")

    required_columns = ["attrition", "gender", "department", "salary"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"🚨 Missing Columns: {', '.join(missing_cols)}. Ensure the dataset includes required DEI fields.")
    else:
        df["gender"] = df["gender"].map({1: "Female", 2: "Male"})
        df["department"] = df["department"].map({1: "HR", 2: "R&D", 3: "Sales"})

        # Attrition by Gender
        st.subheader("📊 Attrition Rate by Gender")
        fig1 = px.bar(df.groupby("gender")["attrition"].mean().reset_index(), x="gender", y="attrition",
                      color="gender", title="Attrition Rate by Gender")
        st.plotly_chart(fig1, use_container_width=True)

        # Attrition by Department
        st.subheader("🏢 Attrition by Department")
        fig2 = px.bar(df.groupby("department")["attrition"].mean().reset_index(), x="department", y="attrition",
                      color="department", title="Attrition Rate by Department")
        st.plotly_chart(fig2, use_container_width=True)

        # Attrition by Salary
        st.subheader("💰 Attrition by Salary Level")
        fig3 = px.box(df, x="attrition", y="salary", color="attrition",
                      title="Salary Distribution Among Employees Who Left vs Stayed")
        st.plotly_chart(fig3, use_container_width=True)
