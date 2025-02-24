# --- Importing Required Libraries ---
import re
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="HR & DEI Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("HR & DEI Analytics Dashboard")
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

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(".", "")

    # Rename 'left' to 'attrition' for clarity
    df.rename(columns={"left": "attrition"}, inplace=True)

    return df

df = load_data("HR_comma_sep.csv")

# --- Train Model ---
@st.cache_data
def train_model(df):
    df = df.copy()
    
    # Convert categorical columns
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    df["department"] = df["department"].astype("category").cat.codes  # Encode departments numerically

    # Select numeric columns
    df = df.select_dtypes(include=[np.number])

    # Fill missing values
    df.fillna(df.mean(), inplace=True)

    # Split features and target
    X = df.drop("attrition", axis=1)
    y = df["attrition"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model(df)

# --- DEI Metrics Page ---
if page == "DEI Metrics":
    st.header("📊 Diversity, Equity, and Inclusion (DEI) Metrics")
    st.write("This section explores **attrition trends** by department, salary levels, promotion status, and workload.")

    # Attrition by Department
    st.subheader("🔹 Attrition by Department")
    attrition_by_dept = df.groupby("department")["attrition"].mean().reset_index()
    fig_dept = px.bar(attrition_by_dept, x="department", y="attrition", color="attrition", 
                      title="Attrition Rate by Department", labels={"attrition": "Attrition Rate"})
    st.plotly_chart(fig_dept, use_container_width=True)

    # Salary vs Attrition
    st.subheader("💰 Salary Level & Attrition")
    attrition_by_salary = df.groupby("salary")["attrition"].mean().reset_index()
    fig_salary = px.bar(attrition_by_salary, x="salary", y="attrition", color="attrition",
                        title="Attrition Rate by Salary Level", labels={"attrition": "Attrition Rate"})
    st.plotly_chart(fig_salary, use_container_width=True)

    # Promotion vs Attrition
    st.subheader("📈 Promotion Status & Attrition")
    promotion_attrition = df.groupby("promotion_last_5years")["attrition"].mean().reset_index()
    promotion_attrition["promotion_last_5years"] = promotion_attrition["promotion_last_5years"].map({0: "No Promotion", 1: "Promoted"})
    fig_promotion = px.bar(promotion_attrition, x="promotion_last_5years", y="attrition", color="attrition",
                            title="Attrition Rate by Promotion Status", labels={"attrition": "Attrition Rate"})
    st.plotly_chart(fig_promotion, use_container_width=True)

    # Work Hours vs Attrition
    st.subheader("⏳ Work Hours & Attrition")
    fig_work_hours = px.histogram(df, x="average_montly_hours", color="attrition",
                                  title="Work Hours Distribution & Attrition", labels={"average_montly_hours": "Avg Monthly Hours"})
    st.plotly_chart(fig_work_hours, use_container_width=True)

    # Job Satisfaction vs Attrition
    st.subheader("😊 Job Satisfaction & Attrition")
    fig_satisfaction = px.box(df, x="attrition", y="satisfaction_level", color="attrition",
                              title="Job Satisfaction Distribution by Attrition", labels={"satisfaction_level": "Satisfaction Level"})
    st.plotly_chart(fig_satisfaction, use_container_width=True)

# --- Home Page ---
if page == "Home":
    st.header('HR & DEI Analytics Dashboard')
    st.write("📌 Welcome! This dashboard helps HR teams analyze attrition trends, predict employee turnover, and gain DEI insights.")

    # Quick Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Rate", f"{df['attrition'].mean()*100:.2f}%", "⬆ Higher than last year")
    col2.metric("Avg. Satisfaction Score", f"{df['satisfaction_level'].mean()*100:.1f}%", "⬆ Improved")
    col3.metric("Top Attrition Factor", "Lack of Career Growth", "🔍 More Insights Available")

    # Attrition Trends Chart
    st.subheader("📉 Attrition Trends Over Time")
    fig = px.line(df, x="time_spend_company", y="attrition", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)
