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

    # 🔹 Print the exact column names in Streamlit for debugging
    st.subheader("📌 Columns in the Uploaded Dataset:")
    st.write(df.columns.tolist())  # Display column names

    # 🔹 Normalize column names (lowercase, remove spaces)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(".", "")

    # 🔹 Check if 'attrition' exists in any format
    if "attrition" not in df.columns:
        attrition_variants = [col for col in df.columns if "attrition" in col.lower()]
        if attrition_variants:
            correct_attrition_col = attrition_variants[0]  # Get the closest match
            df.rename(columns={correct_attrition_col: "attrition"}, inplace=True)
            st.success(f"✔ Automatically renamed '{correct_attrition_col}' to 'attrition'")

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
