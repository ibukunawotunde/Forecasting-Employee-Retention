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

# --- MOVE THIS TO THE TOP (Fixes the Page Config Error) ---
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

    # Ensure correct column name usage
    if "left" not in df.columns:
        st.error("Error: The dataset does not contain a 'left' column. Please check the dataset column names.")
        st.stop()

    return df

df = load_data("HR_comma_sep.csv")

# --- DEI Metrics Page ---
if page == "DEI Metrics":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🌍 Diversity, Equity & Inclusion (DEI) Metrics")
    st.write("This dashboard provides insights into workforce diversity, employee attrition trends among underrepresented groups, and career growth opportunities.")

    # Gender Diversity Breakdown
    st.subheader("👩‍💼 Workforce Gender Diversity")
    if "gender" in df.columns:
        gender_distribution = df["gender"].value_counts()
        fig_gender = px.pie(
            names=gender_distribution.index, 
            values=gender_distribution.values,
            title="Workforce Gender Distribution"
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.warning("⚠️ Gender data is not available in the dataset.")

    # Departmental Diversity Breakdown
    st.subheader("🏢 Diversity Across Departments")
    if "department" in df.columns:
        department_distribution = df["department"].value_counts()
        fig_department = px.bar(
            x=department_distribution.index,
            y=department_distribution.values,
            title="Diversity Representation Across Departments",
            labels={"x": "Department", "y": "Count"},
            color=department_distribution.index
        )
        st.plotly_chart(fig_department, use_container_width=True)
    else:
        st.warning("⚠️ Department data is not available in the dataset.")

    # Attrition Rate Among Underrepresented Groups
    st.subheader("📉 Attrition Rate by Gender & Salary Level")
    if "gender" in df.columns and "left" in df.columns:
        attrition_by_gender = df.groupby("gender")["left"].mean().reset_index()
        fig_attrition_gender = px.bar(
            attrition_by_gender, x="gender", y="left", 
            title="Attrition Rate by Gender",
            labels={"left": "Attrition Rate"},
            color="gender"
        )
        st.plotly_chart(fig_attrition_gender, use_container_width=True)
    else:
        st.warning("⚠️ Gender or Attrition data is missing.")

    if "salary" in df.columns and "left" in df.columns:
        attrition_by_salary = df.groupby("salary")["left"].mean().reset_index()
        fig_attrition_salary = px.bar(
            attrition_by_salary, x="salary", y="left",
            title="Attrition Rate by Salary Level",
            labels={"left": "Attrition Rate"},
            color="salary"
        )
        st.plotly_chart(fig_attrition_salary, use_container_width=True)
    else:
        st.warning("⚠️ Salary or Attrition data is missing.")

    # Salary Disparity Analysis
    st.subheader("💰 Salary Distribution by Gender")
    if "gender" in df.columns and "average_montly_hours" in df.columns:
        fig_salary = px.box(
            df, x="gender", y="average_montly_hours",
            title="Salary Distribution by Gender",
            labels={"average_montly_hours": "Monthly Hours Worked"}
        )
        st.plotly_chart(fig_salary, use_container_width=True)
    else:
        st.warning("⚠️ Salary or Gender data is missing.")

    # Promotion & Career Growth Insights
    st.subheader("🚀 Promotion Rates by Gender")
    if "gender" in df.columns and "promotion_last_5years" in df.columns:
        promotion_by_gender = df.groupby("gender")["promotion_last_5years"].mean().reset_index()
        fig_promotion = px.bar(
            promotion_by_gender, x="gender", y="promotion_last_5years",
            title="Promotion Rates by Gender",
            labels={"promotion_last_5years": "Promotion Rate"},
            color="gender"
        )
        st.plotly_chart(fig_promotion, use_container_width=True)
    else:
        st.warning("⚠️ Gender or Promotion data is missing.")

    st.markdown('</div>', unsafe_allow_html=True)
