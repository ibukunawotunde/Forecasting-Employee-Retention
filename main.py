# --- Importing ToolKits ---
import re
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="HR & DEI Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
        .st-emotion-cache-16txtl3 h1 {
            font: bold 30px Arial;
            text-align: center;
            color: #005DAA;
        }
        div[data-testid=stSidebarContent] {
            background-color: #E6EEF8;
            border-right: 4px solid #005DAA;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("HR & DEI Analytics Dashboard")
    page = option_menu(
        menu_title=None,
        options=['Home', 'DEI Metrics', 'Salary Analysis', 'Work-Life Balance', 'Promotion Trends'],
        icons=['house-fill', "globe-americas", "currency-dollar", "activity", "bar-chart-line"],
        menu_icon="cast",
        default_index=0
    )

# --- Load Dataset ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(".", "")

    return df

df = load_data("HR_comma_sep.csv")

# --- DEI Metrics Page ---
if page == "DEI Metrics":
    st.header("🌍 Diversity, Equity, and Inclusion Metrics")

    # Departmental Representation
    st.subheader("📌 Departmental Representation by Gender")
    gender_dept = df.groupby(["department", "salary"]).size().unstack()
    st.bar_chart(gender_dept)

    # Salary Distribution by Gender
    st.subheader("📌 Salary Distribution Across Genders")
    salary_gender = df.groupby(["salary", "department"]).size().unstack()
    st.bar_chart(salary_gender)

    # Work-Life Balance
    st.subheader("📌 Work-Life Balance Analysis")
    fig = px.scatter(df, x="average_montly_hours", y="satisfaction_level", color="department",
                     title="Correlation between Monthly Hours and Satisfaction Level")
    st.plotly_chart(fig)

# --- Salary Analysis Page ---
if page == "Salary Analysis":
    st.header("💰 Salary Distribution Across Gender & Departments")
    fig = px.box(df, x="salary", y="average_montly_hours", color="department",
                 title="Salary vs. Work Hours by Department")
    st.plotly_chart(fig)

# --- Work-Life Balance Page ---
if page == "Work-Life Balance":
    st.header("⚖ Work-Life Balance Trends")
    fig = px.scatter(df, x="average_montly_hours", y="satisfaction_level", color="salary",
                     title="Work Hours vs. Satisfaction Level")
    st.plotly_chart(fig)

# --- Promotion Trends Page ---
if page == "Promotion Trends":
    st.header("📈 Promotion Trends by Gender & Department")
    fig = px.bar(df, x="department", y="promotion_last_5years", color="salary",
                 title="Promotion Rates Across Departments")
    st.plotly_chart(fig)
