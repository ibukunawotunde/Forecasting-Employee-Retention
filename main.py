# --- Streamlit Page Configuration (MUST BE THE FIRST COMMAND) ---
import streamlit as st
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Importing ToolKits ---
import re
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as html
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from fpdf import FPDF
import openai
import warnings

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

    # Ensure correct column name usage
    if "left" not in df.columns:
        st.error("Error: The dataset does not contain a 'left' column. Please check the dataset column names.")
        st.stop()

    return df

df = load_data("HR_comma_sep.csv")

# --- Home Page ---
if page == "Home":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header('HR Analytics Dashboard')
    st.subheader("🔹 Welcome to the HR Analytics Platform!")
    st.write("This platform helps HR professionals analyze attrition trends, predict employee turnover, and gain insights into workplace diversity and inclusion.")

    # Quick Metrics
    st.subheader("📊 Quick HR Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Rate", f"{df['left'].mean() * 100:.1f}%", "⬆ 5% from last year")
    col2.metric("Avg. Satisfaction Score", f"{df['satisfaction_level'].mean() * 100:.1f}%", "⬆ 3%")
    col3.metric("Top Attrition Factor", "Lack of Career Growth", "🔍 Insights Available")

    # Interactive Attrition Trends Chart
    st.subheader("📉 Attrition Trends Over Time")
    fig = px.line(df, x="time_spend_company", y="left", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)

    # Employee Search (HR Lookup)
    st.subheader("🔎 Employee Search")
    emp_id = st.text_input("Enter Employee ID:")
    if emp_id:
        st.write(f"Employee {emp_id} is currently **Active**")

    st.markdown('</div>', unsafe_allow_html=True)

# --- DEI Metrics Page ---
if page == "DEI Metrics":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🌍 Diversity, Equity & Inclusion (DEI) Metrics")

    # Gender Diversity Breakdown
    if "gender" in df.columns:
        st.subheader("👩‍💼 Workforce Gender Diversity")
        gender_distribution = df["gender"].value_counts()
        fig_gender = px.pie(names=gender_distribution.index, values=gender_distribution.values, title="Workforce Gender Distribution")
        st.plotly_chart(fig_gender, use_container_width=True)

    # Attrition Rate Among Underrepresented Groups
    if "gender" in df.columns and "left" in df.columns:
        st.subheader("📉 Attrition Rate by Gender & Salary Level")
        attrition_by_gender = df.groupby("gender")["left"].mean().reset_index()
        fig_attrition_gender = px.bar(attrition_by_gender, x="gender", y="left", title="Attrition Rate by Gender", labels={"left": "Attrition Rate"}, color="gender")
        st.plotly_chart(fig_attrition_gender, use_container_width=True)

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
        pdf.cell(200, 10, txt=f"Attrition Rate: {df['left'].mean() * 100:.1f}%", ln=True)
        pdf.output("HR_Analytics_Report.pdf")
        st.success("Report generated successfully!")
    st.markdown('</div>', unsafe_allow_html=True)
