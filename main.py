import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import joblib
import openai
from fpdf import FPDF
import matplotlib.pyplot as plt
from time import sleep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set page configuration with improved UI
st.set_page_config(
    page_title="HR Analytics - Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for sleek design
st.markdown("""
    <style>
        .main {background-color: #f4f4f8;}
        .stApp {background-color: #e0e7ff;}
        .css-1d391kg {color: #4B0082 !important;}
        .css-18e3th9 {color: #000080 !important;}
        .stSidebar {background-color: #4B0082; color: white;}
        .css-1aumxhk {color: white !important;}
        .stButton>button {background-color: #6a5acd; color: white; border-radius: 5px;}
        .stTextInput>div>div>input {border-radius: 10px; border: 1px solid #6a5acd;}
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    df.columns = df.columns.str.replace(" ", "_").str.replace(".", "")
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

df = load_data()

# Preprocess Data
@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    return df

df_processed = preprocess_data(df)

# Train Model
@st.cache_resource
def train_model(df):
    X = df.drop("left", axis=1, errors="ignore")
    y = df["left"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(df_processed)

# Sidebar Navigation
st.sidebar.image("https://i.imgur.com/JgQ7pPo.png", width=250)
st.sidebar.title("HR Analytics Dashboard")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Attrition Prediction", "📊 What-If Analysis", "🔬 Feature Importance", "🌍 DEI Metrics", "💬 HR Chatbot", "📄 Generate Report"])

if page == "🏠 Home":
    st.title("📊 HR Analytics - Employee Attrition Prediction")
    st.write("This application helps HR teams analyze and predict employee attrition using data-driven insights while incorporating DEI metrics.")
    st.subheader("📌 Key Insights:")
    fig = px.bar(df, x="Department", y="left", color="left", barmode="group", title="Attrition by Department")
    st.plotly_chart(fig)

elif page == "🔮 Attrition Prediction":
    st.title("🧑‍💼 Employee Attrition Prediction")
    with st.form("attrition_form"):
        satisfaction = st.slider("Satisfaction Level", 0.1, 1.0, 0.5)
        last_evaluation = st.slider("Last Evaluation", 0.1, 1.0, 0.7)
        avg_hours = st.slider("Average Monthly Hours", 50, 320, 150)
        time_spent = st.slider("Time Spent at Company", 1, 20, 5)
        salary = st.selectbox("Salary Level", ["Low", "Medium", "High"])
        salary_map = {"Low": 0, "Medium": 1, "High": 2}
        salary_val = salary_map[salary]
        submit_button = st.form_submit_button("🚀 Predict Attrition")
    
    if submit_button:
        input_data = np.array([[satisfaction, last_evaluation, avg_hours, time_spent, salary_val]])
        prediction = model.predict(input_data)
        result = "🚀 Likely to Leave" if prediction[0] == 1 else "✅ Likely to Stay"
        st.subheader(f"Prediction: {result}")

elif page == "🔬 Feature Importance":
    st.title("🔍 Feature Importance using SHAP")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

elif page == "🌍 DEI Metrics":
    st.title("🌎 Diversity, Equity, and Inclusion Metrics")
    fig = px.bar(df, x="salary", y="left", color="left", barmode="group", title="Attrition by Salary Level")
    st.plotly_chart(fig)

elif page == "💬 HR Chatbot":
    st.title("🤖 AI-Powered HR Chatbot")
    user_input = st.text_input("Ask me about employee attrition or DEI insights:")
    if user_input:
        response = "Attrition and DEI are closely linked. Employees from underrepresented backgrounds may face unique challenges. Data insights can help address these gaps proactively."
        st.write("🤖 HR Chatbot:", response)

elif page == "📄 Generate Report":
    st.title("📜 Generate PDF Report")
    if st.button("📥 Download Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="HR & DEI Attrition Report", ln=True, align='C')
        pdf.output("attrition_report.pdf")
        st.success("✅ Report Generated! Download below.")
        st.download_button("📥 Download Report", "attrition_report.pdf")
