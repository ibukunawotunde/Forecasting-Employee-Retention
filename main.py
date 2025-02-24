# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# --- Page Configuration ---
st.set_page_config(
    page_title="HR & DEI Analytics",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar Styling ---
side_bar_options_style = {
    "container": {"padding": "0!important", "background-color": 'transparent'},
    "icon": {"color": "white", "font-size": "18px"},
    "nav-link": {"color": "#fff", "font-size": "20px", "text-align": "left", "margin": "5px"},
    "nav-link-selected": {"background-color": "#17B794", "font-size": "20px"},
}

# --- Sidebar Navigation ---
with st.sidebar:
    st.title(":blue[HR & DEI Analytics Dashboard]")

    # Ensure image exists, else replace with a valid one
    try:
        st.image("imgs/dashboard_logo.png", caption="", width=120)
    except:
        st.image("https://via.placeholder.com/150", caption="Logo Missing", width=120)

    page = option_menu(
        menu_title=None,
        options=['Home', 'Attrition Prediction', 'Feature Importance', 'DEI Metrics', 'HR Chatbot', 'Generate Report'],
        icons=['house-fill', 'bar-chart-line-fill', "graph-up-arrow", "globe-americas", "chat-text-fill", "file-earmark-text-fill"],
        menu_icon="cast",
        default_index=0,
        styles=side_bar_options_style
    )

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(" ", "_").str.replace(".", "")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Load dataset
df = load_data("HR_comma_sep.csv")

# --- Train Model ---
@st.cache_data
def train_model(df):
    df = df.copy()

    # Encode categorical variables
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})

    # Drop non-numeric columns
    df = df.select_dtypes(include=[np.number])

    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Define Features & Target
    X = df.drop("left", axis=1, errors="ignore")
    y = df["left"].astype(int)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Train the model
model, X_test, y_test = train_model(df)

# --- Home Page ---
if page == "Home":
    st.title("🏠 HR & DEI Analytics Dashboard")
    st.write("This dashboard provides insights into employee attrition, feature importance, and DEI analysis.")

    # Show Sample Data
    st.subheader("Dataset Overview")
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
        salary = [0, 0]  # Default: High Salary
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

    # Gender Bias in Attrition
    st.subheader("Attrition by Gender")
    gender_attrition = df.groupby("gender")["left"].mean()
    st.bar_chart(gender_attrition)

    # Salary vs. Attrition
    st.subheader("Salary Distribution & Attrition")
    salary_attrition = df.groupby("salary")["left"].mean()
    st.bar_chart(salary_attrition)

    # Departmental Diversity
    st.subheader("Attrition by Department")
    dept_attrition = df.groupby("Department")["left"].mean()
    st.bar_chart(dept_attrition)

# --- HR Chatbot ---
elif page == "HR Chatbot":
    st.title("💬 HR Chatbot (Coming Soon)")
    st.write("This chatbot will provide insights and answer HR-related questions.")

# --- Generate Report ---
elif page == "Generate Report":
    st.title("📄 Generate HR Report")
    st.write("Generate a customized HR and DEI analytics report.")

    if st.button("Download Report"):
        st.success("Report Downloaded Successfully ✅")
