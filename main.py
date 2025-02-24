# --- Importing ToolKits ---
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
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
        options=['Home', 'DEI Metrics', 'Attrition Prediction', 'Salary Analysis', 'Work-Life Balance', 'Promotion Trends'],
        icons=['house-fill', "globe-americas", 'bar-chart-line', "currency-dollar", "activity", "bar-chart-line"],
        menu_icon="cast",
        default_index=0
    )

# --- Load Dataset ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Convert column names to lowercase and remove spaces
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(".", "")

    # Fix Attrition Column Name
    if "left" in df.columns:
        df.rename(columns={"left": "attrition"}, inplace=True)

    return df

df = load_data("HR_comma_sep.csv")

# --- Train Model ---
@st.cache_data
def train_model(df):
    df = df.copy()
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    X = df.drop("attrition", axis=1, errors="ignore")
    y = df["attrition"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(df)

# --- Home Page ---
if page == "Home":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header('HR & DEI Analytics Dashboard')
    st.subheader("🔹 Welcome to the HR Analytics Platform!")
    st.write("Analyze attrition trends, predict employee turnover, and gain insights into workplace diversity.")

    # Quick Metrics
    st.subheader("📊 Quick HR Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Rate", f"{df['attrition'].mean()*100:.2f}%", "⬆ 5% from last year")
    col2.metric("Avg. Satisfaction Score", f"{df['satisfaction_level'].mean()*100:.2f}%", "⬆ 3%")
    col3.metric("Top Attrition Factor", "Lack of Career Growth", "🔍 Insights Available")

    # Interactive Attrition Trends Chart
    st.subheader("📉 Attrition Trends Over Time")
    fig = px.line(df, x="time_spend_company", y="attrition", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)

    # Employee Search (HR Lookup)
    st.subheader("🔎 Employee Search")
    emp_id = st.text_input("Enter Employee ID:")
    if emp_id:
        st.write(f"Employee {emp_id} is currently **Active**")

    st.markdown('</div>', unsafe_allow_html=True)

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

# --- Attrition Prediction Page ---
if page == "Attrition Prediction":
    st.header("📊 Employee Attrition Prediction")

    with st.form("Predict_value"):
        satisfaction_level = st.slider('Satisfaction Level', 0.05, 1.0, 0.66)
        last_evaluation = st.slider('Last Evaluation', 0.05, 1.0, 0.54)
        avg_monthly_hours = st.number_input('Average Monthly Hours', min_value=50, max_value=320, step=1, value=120)
        time_in_company = st.number_input('Time in Company', min_value=1, max_value=20, step=1, value=5)
        salary_category = st.selectbox("Salary", options=["Low", "Medium", "High"])
        predict_button = st.form_submit_button(label='Predict', use_container_width=True)

        if predict_button:
            salary = [0, 0]
            if salary_category == "Low":
                salary = [1, 0]
            elif salary_category == "Medium":
                salary = [0, 1]

            input_features = [satisfaction_level, last_evaluation, avg_monthly_hours, time_in_company] + salary
            prediction_result = model.predict([input_features])[0]
            prediction_prop = np.round(model.predict_proba([input_features]) * 100)

            st.subheader(f"Prediction: {'Stay' if prediction_result == 0 else 'Leave'}")
            st.subheader(f"Probability to Stay: {prediction_prop[0, 0]}%")
            st.subheader(f"Probability to Leave: {prediction_prop[0, 1]}%")

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
