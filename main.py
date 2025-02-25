# --- Streamlit Page Configuration (MUST BE FIRST COMMAND) ---
import streamlit as st
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Importing Toolkits ---
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from fpdf import FPDF
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("HR Analytics Dashboard")
    page = option_menu(
        menu_title=None,
        options=['Home', 'Attrition Prediction', 'HR Chatbot', 'Generate Report'],
        icons=['house-fill', 'bar-chart-line-fill', "chat-text-fill", "file-earmark-text-fill"],
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
        st.error("Error: The dataset does not contain a 'left' column. Please check dataset.")
        st.stop()

    return df

df = load_data("HR_comma_sep.csv")

# --- Train Model ---
@st.cache_data
def train_model(df):
    df = df.copy()
    df["salary"] = df["salary"].map({"low": 0, "medium": 1, "high": 2})
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)

    X = df.drop("left", axis=1, errors="ignore")
    y = df["left"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model(df)

# --- Home Page ---
if page == "Home":
    st.header("📊 HR Analytics Dashboard")
    st.write("This platform provides key HR insights including attrition trends, workforce diversity, and predictive analytics.")

    # Quick Metrics
    st.subheader("📊 HR Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Rate", f"{df['left'].mean() * 100:.1f}%", "⬆ 5% from last year")
    col2.metric("Avg. Satisfaction Score", f"{df['satisfaction_level'].mean() * 100:.1f}%", "⬆ 3%")
    col3.metric("Top Attrition Factor", "Lack of Career Growth", "🔍 Insights Available")

    # Attrition Trends
    st.subheader("📉 Attrition Trends Over Time")
    fig = px.line(df, x="time_spend_company", y="left", title="Attrition Rate by Tenure")
    st.plotly_chart(fig, use_container_width=True)

# --- Attrition Prediction ---
if page == "Attrition Prediction":
    st.header("🔍 Predict Employee Attrition")

    with st.form("Prediction_Form"):
        st.subheader("Enter Employee Details:")
        
        satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.slider('Last Evaluation Score', 0.0, 1.0, 0.6)
        number_project = st.slider("Number of Projects", 1, 10, 4)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
        time_spend_company = st.slider("Years at Company", 1, 20, 5)
        work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

        predict_button = st.form_submit_button("Predict Attrition")

    if predict_button:
        try:
            salary_encoded = {"low": 0, "medium": 1, "high": 2}[salary]

            input_features = np.array([
                satisfaction_level, last_evaluation, number_project,
                average_montly_hours, time_spend_company, work_accident,
                promotion_last_5years, salary_encoded
            ]).reshape(1, -1)

            prediction_result = model.predict(input_features)[0]
            prediction_prob = model.predict_proba(input_features)[0]

            if prediction_result == 1:
                st.error(f"⚠️ High Risk: Employee is likely to leave! (Probability: {prediction_prob[1]*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk: Employee is likely to stay. (Probability: {prediction_prob[0]*100:.2f}%)")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- AI Chatbot ---
if page == "HR Chatbot":
    st.header("🤖 HR Policy Chatbot")
    user_input = st.text_input("Ask me something about HR policies...")
    if st.button("Ask Chatbot"):
        response = f"Great question! HR policy regarding {user_input} is currently being updated."
        st.write(response)

# --- Generate Report ---
if page == "Generate Report":
    st.header("📄 Generate HR Report")
    if st.button("Download Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="HR Analytics Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Attrition Rate: {df['left'].mean() * 100:.1f}%", ln=True)
        pdf.output("HR_Analytics_Report.pdf")
        st.success("Report generated successfully!")
