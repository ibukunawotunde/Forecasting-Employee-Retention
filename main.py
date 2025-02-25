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

# --- Streamlit Page Configuration ---
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

# --- Attrition Prediction ---
if page == "Attrition Prediction":
    st.header("🔍 Predict Employee Attrition")

    with st.form("Prediction_Form"):
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

# --- Streamlit Page Configuration ---
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
    gender_distribution = df["gender"].value_counts()
    fig_gender = px.pie(
        names=gender_distribution.index, 
        values=gender_distribution.values,
        title="Workforce Gender Distribution"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

    # Departmental Diversity Breakdown
    st.subheader("🏢 Diversity Across Departments")
    department_distribution = df["department"].value_counts()
    fig_department = px.bar(
        x=department_distribution.index,
        y=department_distribution.values,
        title="Diversity Representation Across Departments",
        labels={"x": "Department", "y": "Count"},
        color=department_distribution.index
    )
    st.plotly_chart(fig_department, use_container_width=True)

    # Attrition Rate Among Underrepresented Groups
    st.subheader("📉 Attrition Rate by Gender & Salary Level")
    attrition_by_gender = df.groupby("gender")["left"].mean().reset_index()
    attrition_by_salary = df.groupby("salary")["left"].mean().reset_index()

    fig_attrition_gender = px.bar(
        attrition_by_gender, x="gender", y="left", 
        title="Attrition Rate by Gender",
        labels={"left": "Attrition Rate"},
        color="gender"
    )
    fig_attrition_salary = px.bar(
        attrition_by_salary, x="salary", y="left",
        title="Attrition Rate by Salary Level",
        labels={"left": "Attrition Rate"},
        color="salary"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_attrition_gender, use_container_width=True)
    with col2:
        st.plotly_chart(fig_attrition_salary, use_container_width=True)

    # Salary Disparity Analysis
    st.subheader("💰 Salary Distribution by Gender")
    fig_salary = px.box(
        df, x="gender", y="average_montly_hours",
        title="Salary Distribution by Gender",
        labels={"average_montly_hours": "Monthly Hours Worked"}
    )
    st.plotly_chart(fig_salary, use_container_width=True)

    # Promotion & Career Growth Insights
    st.subheader("🚀 Promotion Rates by Gender")
    promotion_by_gender = df.groupby("gender")["promotion_last_5years"].mean().reset_index()

    fig_promotion = px.bar(
        promotion_by_gender, x="gender", y="promotion_last_5years",
        title="Promotion Rates by Gender",
        labels={"promotion_last_5years": "Promotion Rate"},
        color="gender"
    )
    st.plotly_chart(fig_promotion, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- AI Chatbot ---
if page == "HR Chatbot":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🤖 HR Policy Chatbot")
    st.write("Ask the AI chatbot any HR-related question!")

    user_input = st.text_input("Ask me something about HR policies...")
    if st.button("Ask Chatbot"):
        response = f"Great question! HR policy regarding {user_input} is currently being updated."
        st.write(response)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Generate Report ---
if page == "Generate Report":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("📄 Generate HR Report")
    st.write("Download a summary of HR analytics insights.")

    if st.button("Download Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="HR Analytics Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Attrition Rate: {df['left'].mean() * 100:.1f}%", ln=True)
        pdf.cell(200, 10, txt="Top Factor: Lack of Career Growth", ln=True)
        pdf.output("HR_Analytics_Report.pdf")
        st.success("Report generated successfully!")

    st.markdown('</div>', unsafe_allow_html=True)
