# Importing ToolKits
import re
import vizualizations
import prediction

from time import sleep
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

import streamlit as st
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
import warnings

def run():
    st.set_page_config(
        page_title="Employee Retention Prediction",
        page_icon="📊",
        layout="wide"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_data(the_file_path):
        df = pd.read_csv(the_file_path)
        df.columns = df.columns.str.replace(" ",  "_").str.replace(".", "")
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    @st.cache_data
    def load_the_model(model_path):
        return pd.read_pickle(model_path)
        
    df = load_data("HR_comma_sep.csv")
    model = load_the_model("random_forest_employee_retention_v1.pkl")

    st.markdown(
        """
    <style>
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px Arial;
         text-align: center;
         margin-bottom: 15px;
         color: #005DAA; /* Genentech Blue */
         }
         div[data-testid=stSidebarContent] {
         background-color: #E6EEF8; /* Light Blue */
         border-right: 4px solid #005DAA;
         padding: 8px!important;
         }
         div.block-containers{
            padding-top: 0.5rem;
         }
         div[data-testid=stFormSubmitButton]> button{
            width: 40%;
            background-color: #005DAA;
            border: 2px solid #004080;
            padding: 12px;
            border-radius: 30px;
            opacity: 0.9;
            color: white;
        }
        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            background-color: #003366;
        }
    </style>
    """,
        unsafe_allow_html=True
    )

    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},
        "icon": {"color": "#005DAA", "font-size": "16px"},
        "nav-link": {"color": "#003366", "font-size": "18px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
        "nav-link-selected": {"background-color": "#0073E6", "font-size": "15px"},
    }

    header = st.container()
    content = st.container()

    with st.sidebar:
        st.title("Employee Retention Prediction")
        page = option_menu(
            menu_title=None,
            options=['Home', 'Visualizations', 'Prediction'],
            icons=['house', 'bar-chart', "graph-up-arrow"],
            menu_icon="cast",
            default_index=0,
            styles=side_bar_options_style
        )

    if page == "Home":
        with header:
            st.header('Employee Retention Analysis')
        with content:
            st.dataframe(df.sample(frac=0.25, random_state=35).reset_index(drop=True), use_container_width=True)

    if page == "Visualizations":
        with header:
            st.header("Data Visualizations")
        with content:
            vizualizations.create_vizualization(df, viz_type="box", data_type="number")
            vizualizations.create_vizualization(df, viz_type="bar", data_type="object")
            vizualizations.create_vizualization(df, viz_type="pie")
            st.plotly_chart(vizualizations.create_heat_map(df), use_container_width=True)

    if page == "Prediction":
        with header:
            st.header("Employee Attrition Prediction")
            prediction_option = option_menu(menu_title=None, options=["One Value", 'From File'],
                                            icons=["input-cursor", "file-earmark-spreadsheet"], menu_icon="cast", default_index=0)
        with content:
            if prediction_option == "One Value":
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
                        new_data = [satisfaction_level, last_evaluation, avg_monthly_hours, time_in_company] + salary
                        predicted_value = model.predict([new_data])[0]
                        prediction_prop = np.round(model.predict_proba([new_data]) * 100)

                        st.subheader(f"Prediction: {'Stay' if predicted_value == 0 else 'Leave'}")
                        st.subheader(f"Probability to Stay: {prediction_prop[0, 0]}%")
                        st.subheader(f"Probability to Leave: {prediction_prop[0, 1]}%")

run()
