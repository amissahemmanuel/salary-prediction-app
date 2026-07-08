import streamlit as st
import pandas as pd
import joblib


from feature_engineering import SalaryFeatureEngineer


# Load complete pipeline
model = joblib.load("final_salary_model.pkl")


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰"
)

st.title("💰 Employee Salary Prediction")

st.write(
    "Enter employee details below to predict monthly salary."
)


# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=5
)

company_years = st.number_input(
    "Years at Company",
    min_value=0,
    max_value=50,
    value=3
)

department = st.selectbox(
    "Department",
    [
        "Engineering",
        "Finance",
        "HR",
        "Marketing",
        "Operations",
        "Sales"
    ]
)

education = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor",
        "Master",
        "PhD"
    ]
)

performance = st.slider(
    "Performance Rating",
    min_value=1,
    max_value=5,
    value=3
)

monthly_hours = st.number_input(
    "Monthly Hours Worked",
    min_value=50,
    max_value=350,
    value=160
)


# -----------------------------
# Create Raw Input DataFrame
# -----------------------------
# The pipeline will handle:
# - Education encoding
# - Department encoding
# - Feature engineering
# - Scaling

input_data = pd.DataFrame({

    "Age": [age],

    "YearsExperience": [experience],

    "YearsAtCompany": [company_years],

    "Department": [department],

    "EducationLevel": [education],

    "PerformanceRating": [performance],

    "MonthlyHoursWorked": [monthly_hours]

})


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Salary"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Monthly Salary: ${prediction:,.2f}"
    )

