import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn

# Load the trained model
model = joblib.load("best_loan_approval_model.pkl")

# Function to convert loan amount to thousands for the model
def convert_loan_amount(user_input_amount):
    return user_input_amount / 1000 

# Function to convert loan term from years to months
def convert_loan_term_in_years_to_months(user_input_years):
    return user_input_years * 12  

# Function to calculate log-transformed features
def calculate_log_features(applicant_income, coapplicant_income, loan_amount, loan_amount_term):
    try:
        total_income = applicant_income + coapplicant_income
        monthly_installement = (loan_amount * 1000) / loan_amount_term if loan_amount_term > 0 else 0
        income_after_loan = total_income - monthly_installement if total_income > monthly_installement else 0
        
        # Log transformations (handling non-positive values by adding 1)
        log_total_income = np.log(total_income + 1)
        log_monthly_installement = np.log(monthly_installement + 1)
        log_income_after_loan = np.log(income_after_loan + 1)
        
        return log_total_income, log_monthly_installement, log_income_after_loan
    except Exception as e:
        st.warning(f"Error in calculating log features: {str(e)}")
        return None, None, None

# Define function to make predictions based on user input
def get_prediction(input_data):
    try:
        # Perform prediction
        prediction = model.predict(input_data)
        # Display result
        st.text(f"The predicted loan approval status is: {'Approved' if prediction[0] == 1 else 'Not Approved'}")
    except Exception as e:
        # Handle any prediction errors
        st.text(f"Error in prediction: {str(e)}")

# Main function for the Streamlit app
def main():
    st.title("Loan Approval Prediction App")
    st.write("This app predicts whether a loan application will be approved or not based on applicant data.")

    # Get user inputs    
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount_input = st.number_input("Loan Amount", min_value=0)
    loan_amount = convert_loan_amount(loan_amount_input)  
    loan_term_years = st.number_input("Loan Amount Term (in years)", min_value=0)
    loan_amount_term = convert_loan_term_in_years_to_months(loan_term_years)  
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Calculate log-transformed features
    log_total_income, log_monthly_installement, log_income_after_loan = calculate_log_features(
        applicant_income, coapplicant_income, loan_amount, loan_amount_term
    )

    # Check if all log features were calculated successfully
    if None in [log_total_income, log_monthly_installement, log_income_after_loan]:
        st.warning("Cannot proceed with prediction due to missing log-transformed features.")
        return

    # Prepare input data
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area],
        "Log_Total_Income": [log_total_income],
        "Log_Monthly_Installement": [log_monthly_installement],
        "Log_Income_After_Loan": [log_income_after_loan]
    })

    # Button to make a prediction
    if st.button("Predict"):
        get_prediction(input_data)

# Run the app
main()
