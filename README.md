#  Loan Approval Prediction App
A machine learning-powered web application that predicts whether a loan application will be approved based on applicant and financial details. This app is built with **Streamlit** for an interactive interface and uses a trained model to deliver real-time predictions. 

## Overview

The Loan Approval Prediction App leverages historical loan data to provide insights into whether a loan application is likely to be approved. The app includes advanced feature engineering, such as calculating total income, monthly instalments, and performing logarithmic transformations to handle skewed financial data.

## Features 

- #### Interactive Interface:
  Users can enter their personal and financial details (e.g., gender, marital status, income, loan amount, etc.) through a user-friendly interface.
- #### Real-Time Predictions:  
  The app predicts loan approval status instantly using a trained Logistic Regression model (or your chosen model).
- #### Advanced Data Transformation: 
  Features like Total Income, Monthly Instalment, and Income After Loan are computed on the fly, with logarithmic transformations applied for improved model performance.
- #### Error Handling:
  The app includes robust error handling to manage invalid inputs and edge cases.

## Tech Stack

- **Python**

- **Streamlit**

- **Scikit-learn**

- **Pandas & NumPy**

- **Joblib (for model serialization)**

