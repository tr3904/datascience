#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("../model/churn_model.pkl", "rb"))

st.title("Customer Churn Prediction")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (months)", 1, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Convert input
gender = 1 if gender == "Male" else 0
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract = contract_map[contract]

input_data = pd.DataFrame([{
    "gender": gender,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract
}])

# Prediction
if st.button("Predict"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer will stay ✅")

