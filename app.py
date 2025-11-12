import streamlit as st
import pandas as pd
import joblib

model = joblib.load("results/logistic_fraud_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.header("Manual Transaction Entry")
st.write("Enter transaction details manually to predict fraud:")

feature_names = ['Time'] + ['V' + str(i) for i in range(1, 29)] + ['Amount']
input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict Single Transaction"):
    proba = model.predict_proba(input_df)[0][1]  # probability of class 1 (fraud)
    result = model.predict(input_df)[0]
    
    st.write(f"**Fraud Probability:** {proba*100:.2f}%")

    if result == 1:
        st.error("This transaction is predicted as FRAUD!")
    else:
        st.success("This transaction is predicted as LEGITIMATE.")
