import streamlit as st
import pickle
import pandas as pd

# Load model
with open('full_pipeline', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Status Prediction", page_icon="💰", layout="centered")
st.title("💰 Loan Approval Prediction App")

# Inputs
Married = st.selectbox("Married", ["Yes", "No"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate", "HSC"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=500)
LoanAmount = st.number_input("Loan Amount", min_value=0, step=50)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

if st.button("Predict"):
    input_df = pd.DataFrame({
        'Married': [Married],
        'Education': [Education],
        'ApplicantIncome': [ApplicantIncome],
        'LoanAmount': [LoanAmount],
        'Credit_History': [Credit_History]
    })

    # Make prediction
    try:
        score = model.predict(input_df)[0]  # regression output (e.g. 0.72)

        if score > 0.5:
            result = "✅ Loan Approved"
            color = "green"
        else:
            result = "❌ Loan Rejected"
            color = "red"

        # Display result
        st.markdown(f"<h2 style='text-align:center; color:{color};'>{result}</h2>", unsafe_allow_html=True)
        st.metric(label="Predicted Score", value=f"{score:.3f}")

        # Display user input for transparency
        st.subheader("Entered Details")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
