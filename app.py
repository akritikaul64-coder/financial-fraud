
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️")

st.title("🛡️ Financial Fraud Detection")

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_detection_model.pkl")

try:
    model = load_model()
    df = pd.read_csv("data/indian_banking_transactions.csv")

    excluded = ["is_fraud", "transaction_id", "customer_id"]
    columns = [c for c in df.columns if c not in excluded]

    st.write("Enter transaction details:")

    with st.form("transaction_form"):
        values = {}

        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                values[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(median) if pd.notna(median) else 0.0
                )
            else:
                options = df[col].dropna().unique().tolist()
                if options:
                    values[col] = st.selectbox(
                        col.replace("_", " ").title(), options
                    )
                else:
                    values[col] = None

        submitted = st.form_submit_button("Predict Transaction")

    if submitted:
        input_df = pd.DataFrame([values], columns=columns)
    

        if prediction == 1:
            st.error("Potential Fraud Transaction Detected")
        else:
            st.success("Transaction Classified as Normal")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check that the model and dataset files exist.")
