
import joblib
import pandas as pd

# Saved model load karna
model = joblib.load("models/fraud_detection_model.pkl")

# Transaction details user se lena
print("\n--- Bank Transaction Fraud Detection ---")

credit_score = int(input("Credit score (300-900): "))
has_loan = input("Has loan? (Yes/No): ").strip().title()
loan_type = input("Loan type: ").strip()
emi_amount = float(input("EMI amount: "))
transaction_status = input("Transaction status: ").strip()
channel = input("Channel (UPI/Net_Banking/ATM): ").strip()
kyc_status = input("KYC status (Verified/Pending/Expired): ").strip()
transaction_hour = int(input("Transaction hour (0-23): "))

# Model ke liye data ko table format mein rakhna
transaction = pd.DataFrame([{
    "credit_score": credit_score,
    "has_loan": has_loan,
    "loan_type": loan_type,
    "emi_amount": emi_amount,
    "transaction_status": transaction_status,
    "channel": channel,
    "kyc_status": kyc_status,
    "transaction_hour": transaction_hour
}])

# Fraud prediction
prediction = model.predict(transaction)[0]

print("\n--- Prediction Result ---")

if prediction == 1:
    print("⚠️ Potential FRAUD detected!")
else:
    print("✅ Transaction classified as NON-FRAUD.")
