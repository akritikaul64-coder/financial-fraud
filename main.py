
import pandas as pd

# Dataset ki file ka path
file_path = "data/indian_banking_transactions.csv"

# CSV file ko Python mein load karna
df = pd.read_csv(file_path)

# Dataset ki pehli 5 rows dikhana
print("\n--- First 5 Transactions ---")
print(df.head())

# Dataset mein kitni rows aur columns hain
print("\n--- Dataset Shape ---")
print(df.shape)

# Saare column names dikhana
print("\n--- Column Names ---")
print(df.columns.tolist())

# Har column ka data type aur missing values
print("\n--- Dataset Information ---")
df.info()

# Fraud aur normal transactions ki counting
print("\n--- Fraud vs Normal Transactions ---")
print(df["is_fraud"].value_counts())

# Fraud aur normal transactions ka percentage
print("\n--- Percentage of Each Class ---")
print(df["is_fraud"].value_counts(normalize=True) * 100)

# Fraud column mein kaun-kaun si values hain
print("\n--- Unique Fraud Labels ---")
print(df["is_fraud"].unique())

# Missing values wale columns
print("\n--- Missing Values ---")
print(df.isnull().sum().sort_values(ascending=False))

print("\n--- Fraud and Normal Transaction Count ---")
print(df["is_fraud"].value_counts())

print("\n--- Fraud Percentage ---")
print(df["is_fraud"].value_counts(normalize=True) * 100)

print("\n--- All Dataset Columns ---")
for column in df.columns:
    print(column)

print("\n--- Unique Values in Important Columns ---")

for column in [
    "has_loan",
    "loan_type",
    "transaction_status",
    "channel",
    "kyc_status",
    "is_fraud"
]:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(15))

print("\n--- Numeric Column Summary ---")
print(
    df[["credit_score", "emi_amount", "transaction_hour"]]
    .describe()
)
print("\n--- Fraud by Channel ---")
print(pd.crosstab(df["channel"], df["is_fraud"]))

print("\n--- Fraud by Transaction Status ---")
print(pd.crosstab(df["transaction_status"], df["is_fraud"]))

print("\n--- Fraud by KYC Status ---")
print(pd.crosstab(df["kyc_status"], df["is_fraud"]))

# STEP 1: Required libraries import karo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# STEP 2: Features (X) aur target (y) banao
numeric_features = [
    "credit_score",
    "emi_amount",
    "transaction_hour"
]

categorical_features = [
    "has_loan",
    "channel",
    "kyc_status"
]

X = df[numeric_features + categorical_features].copy()
y = df["is_fraud"].astype(int)

# STEP 3: Missing values handle aur categories encode karo
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# STEP 4: Training aur testing data alag karo
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# STEP 5: Random Forest model banao
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

# STEP 6: Model train karo
print("\nModel training started...")
model.fit(X_train, y_train)
print("Model training completed!")

# STEP 7: Test data par evaluation
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Dataset load karna
file_path = "data/indian_banking_transactions.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Target column aur features
target = "is_fraud"

# IDs ko model mein use nahi karenge
drop_columns = ["is_fraud", "transaction_id", "customer_id"]
X = df.drop(columns=drop_columns, errors="ignore")
y = df[target]

# 3. Categorical aur numerical columns identify karna
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns),
        ("numerical", StandardScaler(), numerical_columns)
    ]
)

# 5. Training aur testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Model banana
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
("classifier", RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    n_jobs=-1
))
])


# 7. Model train karna
print("Training started...")
model.fit(X_train, y_train)
print("Training completed!")

# 8. Predictions aur evaluation
y_prob=model.predict_proba(X_test)[:,1]
threshold=0.30
y_pred=(y_prob >= threshold).astype(int)

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(
    y_test, y_pred, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_test, y_pred, zero_division=0
), 4))
print("F1-score:", round(f1_score(
    y_test, y_pred, zero_division=0
), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print("\nFraud Analysis:")
print("Actual fraud missed:", cm[1][0])
print("Fraud correctly detected:", cm[1][1])
print("Genuine transactions wrongly flagged:", cm[0][1])


print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_detection_model.pkl")

print("Model saved successfully!")

# Confusion Matrix aur evaluation graphs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Fraud Detection - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("Confusion matrix:")
print(cm)
print("Graph saved in outputs/confusion_matrix.png")

import os
from sklearn.metrics import classification_report

os.makedirs("outputs", exist_ok=True)

with open("outputs/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))

print("Classification report saved successfully!")

from sklearn.metrics import classification_report

with open("outputs/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))

print("Classification report saved successfully!")
with open("outputs/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))

print("Report saved successfully!")
from sklearn.metrics import classification_report

with open("outputs/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))

print("Classification report saved successfully!")