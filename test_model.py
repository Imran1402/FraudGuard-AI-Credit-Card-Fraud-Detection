
import pandas as pd
import joblib

# Load the saved AI model
model = joblib.load("fraud_model.pkl")

print("AI model loaded successfully!")

# Load transaction dataset
data = pd.read_csv("data/creditcard.csv")

# Select one transaction
transaction = data.drop("Class", axis=1).iloc[[0]]

# Predict transaction
prediction = model.predict(transaction)

# Get fraud probability
probability = model.predict_proba(transaction)[0][1]

print("\nTRANSACTION ANALYSIS")

if prediction[0] == 1:
    print("Result: Potential Fraud")
else:
    print("Result: Legitimate")

print("Fraud Probability:", round(probability * 100, 2), "%")

print("Actual Dataset Label:", data["Class"].iloc[0])

print("\nPrediction test completed!")
