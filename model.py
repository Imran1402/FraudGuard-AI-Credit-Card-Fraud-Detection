
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# STEP 1: LOAD DATASET

data = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")

print("Dataset shape:", data.shape)

# STEP 2: SEPARATE INPUTS AND OUTPUT

X = data.drop("Class", axis=1)
y = data["Class"]

# STEP 3: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 4: CREATE MACHINE LEARNING MODEL

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# STEP 5: TRAIN MODEL

print("Training model...")

model.fit(X_train, y_train)

# STEP 6: PREDICTIONS

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# STEP 7: EVALUATION

print("\nMODEL PERFORMANCE")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("ROC-AUC Score:")
print(roc_auc_score(y_test, probabilities))

print("\nModel training completed!")

import joblib

joblib.dump(model, "fraud_model.pkl")

print("AI model saved successfully!")
