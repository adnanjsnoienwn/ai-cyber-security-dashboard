
import os
import time
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_DIR = "data"
MODEL_DIR = "models"

TRAIN_FILE = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading datasets...")
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# Binary target: 0 = Normal, 1 = Attack
X_train = train.drop(columns=["id", "attack_cat", "label"])
y_train = train["label"]

X_test = test.drop(columns=["id", "attack_cat", "label"])
y_test = test["label"]

# Identify categorical and numerical columns
categorical = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        ("numerical", "passthrough", numerical),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]
)

print("Training Random Forest model...")
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.1f} seconds.")

print("\nEvaluating on the testing dataset...")
predictions = model.predict(X_test)

print(f"\nAccuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

model_path = os.path.join(MODEL_DIR, "intrusion_model.joblib")
joblib.dump(model, model_path)

print(f"\nSaved model to: {model_path}")
print("STEP 4 COMPLETE")
