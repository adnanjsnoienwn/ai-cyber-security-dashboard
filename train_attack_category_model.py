
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"
BINARY_MODEL_FILE = MODEL_DIR / "intrusion_model.joblib"
CATEGORY_MODEL_FILE = MODEL_DIR / "attack_category_model.joblib"

print("Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Use only confirmed attack records for attack-type classification.
train_df = train_df[
    (train_df["label"] == 1)
    & (train_df["attack_cat"].notna())
    & (train_df["attack_cat"] != "Normal")
].copy()

test_df = test_df[
    (test_df["label"] == 1)
    & (test_df["attack_cat"].notna())
    & (test_df["attack_cat"] != "Normal")
].copy()

excluded = ["id", "label", "attack_cat"]
X_train = train_df.drop(columns=excluded, errors="ignore")
y_train = train_df["attack_cat"].astype(str)

X_test = test_df.drop(columns=excluded, errors="ignore")
y_test = test_df["attack_cat"].astype(str)

print(f"Training attack records: {len(X_train):,}")
print(f"Testing attack records:  {len(X_test):,}")
print(f"Attack categories: {sorted(y_train.unique())}")

print("\nLoading existing binary model's preprocessing...")
binary_pipeline = joblib.load(BINARY_MODEL_FILE)

# Reuse the existing preprocessing design without changing the binary model.
preprocessor = clone(binary_pipeline.named_steps["preprocessor"])

category_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    ),
])

print("\nTraining attack-category model...")
start = time.time()
category_pipeline.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.1f} seconds.")

print("\nEvaluating on the separate test dataset...")
predictions = category_pipeline.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Macro F1: {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=sorted(y_train.unique()),
        zero_division=0,
    )
)

print("\nConfusion matrix (rows=true, columns=predicted):")
categories = sorted(y_train.unique())
print(pd.DataFrame(
    confusion_matrix(y_test, predictions, labels=categories),
    index=categories,
    columns=categories,
).to_string())

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(category_pipeline, CATEGORY_MODEL_FILE)

print(f"\nSaved separate model to: {CATEGORY_MODEL_FILE}")
print("The existing binary intrusion model was not modified.")
