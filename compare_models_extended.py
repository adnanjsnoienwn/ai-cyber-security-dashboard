
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("default")

TRAIN_PATH = Path("data/UNSW_NB15_training-set.csv")
TEST_PATH = Path("data/UNSW_NB15_testing-set.csv")
MODEL_PATH = Path("models/intrusion_model.joblib")
OUTPUT_PATH = Path("model_comparison_extended.csv")

for path in (TRAIN_PATH, TEST_PATH, MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

drop_columns = ["id", "attack_cat", "label"]
X_train = train_df.drop(columns=drop_columns, errors="ignore")
y_train = train_df["label"]
X_test = test_df.drop(columns=drop_columns, errors="ignore")
y_test = test_df["label"]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test feature columns do not match.")

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")

# Reuse the original model's preprocessing configuration.
saved_model = joblib.load(MODEL_PATH)
preprocessor_template = saved_model.named_steps["preprocessor"]

models = {
    "Random Forest (saved)": saved_model,
    "Decision Tree": Pipeline([
        ("preprocessor", clone(preprocessor_template)),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        )),
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", clone(preprocessor_template)),
        ("classifier", XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(
                (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            ),
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ]),
    "Logistic Regression": Pipeline([
        ("preprocessor", clone(preprocessor_template)),
        ("classifier", LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
}

results = []

for name, model in models.items():
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    try:
        if name == "Random Forest (saved)":
            training_seconds = None
            print("Using existing saved model; not retraining it.")
        else:
            start = time.time()
            model.fit(X_train, y_train)
            training_seconds = round(time.time() - start, 2)

        start = time.time()
        predictions = model.predict(X_test)
        evaluation_seconds = round(time.time() - start, 2)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Training Time (seconds)": training_seconds,
            "Evaluation Time (seconds)": evaluation_seconds,
            "Status": "Success",
        }
        results.append(metrics)

        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        ))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        results.append({
            "Model": name,
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1 Score": None,
            "Training Time (seconds)": None,
            "Evaluation Time (seconds)": None,
            "Status": f"Failed: {type(exc).__name__}: {exc}",
        })

comparison = pd.DataFrame(results)
comparison.to_csv(OUTPUT_PATH, index=False)

print("\nMODEL COMPARISON SUMMARY")
print(comparison.to_string(index=False))
print(f"\nResults saved to: {OUTPUT_PATH}")
print("The dashboard and saved Random Forest model were not replaced.")
