
import os
import time

import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"
MODEL_PATH = "models/intrusion_model.joblib"

if not os.path.exists(DATA_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        "Dataset files not found. Check the data folder."
    )

print("Loading UNSW-NB15 datasets...")
train_df = pd.read_csv(DATA_PATH)
test_df = pd.read_csv(TEST_PATH)

drop_columns = ["id", "attack_cat", "label"]

X_train = train_df.drop(columns=drop_columns, errors="ignore")
y_train = train_df["label"]

X_test = test_df.drop(columns=drop_columns, errors="ignore")
y_test = test_df["label"]

results = []

def evaluate_model(name, model, X, y):
    predictions = model.predict(X)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
    }

    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)
    print(classification_report(
        y, predictions,
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

    return metrics


print("\nLoading your existing Random Forest model...")
random_forest = joblib.load(MODEL_PATH)

print("\nEvaluating Random Forest...")
start = time.time()
rf_metrics = evaluate_model(
    "Random Forest", random_forest, X_test, y_test
)
rf_metrics["Training Time (seconds)"] = None
rf_metrics["Evaluation Time (seconds)"] = round(
    time.time() - start, 2
)
results.append(rf_metrics)

print("\nTraining Decision Tree...")


preprocessor = clone(random_forest.named_steps["preprocessor"])

decision_tree = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    )),
])
start = time.time()
decision_tree.fit(X_train, y_train)
dt_training_time = round(time.time() - start, 2)

print("\nEvaluating Decision Tree...")
start = time.time()
dt_metrics = evaluate_model(
    "Decision Tree", decision_tree, X_test, y_test
)
dt_metrics["Training Time (seconds)"] = dt_training_time
dt_metrics["Evaluation Time (seconds)"] = round(
    time.time() - start, 2
)
results.append(dt_metrics)

comparison = pd.DataFrame(results)
comparison.to_csv("model_comparison.csv", index=False)

print("\n\nMODEL COMPARISON SUMMARY")
print(comparison.to_string(index=False))
print("\nSaved comparison results to model_comparison.csv")
