
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("default")

TRAIN_PATH = Path("data/UNSW_NB15_training-set.csv")
TEST_PATH = Path("data/UNSW_NB15_testing-set.csv")
MODEL_PATH = Path("models/intrusion_model.joblib")
OUTPUT_PATH = Path("feature_imbalance_results.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

drop_columns = ["id", "attack_cat", "label"]
X_train = train_df.drop(columns=drop_columns, errors="ignore")
y_train = train_df["label"]
X_test = test_df.drop(columns=drop_columns, errors="ignore")
y_test = test_df["label"]

saved_model = joblib.load(MODEL_PATH)
preprocessor_template = saved_model.named_steps["preprocessor"]

results = []


def record_result(name, predictions, train_seconds):
    results.append({
        "Experiment": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "Training Time (seconds)": train_seconds,
    })


# Baseline: saved Random Forest
print("Evaluating saved Random Forest baseline...")
predictions = saved_model.predict(X_test)
record_result("Random Forest baseline", predictions, None)

# Feature selection: train a selector on training data only.
# The selector and classifier are fitted inside a pipeline.
print("\nTraining feature-selection experiment...")
selector_model = Pipeline([
    ("preprocessor", clone(preprocessor_template)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

selected_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor_template)),
    ("feature_selection", SelectFromModel(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        threshold="median",
    )),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )),
])

start = time.time()
selected_pipeline.fit(X_train, y_train)
selection_seconds = round(time.time() - start, 2)
selected_predictions = selected_pipeline.predict(X_test)
record_result(
    "XGBoost with model-based feature selection",
    selected_predictions,
    selection_seconds,
)

preprocessor = selected_pipeline.named_steps["preprocessor"]
selector = selected_pipeline.named_steps["feature_selection"]
feature_count = len(selector.get_support())
selected_count = int(selector.get_support().sum())
print(
    f"Encoded features retained: {selected_count} of {feature_count}"
)

# SMOTE: fit preprocessing on training data, then resample only training rows.
print("\nPreparing training features for SMOTE...")
preprocessor_smote = clone(preprocessor_template)
X_train_encoded = preprocessor_smote.fit_transform(X_train)
X_test_encoded = preprocessor_smote.transform(X_test)

print("Original training class counts:")
print(y_train.value_counts().sort_index().to_dict())

smote = SMOTE(random_state=42, sampling_strategy="auto")
start = time.time()
X_resampled, y_resampled = smote.fit_resample(
    X_train_encoded, y_train
)
smote_seconds = round(time.time() - start, 2)

print("Resampled training class counts:")
print(pd.Series(y_resampled).value_counts().sort_index().to_dict())

# Compare class-weighted XGBoost with SMOTE-trained XGBoost.
print("\nTraining XGBoost with SMOTE...")
smote_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

start = time.time()
smote_model.fit(X_resampled, y_resampled)
smote_train_seconds = round(time.time() - start, 2) + smote_seconds

smote_predictions = smote_model.predict(X_test_encoded)
record_result(
    "XGBoost with SMOTE",
    smote_predictions,
    smote_train_seconds,
)

comparison = pd.DataFrame(results)
comparison.to_csv(OUTPUT_PATH, index=False)

print("\nFEATURE SELECTION AND IMBALANCE RESULTS")
print(comparison.to_string(index=False))
print(f"\nResults saved to {OUTPUT_PATH}")
print("The saved dashboard model was not modified.")
