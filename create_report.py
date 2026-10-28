
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/UNSW_NB15_testing-set.csv"
MODEL_PATH = "models/intrusion_model.joblib"
CATEGORY_MODEL_PATH = "models/attack_category_model.joblib"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading test dataset and models...")

test_df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
category_model = joblib.load(CATEGORY_MODEL_PATH)

X_test = test_df.drop(
    columns=["id", "attack_cat", "label"],
    errors="ignore",
)
y_test = test_df["label"]

# 1. Binary intrusion detection evaluation
binary_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, binary_predictions)
precision = precision_score(
    y_test, binary_predictions, zero_division=0
)
recall = recall_score(
    y_test, binary_predictions, zero_division=0
)
f1 = f1_score(
    y_test, binary_predictions, zero_division=0
)
matrix = confusion_matrix(y_test, binary_predictions)

binary_report = classification_report(
    y_test,
    binary_predictions,
    labels=[0, 1],
    target_names=["Normal", "Attack"],
    zero_division=0,
)

# 2. Attack-category evaluation on actual attack records only
attack_mask = (
    (test_df["label"] == 1)
    & test_df["attack_cat"].notna()
    & (test_df["attack_cat"] != "Normal")
)

X_attack = X_test.loc[attack_mask]
y_attack = test_df.loc[attack_mask, "attack_cat"].astype(str)

category_predictions = category_model.predict(X_attack)
category_labels = sorted(
    set(y_attack.unique())
    | set(category_predictions)
)

category_accuracy = accuracy_score(
    y_attack, category_predictions
)
category_macro_f1 = f1_score(
    y_attack,
    category_predictions,
    labels=category_labels,
    average="macro",
    zero_division=0,
)

category_report = classification_report(
    y_attack,
    category_predictions,
    labels=category_labels,
    zero_division=0,
)

category_matrix = confusion_matrix(
    y_attack,
    category_predictions,
    labels=category_labels,
)

report = f"""
AI-BASED CYBER SECURITY DASHBOARD
NETWORK INTRUSION DETECTION
MODEL EVALUATION REPORT

Dataset: UNSW-NB15 Testing Dataset
Total test records: {len(y_test):,}

1. BINARY INTRUSION DETECTION
Model: Random Forest

Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)
Precision: {precision:.4f} ({precision * 100:.2f}%)
Recall:    {recall:.4f} ({recall * 100:.2f}%)
F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)

Confusion Matrix
Rows = Actual class; columns = Predicted class

                 Predicted Normal   Predicted Attack
Actual Normal         {matrix[0, 0]:,}                {matrix[0, 1]:,}
Actual Attack         {matrix[1, 0]:,}               {matrix[1, 1]:,}

Classification Report
{binary_report}

2. MULTI-CLASS ATTACK CATEGORY CLASSIFICATION
Model: Random Forest
Evaluation records: {len(y_attack):,}
Evaluation includes actual attack records only.

Accuracy: {category_accuracy:.4f} ({category_accuracy * 100:.2f}%)
Macro F1 Score: {category_macro_f1:.4f} ({category_macro_f1 * 100:.2f}%)

Per-category Classification Report
{category_report}

Attack Category Confusion Matrix
Rows = actual category; columns = predicted category

{pd.DataFrame(
    category_matrix,
    index=category_labels,
    columns=category_labels,
).to_string()}

3. DECISION TREE VS RANDOM FOREST

Previously recorded results:
Random Forest accuracy: approximately 89.29%
Random Forest attack F1-score: approximately 90.93%
Decision Tree accuracy: approximately 86.17%
Decision Tree attack F1-score: approximately 88.49%

These comparison figures are retained from the existing evaluation.
They are not recalculated by this script.

4. INTERPRETATION AND LIMITATIONS

The binary model detects whether a traffic record is Normal or Attack.
The category model predicts an attack type for records classified as
attacks by the dashboard.

The category model is evaluated separately on actual attack records.
Its results therefore do not represent the accuracy of the entire
two-stage dashboard pipeline.

Performance may vary considerably between attack categories,
especially for categories with few training examples. The classification
report should be consulted for per-category precision, recall and F1.

These results use the UNSW-NB15 test dataset and do not guarantee
equivalent performance on live network traffic.

5. CONCLUSION

The dashboard combines binary intrusion detection with a separate
multi-class attack categorization model, alert severity assignment,
and alert history storage.

The binary Random Forest achieved the metrics shown above on the
UNSW-NB15 test dataset. The category model's performance is reported
separately to make its strengths and limitations transparent.
"""

output_path = os.path.join(
    OUTPUT_DIR,
    "model_evaluation_report.txt",
)

with open(output_path, "w", encoding="utf-8") as file:
    file.write(report)

print("\n" + "=" * 55)
print("REPORT CREATED SUCCESSFULLY")
print("=" * 55)
print(report)
print(f"\nSaved to: {output_path}")
