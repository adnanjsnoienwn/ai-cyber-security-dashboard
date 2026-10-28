# AI-Based Cyber Security Dashboard for Network Intrusion Detection

## 1. Project Overview

This project is a machine-learning-based dashboard that analyzes network traffic records and classifies them as normal traffic or potential attacks.

It uses the UNSW-NB15 dataset and a trained Random Forest classifier for binary intrusion detection. Decision Tree, XGBoost, and Logistic Regression models have also been evaluated for comparison.

## 2. Main Features

- CSV-based network traffic analysis
- Normal versus attack classification
- Attack probability estimation and severity assessment
- Interactive charts and prediction filtering
- SQLite-based alert storage
- Alert history with date, severity, and confidence filters
- Daily alert record visualization
- CSV export of filtered results and alert history
- Random Forest feature importance visualization
- SHAP-based explanation of individual predictions
- Model comparison and evaluation reports
- PDF report download

## 3. Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- SQLite
- Joblib
- SHAP
- ReportLab

## 4. Dataset

The project uses the UNSW-NB15 training and testing CSV files.

The model predicts a binary label:

- `0` — Normal traffic
- `1` — Potential attack

The uploaded CSV must contain the required model input features.

## 5. How to Run

1. Open PowerShell in the project directory.
2. Install the dependencies if necessary:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

4. Open the local URL displayed in the terminal, usually `http://localhost:8501`.
5. Navigate through Overview, Analyze Traffic, and Alert History.

## 6. Model Evaluation

On the supplied UNSW-NB15 testing dataset, the Random Forest model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 89.29% |
| Attack precision | 85.19% |
| Attack recall | 97.51% |
| Attack F1-score | 90.93% |

The Decision Tree comparison model achieved 86.17% accuracy and an 88.49% attack F1-score.

In a separate model comparison experiment, XGBoost achieved 91.11% accuracy and a 92.28% attack F1-score. The dashboard's deployed classifier remains Random Forest.

Detailed evaluation reports are available in the `reports` directory.

## 7. Testing

The automated model tests verify:

- Binary prediction output
- Attack probabilities between 0 and 1
- Expected preprocessing and classifier pipeline steps

The latest recorded test run passed all three tests.

## 8. Important Limitations

This is an academic prototype that analyzes CSV-based network traffic records. Predictions are not proof of malicious activity, and performance on real-world traffic may differ from performance on the benchmark dataset.

The application does not independently capture live network packets or automatically block attacks. Alert-history counts represent saved records and may include duplicate records when the same data is saved repeatedly.
