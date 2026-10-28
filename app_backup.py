
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AI Cyber Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = "models/intrusion_model.joblib"
DB_PATH = "alerts.db"

st.title("🛡️ AI-Based Cyber Security Dashboard")
st.caption("Network Intrusion Detection | Random Forest | UNSW-NB15")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prediction TEXT,
                severity TEXT,
                confidence REAL
            )
        """)

def save_alerts(results):
    alerts = results[results["Prediction"] == "Attack"]
    if alerts.empty:
        return

    rows = [
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            row["Prediction"],
            row["Severity"],
            float(row["Attack Probability"]),
        )
        for _, row in alerts.iterrows()
    ]

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany("""
            INSERT INTO alerts (timestamp, prediction, severity, confidence)
            VALUES (?, ?, ?, ?)
        """, rows)

init_db()

with st.sidebar:
    st.header("Dashboard Controls")
    st.write("**Model:** Random Forest")
    st.write("**Dataset:** UNSW-NB15")
    st.write("**Mode:** CSV / benchmark analysis")
    st.divider()
    st.info(
        "Upload a CSV containing the network feature columns used "
        "during training. The model does not capture live network traffic."
    )

tab1, tab2, tab3 = st.tabs(
    ["📊 Overview", "🔍 Analyze Traffic", "🚨 Alert History"]
)

with tab1:
    st.subheader("Model Evaluation")

    test_path = "data/UNSW_NB15_testing-set.csv"

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(
            columns=["id", "attack_cat", "label"],
            errors="ignore",
        )
        y_test = test_df["label"]
        predictions = model.predict(X_test)

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
        c2.metric(
            "Attack Precision",
            f"{precision_score(y_test, predictions, zero_division=0):.2%}",
        )
        c3.metric(
            "Attack Recall",
            f"{recall_score(y_test, predictions, zero_division=0):.2%}",
        )
        c4.metric(
            "Attack F1-score",
            f"{f1_score(y_test, predictions, zero_division=0):.2%}",
        )

        st.subheader("Actual vs Predicted Traffic")
        chart_df = pd.DataFrame({
            "Class": ["Normal", "Attack"],
            "Records": [
                int((y_test == 0).sum()),
                int((y_test == 1).sum()),
            ],
        })
        st.plotly_chart(
            px.bar(
                chart_df,
                x="Class",
                y="Records",
                color="Class",
                title="Actual Test Dataset Distribution",
            ),
            use_container_width=True,
        )
    else:
        st.warning("Test dataset not found in the data folder.")

with tab2:
    st.subheader("Analyze Network Traffic")
    uploaded_file = st.file_uploader(
        "Upload a network traffic CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            required_columns = list(model.feature_names_in_)
            missing_columns = [
                col for col in required_columns
                if col not in uploaded_df.columns
            ]

            if missing_columns:
                st.error(
                    "This CSV is missing required model features: "
                    + ", ".join(missing_columns)
                )
            else:
                X_input = uploaded_df[required_columns].copy()
                predictions = model.predict(X_input)

                probabilities = model.predict_proba(X_input)[:, 1]

                results = uploaded_df.copy()
                results["Prediction"] = [
                    "Attack" if p == 1 else "Normal"
                    for p in predictions
                ]
                results["Attack Probability"] = probabilities

                def severity(probability, prediction):
                    if prediction == "Normal":
                        return "Low"
                    if probability >= 0.90:
                        return "Critical"
                    if probability >= 0.70:
                        return "High"
                    if probability >= 0.40:
                        return "Medium"
                    return "Low"

                results["Severity"] = [
                    severity(prob, pred)
                    for prob, pred in zip(
                        results["Attack Probability"],
                        results["Prediction"],
                    )
                ]

                total = len(results)
                attacks = int((results["Prediction"] == "Attack").sum())
                normal = total - attacks

                c1, c2, c3 = st.columns(3)
                c1.metric("Records Analyzed", f"{total:,}")
                c2.metric("Predicted Attacks", f"{attacks:,}")
                c3.metric("Predicted Normal", f"{normal:,}")

                st.subheader("Prediction Distribution")
                distribution = results["Prediction"].value_counts()
                st.plotly_chart(
                    px.pie(
                        values=distribution.values,
                        names=distribution.index,
                        title="Normal vs Attack",
                    ),
                    use_container_width=True,
                )

                st.subheader("Analyzed Records")
                st.dataframe(results, use_container_width=True)

                csv_data = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="intrusion_predictions.csv",
                    mime="text/csv",
                )

                if st.button("Save detected attacks to alert history"):
                    save_alerts(results)
                    st.success("Attack alerts saved to the local database.")

        except Exception as error:
            st.error(f"Could not analyze this CSV: {error}")

with tab3:
    st.subheader("Stored Attack Alerts")

    with sqlite3.connect(DB_PATH) as conn:
        history = pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY id DESC",
            conn,
        )

    if history.empty:
        st.info("No alerts have been saved yet. Analyze a CSV and save its attacks.")
    else:
        st.metric("Stored Alerts", len(history))
        st.dataframe(history, use_container_width=True)

        st.download_button(
            "Download Alert History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="alert_history.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Academic prototype using benchmark/CSV data. Predictions are not "
    "proof of an attack; validate findings before taking security action."
)
