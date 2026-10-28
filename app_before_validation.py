import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(
    page_title="AI Cyber Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = "models/intrusion_model.joblib"
DB_PATH = "alerts.db"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

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
        return 0

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

    return len(rows)


def assign_severity(probability, prediction):
    if prediction == "Normal":
        return "Low"
    if probability >= 0.90:
        return "Critical"
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


init_db()

with st.sidebar:
    st.header("Dashboard Controls")
    st.write("**Model:** Random Forest")
    st.write("**Dataset:** UNSW-NB15")
    st.write("**Mode:** CSV / benchmark analysis")
    st.divider()
    st.info(
        "Upload a CSV containing the network feature columns used "
        "during training. This prototype does not capture live traffic."
    )

tab1, tab2, tab3 = st.tabs(
    ["📊 Overview", "🔍 Analyze Traffic", "🚨 Alert History"]
)


with tab1:
    st.subheader("Model Evaluation")
    st.subheader("Decision Tree vs Random Forest")

    try:
        comparison_df = pd.read_csv("model_comparison.csv")

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

        comparison_metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

        comparison_chart_df = comparison_df.melt(
            id_vars="Model",
            value_vars=comparison_metrics,
            var_name="Metric",
            value_name="Score",
        )

        st.plotly_chart(
            px.bar(
                comparison_chart_df,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Model Performance Comparison",
                range_y=[0, 1],
                text_auto=".2f",
            ),
            use_container_width=True,
        )

    except FileNotFoundError:
        st.warning(
            "Model comparison file not found. Run compare_models.py first."
        )

    st.subheader("Top 10 Important Network Features")

    try:
        feature_importances = model.named_steps["classifier"].feature_importances_
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances,
        }).sort_values("Importance", ascending=False).head(10)

        importance_df = importance_df.sort_values("Importance")

        st.plotly_chart(
            px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Features Used by the Random Forest Model",
            ),
            use_container_width=True,
        )

    except Exception as error:
        st.warning(f"Feature importance could not be displayed: {error}")

    if os.path.exists(TEST_PATH):
        test_df = pd.read_csv(TEST_PATH)
        X_test = test_df.drop(
            columns=["id", "attack_cat", "label"],
            errors="ignore",
        )
        y_test = test_df["label"]
        test_predictions = model.predict(X_test)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, test_predictions):.2%}")
        c2.metric(
            "Attack Precision",
            f"{precision_score(y_test, test_predictions, zero_division=0):.2%}",
        )
        c3.metric(
            "Attack Recall",
            f"{recall_score(y_test, test_predictions, zero_division=0):.2%}",
        )
        c4.metric(
            "Attack F1-score",
            f"{f1_score(y_test, test_predictions, zero_division=0):.2%}",
        )

        st.subheader("Actual Test Dataset Distribution")
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
                title="Normal vs Attack Records",
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
            elif uploaded_df.empty:
                st.warning("The uploaded CSV contains no records.")
            else:
                X_input = uploaded_df[required_columns].copy()
                predictions = model.predict(X_input)
                probabilities = model.predict_proba(X_input)[:, 1]

                results = uploaded_df.copy()
                results["Prediction"] = [
                    "Attack" if prediction == 1 else "Normal"
                    for prediction in predictions
                ]
                results["Attack Probability"] = probabilities
                results["Severity"] = [
                    assign_severity(probability, prediction)
                    for probability, prediction in zip(
                        results["Attack Probability"],
                        results["Prediction"],
                    )
                ]

                total = len(results)
                attacks = int((results["Prediction"] == "Attack").sum())
                normal = total - attacks
                attack_rate = attacks / total * 100

                st.subheader("Traffic Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Records Analyzed", f"{total:,}")
                c2.metric("Predicted Attacks", f"{attacks:,}")
                c3.metric("Predicted Normal", f"{normal:,}")
                c4.metric("Attack Rate", f"{attack_rate:.2f}%")

                st.caption(
                    "Attack rate is the percentage of uploaded records "
                    "classified as attacks by the model."
                )

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.subheader("Prediction Distribution")
                    prediction_counts = (
                        results["Prediction"]
                        .value_counts()
                        .reindex(["Normal", "Attack"], fill_value=0)
                    )
                    pie_df = prediction_counts.rename_axis(
                        "Prediction"
                    ).reset_index(name="Records")

                    st.plotly_chart(
                        px.pie(
                            pie_df,
                            names="Prediction",
                            values="Records",
                            title="Normal vs Attack",
                        ),
                        use_container_width=True,
                    )

                with chart_col2:
                    st.subheader("Severity Distribution")
                    severity_order = [
                        "Low", "Medium", "High", "Critical"
                    ]
                    severity_counts = (
                        results["Severity"]
                        .value_counts()
                        .reindex(severity_order, fill_value=0)
                    )
                    severity_df = severity_counts.rename_axis(
                        "Severity"
                    ).reset_index(name="Records")

                    st.plotly_chart(
                        px.bar(
                            severity_df,
                            x="Severity",
                            y="Records",
                            title="Records by Severity",
                        ),
                        use_container_width=True,
                    )

                st.subheader("Filter Analyzed Records")
                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    prediction_filter = st.multiselect(
                        "Prediction",
                        options=["Normal", "Attack"],
                        default=["Normal", "Attack"],
                    )

                with filter_col2:
                    severity_filter = st.multiselect(
                        "Severity",
                        options=["Low", "Medium", "High", "Critical"],
                        default=["Low", "Medium", "High", "Critical"],
                    )

                filtered_results = results[
                    results["Prediction"].isin(prediction_filter)
                    & results["Severity"].isin(severity_filter)
                ]

                st.caption(
                    f"Showing {len(filtered_results):,} of {total:,} records."
                )
                st.dataframe(filtered_results, use_container_width=True)

                csv_data = filtered_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Filtered Predictions CSV",
                    data=csv_data,
                    file_name="intrusion_predictions.csv",
                    mime="text/csv",
                )

                if st.button("Save detected attacks to alert history"):
                    saved_count = save_alerts(results)
                    if saved_count:
                        st.success(
                            f"Saved {saved_count:,} attack alerts to the database."
                        )
                    else:
                        st.info("No predicted attacks to save.")

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
        st.info(
            "No alerts have been saved yet. Analyze a CSV and save its attacks."
        )
    else:
        st.metric("Stored Alerts", f"{len(history):,}")

        severity_counts = (
            history["severity"]
            .value_counts()
            .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        )
        st.plotly_chart(
            px.bar(
                severity_counts.rename_axis("Severity").reset_index(
                    name="Alerts"
                ),
                x="Severity",
                y="Alerts",
                title="Saved Alerts by Severity",
            ),
            use_container_width=True,
        )

        history_filter = st.multiselect(
            "Filter saved alerts by severity",
            options=["Low", "Medium", "High", "Critical"],
            default=["Low", "Medium", "High", "Critical"],
            key="history_severity_filter",
        )

        filtered_history = history[
            history["severity"].isin(history_filter)
        ]

        st.dataframe(filtered_history, use_container_width=True)

        st.download_button(
            "Download Alert History CSV",
            data=filtered_history.to_csv(index=False).encode("utf-8"),
            file_name="alert_history.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Academic prototype using benchmark/CSV data. Predictions are not "
    "proof of an attack; validate findings before taking security action."
)
