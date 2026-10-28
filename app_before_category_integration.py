import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import shap
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
    page_icon="C",
    layout="wide",
)

MODEL_PATH = "models/intrusion_model.joblib"
DB_PATH = "alerts.db"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

st.title("AI-Based Cyber Security Dashboard")
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
    ["Overview", "Analyze Traffic", "Alert History"]
)


with tab1:
    st.subheader("Model Evaluation")

    pdf_report_path = "reports/model_evaluation_report.pdf"

    try:
        with open(pdf_report_path, "rb") as pdf_file:
            st.download_button(
                label="Download Model Evaluation Report (PDF)",
                data=pdf_file,
                file_name="model_evaluation_report.pdf",
                mime="application/pdf",
            )
    except FileNotFoundError:
        st.warning(
            "PDF report not found. Run create_pdf_report.py first."
        )

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

                # Validate uploaded feature values before prediction
                validation_errors = []

                missing_counts = X_input.isna().sum()
                missing_counts = missing_counts[missing_counts > 0]

                if not missing_counts.empty:
                    details = ", ".join(
                        f"{column}: {int(count)}"
                        for column, count in missing_counts.items()
                    )
                    validation_errors.append(
                        f"Missing values detected ({details})."
                    )

                preprocessor = model.named_steps["preprocessor"]

                categorical_columns = list(
                    preprocessor.transformers_[0][2]
                )
                numerical_columns = list(
                    preprocessor.transformers_[1][2]
                )

                for column in numerical_columns:
                    converted = pd.to_numeric(
                        X_input[column],
                        errors="coerce",
                    )

                    invalid_mask = (
                        X_input[column].notna() & converted.isna()
                    )

                    if invalid_mask.any():
                        validation_errors.append(
                            f"Column '{column}' contains "
                            f"{int(invalid_mask.sum())} non-numeric value(s)."
                        )

                    if converted.isin(
                        [float("inf"), float("-inf")]
                    ).any():
                        validation_errors.append(
                            f"Column '{column}' contains infinite values."
                        )

                    X_input[column] = converted

                if validation_errors:
                    raise ValueError(
                        "Input validation failed: "
                        + " ".join(validation_errors)
                    )

                for column in categorical_columns:
                    X_input[column] = X_input[column].astype(str)

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

                st.progress(
                    min(max(attack_rate / 100, 0.0), 1.0),
                    text=f"Attack rate: {attack_rate:.2f}%",
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

                # Show dataset attack categories when available
                if "attack_cat" in results.columns:
                    st.subheader("Attack Category Distribution")

                    category_data = results.copy()
                    category_data["attack_cat"] = (
                        category_data["attack_cat"]
                        .fillna("Unknown")
                        .astype(str)
                    )

                    category_data = category_data[
                        category_data["attack_cat"].str.strip().ne("")
                    ]

                    category_counts = (
                        category_data["attack_cat"]
                        .value_counts()
                        .reset_index()
                    )
                    category_counts.columns = ["Attack Category", "Records"]

                    st.plotly_chart(
                        px.bar(
                            category_counts,
                            x="Attack Category",
                            y="Records",
                            title="Records by Dataset Attack Category",
                            text_auto=True,
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Attack-category analysis is unavailable because "
                        "the uploaded CSV has no 'attack_cat' column."
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

                st.subheader("Explain an Individual Prediction (SHAP)")
                st.caption(
                    "Shows which features most influenced the model's "
                    "Attack probability for one selected record."
                )

                if len(results) > 0:
                    explain_index = st.selectbox(
                        "Select record to explain",
                        options=list(range(len(results))),
                        format_func=lambda i: (
                            f"Record {i + 1} ? "
                            f"{results.iloc[i]['Prediction']} "
                            f"({results.iloc[i]['Attack Probability']:.1%} attack probability)"
                        ),
                        key="shap_record_index",
                    )

                    if st.button("Explain selected record", key="run_shap"):
                        with st.spinner("Calculating SHAP explanation..."):
                            try:
                                preprocessor = model.named_steps["preprocessor"]
                                classifier = model.named_steps["classifier"]

                                one_record = X_input.iloc[[explain_index]]
                                transformed = preprocessor.transform(one_record)

                                if hasattr(transformed, "toarray"):
                                    transformed = transformed.toarray()

                                transformed = np.asarray(
                                    transformed, dtype=np.float64
                                )

                                explainer = shap.TreeExplainer(classifier)
                                values = explainer.shap_values(transformed)

                                # For binary classification, select class 1 (Attack).
                                if isinstance(values, list):
                                    attack_values = np.asarray(values[1])[0]
                                else:
                                    values = np.asarray(values)
                                    if values.ndim == 3:
                                        attack_values = values[0, :, 1]
                                    elif values.ndim == 2:
                                        attack_values = values[0]
                                    else:
                                        raise ValueError(
                                            "Unexpected SHAP output shape."
                                        )

                                feature_names = preprocessor.get_feature_names_out()

                                explanation_df = pd.DataFrame({
                                    "Feature": feature_names,
                                    "SHAP impact": attack_values,
                                })
                                explanation_df["Absolute impact"] = (
                                    explanation_df["SHAP impact"].abs()
                                )
                                explanation_df = explanation_df.nlargest(
                                    5, "Absolute impact"
                                ).sort_values("SHAP impact")

                                explanation_df["Direction"] = explanation_df[
                                    "SHAP impact"
                                ].apply(
                                    lambda v: (
                                        "Pushes toward Attack"
                                        if v > 0
                                        else "Pushes away from Attack"
                                        if v < 0
                                        else "No directional impact"
                                    )
                                )

                                st.markdown(
                                    "**Top five feature contributions** "
                                    "for the selected record:"
                                )
                                st.caption(
                                    "Positive SHAP values push the model output "
                                    "toward Attack; negative values push it away. "
                                    "These are model explanations, not proof of "
                                    "malicious activity."
                                )
                                st.dataframe(
                                    explanation_df[
                                        ["Feature", "SHAP impact", "Direction"]
                                    ].sort_values(
                                        "SHAP impact",
                                        ascending=False
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                            except Exception as exc:
                                st.error(
                                    f"Could not calculate SHAP explanation: {exc}"
                                )


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
        # Parse timestamps for date filtering and trend analysis.
        history["timestamp"] = pd.to_datetime(
            history["timestamp"], errors="coerce"
        )
        history = history.dropna(subset=["timestamp"]).copy()
        history["Alert Date"] = history["timestamp"].dt.date

        if history.empty:
            st.warning("No valid alert timestamps are available.")
        else:
            st.metric("Total Stored Alert Records", f"{len(history):,}")

            critical_count = int((history["severity"] == "Critical").sum())
            high_count = int((history["severity"] == "High").sum())
            average_confidence = history["confidence"].mean()

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Critical Alerts", f"{critical_count:,}")
            metric_col2.metric("High Alerts", f"{high_count:,}")
            metric_col3.metric(
                "Average Attack Probability",
                f"{average_confidence:.1%}"
                if pd.notna(average_confidence)
                else "N/A",
            )

           
            st.subheader("Daily Alert Trend")
            daily_counts = (
                history.groupby("Alert Date")
                .size()
                .reset_index(name="Alert Records")
                .sort_values("Alert Date")
            )

            if len(daily_counts) < 2:
                st.info(
                    "Only one day of alert history is available. "
                    "Additional dates will make a trend visible."
                )

            st.plotly_chart(
                px.bar(
                    daily_counts,
                    x="Alert Date",
                    y="Alert Records",
                    text="Alert Records",
                    title="Daily Saved Alert Records",
                ),
                use_container_width=True,
            )
    
            st.caption(
                "Counts represent saved records, not unique attacks. "
                "Saving the same analyzed data more than once creates duplicates."
            )
            st.subheader("Filter Alert History")

            date_col1, date_col2 = st.columns(2)
            min_date = history["Alert Date"].min()
            max_date = history["Alert Date"].max()

            with date_col1:
                start_date = st.date_input(
                    "Start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="history_start_date",
                )

            with date_col2:
                end_date = st.date_input(
                    "End date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="history_end_date",
                )

            history_filter = st.multiselect(
                "Severity",
                options=["Low", "Medium", "High", "Critical"],
                default=["Low", "Medium", "High", "Critical"],
                key="history_severity_filter",
            )

            min_confidence = st.slider(
                "Minimum attack probability",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                format="%d%%",
                key="history_min_confidence",
            )

            if start_date > end_date:
                st.error("Start date must be on or before end date.")
                filtered_history = history.iloc[0:0].copy()
            else:
                filtered_history = history[
                    history["Alert Date"].between(start_date, end_date)
                    & history["severity"].isin(history_filter)
                    & (history["confidence"] >= min_confidence / 100)
                ].copy()

            st.caption(
                f"Showing {len(filtered_history):,} of {len(history):,} "
                "saved alert records for the selected filters."
            )

            display_history = filtered_history.drop(
                columns=["Alert Date"], errors="ignore"
            )
            st.dataframe(display_history, use_container_width=True)

            st.download_button(
                "Download Filtered Alert History CSV",
                data=display_history.to_csv(index=False).encode("utf-8"),
                file_name="alert_history_filtered.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Academic prototype using benchmark/CSV data. Predictions are not "
    "proof of an attack; validate findings before taking security action."
)

