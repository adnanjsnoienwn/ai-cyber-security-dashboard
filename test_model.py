import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/intrusion_model.joblib")
TEST_DATA_PATH = Path("data/UNSW_NB15_testing-set.csv")


def load_model_and_sample():
    assert MODEL_PATH.exists(), "Trained model file was not found."
    assert TEST_DATA_PATH.exists(), "UNSW-NB15 test dataset was not found."

    model = joblib.load(MODEL_PATH)
    dataset = pd.read_csv(TEST_DATA_PATH, nrows=5)

    X = dataset.drop(
        columns=["id", "attack_cat", "label"],
        errors="ignore",
    )

    return model, X


def test_model_returns_binary_predictions():
    model, X = load_model_and_sample()

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_attack_probabilities_are_between_zero_and_one():
    model, X = load_model_and_sample()

    probabilities = model.predict_proba(X)[:, 1]

    assert len(probabilities) == len(X)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_has_expected_pipeline_steps():
    model, _ = load_model_and_sample()

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps
