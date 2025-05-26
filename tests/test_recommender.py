import joblib
import numpy as np
import pytest
import os
import pandas as pd

# Load model and scaler
base_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(base_path, "..", "models", "recommender")

model = joblib.load(os.path.join(models_path, "recommender_model.joblib"))
scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))
sp_cols = joblib.load(os.path.join(models_path, "sp_cols.joblib"))

print("Recommender model, scaler, and SP columns loaded successfully")

# Features model expects
EXPECTED_FEATURES = list(scaler.feature_names_in_)

# ─── Test Cases ─────────────────────────────────────────────────────────

def test_normal_case():
    sample_input = pd.DataFrame([np.random.rand(len(EXPECTED_FEATURES))], columns=EXPECTED_FEATURES)
    scaled = scaler.transform(sample_input)
    preds = model.predict(scaled)

    assert preds.shape[1] == len(sp_cols)
    print("Normal Case Passed")

def test_large_batch():
    batch = pd.DataFrame(np.random.rand(100, len(EXPECTED_FEATURES)), columns=EXPECTED_FEATURES)
    scaled = scaler.transform(batch)
    preds = model.predict(scaled)

    assert preds.shape == (100, len(sp_cols))
    print("Large Batch Case Passed")

def test_missing_features():
    wrong_input = pd.DataFrame(np.random.rand(1, 3))  # Too few columns
    with pytest.raises(ValueError):
        scaler.transform(wrong_input)
    print("Missing Feature Case Passed")

def test_invalid_values():
    bad_input = pd.DataFrame([["bad"] * len(EXPECTED_FEATURES)], columns=EXPECTED_FEATURES)
    with pytest.raises(ValueError):
        scaler.transform(bad_input)
    print("Invalid Value Case Passed")

# ─── Run All Tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_normal_case()
    test_large_batch()
    test_missing_features()
    test_invalid_values()
    print("\nAll recommender tests executed.")
