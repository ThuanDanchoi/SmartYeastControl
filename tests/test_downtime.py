import joblib
import numpy as np
import pytest
import os
import pandas as pd

# Load only the model (downtime detection doesn't use encoder or scaler)
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "..", "models", "downtime", "downtime_model.joblib")

downtime_model = joblib.load(model_path)
print("Downtime model loaded successfully")

EXPECTED_FEATURES = 10

#  Basic Model Tests

def test_normal_case():
    sample_input = np.random.rand(1, EXPECTED_FEATURES)
    result = downtime_model.predict(sample_input)

    assert result is not None, "Model should return a prediction"
    assert len(result) == 1, "Should return a single prediction"
    print(f"Normal Case Passed: {result}")

def test_edge_case():
    extreme_input = np.random.uniform(-10000, 10000, (1, EXPECTED_FEATURES))
    result = downtime_model.predict(extreme_input)

    assert result is not None, "Model should handle extreme values"
    print(f"Edge Case Passed: {result}")

def test_invalid_data():
    with pytest.raises(ValueError):
        downtime_model.predict([["not", "a", "number"]])
    print("Invalid Data Case Passed")

def test_empty_input():
    with pytest.raises(ValueError):
        downtime_model.predict(np.array([[]]))
    print("Empty Input Case Passed")

def test_large_dataset():
    large_input = np.random.rand(1000, EXPECTED_FEATURES)
    result = downtime_model.predict(large_input)

    assert len(result) == 1000, "Model should handle large inputs"
    print(f"Large Dataset Case Passed: {len(result)} samples")

# Full Pipeline Test

def test_full_pipeline():
    feature_names = ['Dept', 'Line', 'Sub Line', 'Shift', 'Waterfall', 'From Product',
                     'Cause Category', 'Cause', 'Total Time Mins', 'Freq']

    sample_df = pd.DataFrame(np.random.rand(1, len(feature_names)), columns=feature_names)

    result = downtime_model.predict(sample_df)

    assert result is not None
    assert len(result) == 1
    print(f"Full Pipeline Case Passed: {result}")

# Edge Case Tests

def test_missing_values():
    incomplete_input = np.array([[np.nan] * EXPECTED_FEATURES])
    try:
        result = downtime_model.predict(incomplete_input)
        assert result is not None
        print(f"Missing Values Case Passed: {result}")
    except Exception as e:
        pytest.fail(f"Model raised unexpected error on NaNs: {e}")

def test_feature_mismatch():
    mismatch_input = np.random.rand(1, 5)  # Less than expected features
    with pytest.raises(ValueError):
        downtime_model.predict(mismatch_input)
    print("Feature Mismatch Case Passed")

# Run All Tests
if __name__ == "__main__":
    test_normal_case()
    test_edge_case()
    test_invalid_data()
    test_empty_input()
    test_large_dataset()
    test_full_pipeline()
    test_missing_values()
    test_feature_mismatch()
    print("\nAll tests executed.")
