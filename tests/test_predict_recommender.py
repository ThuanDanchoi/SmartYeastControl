# tests/test_predict_recommender.py
import sys
import importlib.util
import joblib
from pathlib import Path
import csv

def test_predict_recommender_writes_csv(tmp_path, monkeypatch, input_df):
    # 1) Patch pd.read_csv → input_df
    import pandas as pd
    monkeypatch.setattr(pd, "read_csv", lambda path: input_df)

    # 2) Dummy scaler & model & sp_cols
    class DummyScaler:
        def transform(self, X): return X  # identity
    class DummyModel:
        def predict(self, X): 
            # for each row, return value+1 in a nested list
            return [[val+1] for val in X['feat1']]
    def fake_load(path):
        name = Path(path).name
        if name == "scaler.joblib":
            return DummyScaler()
        if name == "recommender_model.joblib":
            return DummyModel()
        if name == "sp_cols.joblib":
            return ['targetSP']
        raise FileNotFoundError(path)
    monkeypatch.setattr(joblib, "load", fake_load)

    # 3) Arrange CLI args for --input / --output
    output_file = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", [
        "predict_recommender.py",
        "--input", "irrelevant.csv",
        "--output", str(output_file),
    ])

    # 4) Exec the predict_recommender.py script
    script = Path(__file__).parent.parent / "src" / "inference" / "predict_recommender.py"
    spec   = importlib.util.spec_from_file_location("predict_rec", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 5) Check the CSV was written with expected headers & values
    assert output_file.exists(), "Output CSV not created"
    with open(output_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Expect columns: feat1,feat2,targetSP
    assert reader.fieldnames == ['feat1','feat2','targetSP']
    # For each input row, your DummyModel adds +1 to feat1
    assert int(rows[0]['targetSP']) == input_df['feat1'][0] + 1
    assert int(rows[1]['targetSP']) == input_df['feat1'][1] + 1
