# tests/test_train_recommender.py
import importlib.util
import joblib
import pandas as pd
from pathlib import Path

def test_train_recommender_saves_artifacts(tmp_path, monkeypatch, recommender_df):
    # 1) Patch pandas.read_csv → our fixture
    monkeypatch.setattr(pd, "read_csv", lambda path: recommender_df)

    # 2) Capture joblib.dump calls
    saved = []
    def fake_dump(obj, path):
        saved.append(Path(path).name)
    monkeypatch.setattr(joblib, "dump", fake_dump)

    # 3) Execute the train_recommender.py script
    script = Path(__file__).parent.parent / "src" / "modeling" / "train_recommender.py"
    spec   = importlib.util.spec_from_file_location("train_rec", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 4) Verify that our three artifacts were dumped
    assert "scaler.joblib" in saved,          "Expected to save scaler.joblib"
    assert "recommender_model.joblib" in saved, "Expected to save recommender_model.joblib"
    assert "sp_cols.joblib" in saved,         "Expected to save sp_cols.joblib"
