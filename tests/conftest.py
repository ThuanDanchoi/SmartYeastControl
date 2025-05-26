# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def recommender_df():
    """
    Toy dataset for train_recommender.py:
    - 2 feature columns: feat1, feat2
    - 1 target SP column: targetSP
    - A Quality column to be filtered on == 'good'
    """
    return pd.DataFrame({
        'feat1':   [1.0, 2.0, 3.0, 4.0],
        'feat2':   [0.1, 0.2, 0.3, 0.4],
        'targetSP':[10.0, 20.0, 30.0, 40.0],
        'Quality': ['good','good','good','good'],
    })

@pytest.fixture
def input_df():
    """
    Toy dataset for predict_recommender.py:
    - Same features + original SP column
    """
    return pd.DataFrame({
        'feat1':   [1,2],
        'feat2':   [3,4],
        'targetSP':[100,200],
    })
