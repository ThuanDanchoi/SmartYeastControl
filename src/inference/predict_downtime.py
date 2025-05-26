import joblib
from pathlib import Path

def predict_downtime(df):
    # load model training
    model_path = Path("models/downtime/downtime_model.joblib")
    model = joblib.load(model_path)

    # drop the columns
    drop_cols = [col for col in ["Timestamp", "label"] if col in df.columns]
    X = df.drop(columns=drop_cols)

    # predict the downtime
    y_pred = model.predict(X)

    # result into dataframe
    df["downtime_prediction"] = y_pred

    return df
