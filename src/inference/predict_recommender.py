import pandas as pd
import joblib
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description='Load a preprocessed feature CSV and output SP recommendation.'
)
parser.add_argument(
    '--input', type=str,
    default='data/processed/recommender_features.csv',
    help='Path to feature CSV (X + original y columns)'
)

parser.add_argument(
    '--output', type=str,
    default='data/predictions/recommender_predictions.csv',
    help='Path to save predictions CSV'
)

args = parser.parse_args()

# Load features
df = pd.read_csv(args.input)
# Identify target Sp columns by suffix
sp_cols = [col for col in df.columns if col.endswith('SP')]
# Build feature matrix X (drop SP columns)
X = df.drop(columns=sp_cols)
# Load scaler and model
root = Path(__file__).resolve().parents[2]
scaler_path = root / 'models' / 'recommender' / 'scaler.joblib'
model_path = root / 'models' / 'recommender' / 'recommender_model.joblib'
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Scale and predict
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

# Build output DataFrame
preds_df = pd.DataFrame(preds, columns=sp_cols, index=df.index)
result = pd.concat([X, preds_df], axis=1)

# Save the results
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out_path, index=False)
print(f'Saved predictions to {out_path}')
