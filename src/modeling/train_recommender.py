import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed data
df = pd.read_csv('data/processed/recommender_features.csv')
if 'Quality' in df.columns:
    df["Quality"] = df["Quality"].astype(str).str.strip().str.lower()
    df = df[df['Quality'] == 'good'].reset_index(drop=True)

# Identify SP columns (target) and feature columns
sp_cols = [col for col in df.columns if col.endswith('SP')]
drop_cols = sp_cols + ['Quality', 'Set Time', 'VYP batch', 'Part']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[sp_cols]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# evaluation
preds = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²:   {r2:.4f}")

# Save scaler and model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
models_dir = PROJECT_ROOT / 'models' / 'recommender'
joblib.dump(scaler, models_dir / 'scaler.joblib')
joblib.dump(model, models_dir / 'recommender_model.joblib')
joblib.dump(sp_cols, models_dir / 'sp_cols.joblib')
print(f"\nSaved scaler to {models_dir / 'scaler.joblib'}")
print(f"Saved model  to {models_dir / 'recommender_model.joblib'}")
