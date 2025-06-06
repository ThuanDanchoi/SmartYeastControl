import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Load preprocessed data
data_path = PROJECT_ROOT / 'data' / 'processed' / 'recommender_features.csv'
df = pd.read_csv(data_path)
if 'Quality' in df.columns:
    df["Quality"] = df["Quality"].astype(str).str.strip().str.lower()
    df = df[df['Quality'] == 'good'].reset_index(drop=True)

# Identify SP columns (target) and feature columns
sp_cols = [col for col in df.columns if col.endswith('SP')]
drop_cols = sp_cols + ['Quality', 'Set Time', 'Timestamp', 'VYP batch', 'Part']
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

# Hyperparameter grid for RF Regressor 
param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.5]
}

# Grid Search
grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
print("Running RF Regressor Grid Search…")
grid.fit(X_train_scaled, y_train)

# Extracting best model + CV Results
best_rf = grid.best_estimator_
best_params = grid.best_params_
best_cv_rmse = (-grid.best_score_)**0.5
# Test Set Evaluation
preds = best_rf.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, preds) # can't do squared=false
rmse_test = mse_test**0.5 # manually doing sqrt
test_r2   = r2_score(y_test, preds)

print(f"Test RMSE: {rmse_test:.4f}")
print(f"Test R²:   {test_r2:.4f}")

# Save scaler and model

models_dir = PROJECT_ROOT / 'models' / 'recommender'
joblib.dump(scaler, models_dir / 'scaler.joblib')
joblib.dump(best_rf, models_dir / 'recommender_model.joblib')
joblib.dump(sp_cols, models_dir / 'sp_cols.joblib')
print(f"\nSaved scaler to {models_dir / 'scaler.joblib'}")
print(f"Saved model  to {models_dir / 'recommender_model.joblib'}")
