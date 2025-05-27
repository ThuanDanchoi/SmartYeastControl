import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# load preprocess data
data_path = PROJECT_ROOT / 'data' / 'processed' / 'downtime_features.csv'
df = pd.read_csv(data_path)

X = df.drop(columns=["Timestamp", "label"])
y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Defining hyperparameter distributions
param_dist = {
    'n_estimators':      [100, 200, 300],
    'learning_rate':     [0.01, 0.05, 0.1],
    'max_depth':         [3, 5, 7, 9],
    'subsample':         [0.7, 1.0],
    'colsample_bytree':  [0.7, 1.0],
    'gamma':             [0, 1, 5]
}

# RandomisedSearchCV
search = RandomizedSearchCV(
    estimator=XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_dist,
    n_iter=30,                    # number of random combos
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=2
)
print("Running downtime XGBoost RandomizedSearchCV…")
search.fit(X_train, y_train)

# Grabbing best estimator + print metrics
best_xgb = search.best_estimator_
print("\nBest params:", search.best_params_)
print("Best CV weighted-F1:", search.best_score_)
# evaluation
y_pred = best_xgb.predict(X_test)
print("\nTest Set Evaluation")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-weighted:", f1_score(y_test, y_pred, average='weighted'))

# Saving the tuned model
models_dir = PROJECT_ROOT / 'models' / 'downtime'
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / 'downtime_model_tuned.joblib'
joblib.dump(best_xgb, model_path)
print(f"\nSaved tuned downtime model to {model_path}")
