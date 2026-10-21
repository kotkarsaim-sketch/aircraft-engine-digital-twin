import sys, time, joblib
from pathlib import Path
sys.path.insert(0, r"C:\pylibs")
sys.path.insert(0, r"C:\twin\ml_prediction\src")
from data.dataset import C172PDataset, load_config
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import numpy as np

cfg = load_config("config.yaml")
ds = C172PDataset(cfg)
raw = ds.generate_raw_data()
raw = ds.compute_rul(raw)
raw = ds.engineer_features(raw)
train, val, test = ds.split_by_seed(raw)
ds.fit_scalers(train)

# Full train+val
import pandas as pd
full_train = pd.concat([train, val])
X_train_clf, y_train_clf = ds.prepare_classifier_data(full_train)
X_train_clf_s = ds.scaler.transform(X_train_clf)
y_train_clf_enc = ds.label_encoder.transform(y_train_clf)

window = cfg["features"]["default_rul_window"]
X_train_rul, y_train_rul = ds.create_rolling_features(full_train, window)
X_train_rul_s = ds.rul_scaler.transform(X_train_rul)

print("Training final models...")
clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train_clf_s, y_train_clf_enc)

rul = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
rul.fit(X_train_rul_s, y_train_rul)

# Save
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
joblib.dump(clf, models_dir / "best_classifier.joblib")
joblib.dump(rul, models_dir / "best_rul.joblib")
ds.save_artifacts(Path("models/artifacts"))

meta = {
    "experiment": "c172p-fault-rul-v1",
    "best_classifier": "random_forest",
    "best_rul": "gradient_boosting",
    "sensor_keys": cfg["features"]["sensor_keys"],
    "window": window,
    "rul_max": 1.0,
    "fault_types": ds.label_encoder.classes_.tolist(),
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
import json
with open(models_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Models saved!")