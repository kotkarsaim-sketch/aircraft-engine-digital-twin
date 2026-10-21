import sys, time, joblib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\pylibs")
sys.path.insert(0, r"C:\twin\ml_prediction\src")
from data.dataset import C172PDataset, load_config
from sklearn.ensemble import GradientBoostingRegressor

cfg = load_config("config.yaml")
ds = C172PDataset(cfg)
raw = ds.generate_raw_data()
raw = ds.compute_rul(raw)
raw = ds.engineer_features(raw)
train, val, test = ds.split_by_seed(raw)
ds.fit_scalers(train)

full_train = pd.concat([train, val])
import pandas as pd

window = cfg["features"]["default_rul_window"]
X_train_rul, y_train_rul = ds.create_rolling_features(full_train, window)
X_train_rul_s = ds.rul_scaler.transform(X_train_rul)

# Create SIMPLE rolling features: just 7 sensor means + t
def create_simple_rolling_features(df, window):
    X_list, y_list = [], []
    sensor_keys = cfg["features"]["sensor_keys"]
    for seed, group in df.groupby("seed"):
        group = group.sort_values("t").reset_index(drop=True)
        X_seed = group[sensor_keys].values
        y_seed = group["rul_h"].values
        for i in range(window - 1, len(X_seed)):
            win_mean = X_seed[i - window + 1:i + 1].mean(axis=0)
            X_list.append(np.hstack([win_mean, group.iloc[i]["t"]]))
            y_list.append(y_seed[i])
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

X_train_simple, y_train_simple = create_simple_rolling_features(full_train, window)
X_val_simple, y_val_simple = create_simple_rolling_features(val, window)

# Fit new scaler on simple features
from sklearn.preprocessing import StandardScaler
simple_scaler = StandardScaler()
X_train_simple_s = simple_scaler.fit_transform(X_train_simple)
X_val_simple_s = simple_scaler.transform(X_val_simple)

print("Training SIMPLE RUL model (7 sensors + t)...")
rul = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
rul.fit(X_train_simple_s, y_train_simple)

# Evaluate
from sklearn.metrics import mean_absolute_error
y_val_pred = rul.predict(X_val_simple_s)
print(f"Val MAE: {mean_absolute_error(y_val_simple, y_val_pred):.4f}h")

# Save
models_dir = Path("models")
joblib.dump(rul, models_dir / "best_rul.joblib")
joblib.dump(simple_scaler, models_dir / "rul_simple_scaler.joblib")

# Update meta
import json
with open(models_dir / "meta.json") as f:
    meta = json.load(f)
meta["rul_simple"] = True
meta["rul_feature_count"] = 8  # 7 sensors + t
with open(models_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Simple RUL model saved!")