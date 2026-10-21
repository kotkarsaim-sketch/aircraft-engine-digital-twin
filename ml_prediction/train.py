"""
Train ML models from the collected simulation data:
  1. Fault classifier  -> sklearn RandomForest (7-sensor -> fault label)
  2. RUL regressor     -> sklearn GradientBoosting over a telemetry window
Outputs saved models + a metadata json for the prediction server.

No TensorFlow dependency — runs on sklearn only (works under the current
OS Application Control policy which blocks TF's optree native DLL).
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

HERE = Path(__file__).parent
SENSOR_KEYS = ["rpm", "egt", "cht", "oil_temp", "oil_press",
               "fuel_flow", "manifold"]
WINDOW = 15          # rolling-window width for RUL features
RUL_MAX = 1.0        # hours scale (simulated horizon ~0.5h)
RUL_STEP = 0.1       # simulator tick RUL decrement (hours per tick)


def load_data(path: Path):
    return pd.read_csv(path)


def train_classifier(df):
    X = df[SENSOR_KEYS].values.astype(np.float32)
    y = df["fault"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scl = StandardScaler()
    X_s = scl.fit_transform(X)

    print(f"  Classifier: {X.shape[0]} samples, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    clf = RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42)
    t0 = time.time()
    clf.fit(X_s, y_enc)
    print(f"  Classifier trained in {time.time()-t0:.1f}s")
    return clf, scl, le


def rolling_features(df):
    """Build window-based feature vectors for RUL regression."""
    X = df[SENSOR_KEYS].values.astype(np.float32)
    rul = df["rul_h"].values.astype(np.float32)
    t = df["t"].values.astype(np.float32)

    # Rolling mean/std over WINDOW for each sensor (padded at start)
    Xr = np.zeros_like(X)
    for i in range(len(X)):
        lo = max(0, i - WINDOW + 1)
        win = X[lo:i + 1]
        Xr[i] = win.mean(axis=0)
    # Add a time feature so the RUL regressor sees the age signal
    Xfeat = np.hstack([Xr, t.reshape(-1, 1)])
    return Xfeat, rul


def train_rul(df):
    Xr, yr = rolling_features(df)
    yr = yr / RUL_MAX   # normalize to [0,1]
    print(f"  RUL: {Xr.shape[0]} samples, {Xr.shape[1]} features")

    split = int(0.85 * len(Xr))
    Xtr, Xva = Xr[:split], Xr[split:]
    ytr, yva = yr[:split], yr[split:]

    # Scale features for the tree ensemble
    fs = StandardScaler()
    Xtr_s = fs.fit_transform(Xtr)
    Xva_s = fs.transform(Xva)

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=4, subsample=0.9,
        random_state=42)
    t0 = time.time()
    model.fit(Xtr_s, ytr)
    print(f"  RUL trained in {time.time()-t0:.1f}s "
          f"({len(model.estimators_)} trees)")

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    pred = model.predict(Xva_s)
    yva_real = yva * RUL_MAX
    pred_real = pred * RUL_MAX
    print(f"  RUL val RMSE={mean_squared_error(yva_real, pred_real)**0.5:.2f}h "
          f"MAE={mean_absolute_error(yva_real, pred_real):.2f}h (target scale)")
    return model, fs


def main():
    data_file = HERE / "training_data.csv"
    if not data_file.exists():
        print(f"Missing {data_file}. Run collect_data.py first.")
        return
    print(f"Loading {data_file} ...")
    df = load_data(data_file)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    print("Training fault classifier (RandomForest)...")
    clf, scl, le = train_classifier(df)

    print("Training RUL regressor (GradientBoosting)...")
    rul_model, rul_scaler = train_rul(df)

    out = HERE / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(clf, out / "classifier.joblib")
    joblib.dump(scl, out / "scaler.joblib")
    joblib.dump(le, out / "label_encoder.joblib")
    joblib.dump(rul_model, out / "rul_model.joblib")
    joblib.dump(rul_scaler, out / "rul_scaler.joblib")
    meta = {
        "sensor_keys": SENSOR_KEYS,
        "window": WINDOW,
        "rul_max": RUL_MAX,
        "rul_step": RUL_STEP,
        "fault_types": le.classes_.tolist(),
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": int(len(df)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nArtifacts saved to {out}")
    print("  classifier.joblib, scaler.joblib, label_encoder.joblib,")
    print("  rul_model.joblib, rul_scaler.joblib, meta.json")


if __name__ == "__main__":
    main()
