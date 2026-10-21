"""
Production-ready Flask API for C172P fault classification and RUL prediction.
Supports batch inference, model versioning, and health monitoring.
"""
import sys
sys.path.insert(0, r"C:\pylibs")

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import joblib

# Paths
HERE = Path(__file__).parent
ROOT = HERE.parent.parent
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"

app = Flask(__name__, static_folder=str(ROOT / "ui"), static_url_path="/")

# Model artifacts
clf_model = None
rul_model = None
scaler = None
label_encoder = None
rul_scaler = None
meta = {}

import threading

# Rolling window for RUL
RUL_WINDOW = []
MAX_WINDOW = 15
RUL_LOCK = threading.Lock()


def load_models():
    """Load all model artifacts."""
    global clf_model, rul_model, scaler, label_encoder, rul_scaler, meta

    # Load metadata
    with open(MODELS_DIR / "meta.json") as f:
        meta = json.load(f)

    # Load artifacts
    clf_model = joblib.load(MODELS_DIR / "best_classifier.joblib")
    rul_model = joblib.load(MODELS_DIR / "best_rul.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    # Use simple scaler for RUL (7 sensors + t = 8 features)
    rul_scaler = joblib.load(MODELS_DIR / "rul_simple_scaler.joblib")

    print(f"Models loaded: {meta.get('best_classifier', 'unknown')} + {meta.get('best_rul', 'unknown')}")
    print(f"Fault types: {label_encoder.classes_.tolist()}")


def classify(sensors_vec: np.ndarray) -> Tuple[str, float, Dict]:
    """Classify fault from sensor vector."""
    x = scaler.transform([sensors_vec])
    proba = clf_model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    fault = label_encoder.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    prob_dict = {n: float(p) for n, p in zip(label_encoder.classes_, proba)}
    return fault, confidence, prob_dict


def predict_rul(sensors_vec: np.ndarray, t: float, window_len: int = MAX_WINDOW) -> Optional[float]:
    """Predict RUL using rolling window."""
    global RUL_WINDOW
    with RUL_LOCK:
        RUL_WINDOW.append(sensors_vec)
        if len(RUL_WINDOW) > window_len:
            RUL_WINDOW.pop(0)

        if len(RUL_WINDOW) < window_len:
            return None

        xr = np.asarray(RUL_WINDOW[-window_len:], dtype=np.float32).mean(axis=0)
        feat = np.hstack([xr, [t]]).reshape(1, -1)
        feat_s = rul_scaler.transform(feat)
        n = float(rul_model.predict(feat_s)[0])
        return float(np.clip(n, 0.0, meta.get("rul_max", 1.0)))


@app.route("/")
def index():
    return send_from_directory(ROOT / "ui", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": clf_model is not None,
        "fault_types": label_encoder.classes_.tolist() if label_encoder else [],
        "window_size": MAX_WINDOW,
        "best_classifier": meta.get("best_classifier"),
        "best_rul": meta.get("best_rul"),
        "uptime_s": time.time() - START_TIME,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if clf_model is None:
        return jsonify({"error": "models not loaded"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "invalid json"}), 400

        sensors = data.get("sensors")
        t = float(data.get("t", 0.0))

        if not sensors:
            return jsonify({"error": "missing sensors"}), 400

        keys = meta["sensor_keys"]
        vec = np.array([float(sensors.get(k, 0)) for k in keys], dtype=np.float32)

        fault, conf, proba = classify(vec)
        rul = predict_rul(vec, t, meta.get("window", MAX_WINDOW))

        return jsonify({
            "fault": fault,
            "confidence": conf,
            "probabilities": proba,
            "rul_h": rul,
            "window_filled": len(RUL_WINDOW) >= MAX_WINDOW,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "internal server error", "detail": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Batch prediction for multiple sensor readings."""
    if clf_model is None:
        return jsonify({"error": "models not loaded"}), 503

    data = request.get_json()
    readings = data.get("readings", [])

    if not readings:
        return jsonify({"error": "missing readings"}), 400

    results = []
    for reading in readings:
        sensors = reading.get("sensors")
        t = float(reading.get("t", 0.0))
        if not sensors:
            results.append({"error": "missing sensors"})
            continue

        keys = meta["sensor_keys"]
        vec = np.array([float(sensors.get(k, 0)) for k in keys], dtype=np.float32)
        fault, conf, proba = classify(vec)
        rul = predict_rul(vec, t, meta.get("window", MAX_WINDOW))

        results.append({
            "fault": fault,
            "confidence": conf,
            "probabilities": proba,
            "rul_h": rul,
        })

    return jsonify({"predictions": results})


@app.route("/reset", methods=["POST"])
def reset_history():
    global RUL_WINDOW
    with RUL_LOCK:
        RUL_WINDOW.clear()
    return jsonify({"status": "ok", "cleared": True, "window_size": 0})


@app.route("/models/info")
def models_info():
    return jsonify({
        "classifier": {
            "type": meta.get("best_classifier"),
            "fault_types": label_encoder.classes_.tolist(),
        },
        "rul": {
            "type": meta.get("best_rul"),
            "window_size": meta.get("window"),
            "max_rul_h": meta.get("rul_max"),
        },
        "sensor_keys": meta.get("sensor_keys"),
        "feature_config": meta.get("feature_config", {}),
    })


START_TIME = time.time()

if __name__ == "__main__":
    load_models()
    print(f"Serving C172P ML Prediction API at http://localhost:5000")
    print(f"  POST /predict        - single prediction")
    print(f"  POST /predict_batch  - batch prediction")
    print(f"  POST /reset          - clear RUL window")
    print(f"  GET  /health         - health check")
    print(f"  GET  /models/info    - model metadata")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)