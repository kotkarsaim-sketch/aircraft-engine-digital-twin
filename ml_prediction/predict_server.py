"""
Prediction server for the ML digital twin playground.
Serves a Flask REST API that classifies faults and predicts RUL from a
7-sensor telemetry vector. The UI (ui/index.html) calls this with manual
slider values and gets genuine model output back.

Endpoints:
  POST /predict   {sensors: {7 values}} -> {fault, confidence, rul_h}
  GET  /health    server + model status
  GET  /          serves the static UI (ui/)
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

from flask import Flask, request, jsonify, send_from_directory
import joblib

app = Flask(__name__, static_folder=str(HERE / "ui"), static_url_path="/")

MODELS = HERE / "models"
clf = None
scl = None
le = None
rul_model = None
rul_scaler = None
meta = {}

# Rolling window buffer for RUL regression features
RUL_WINDOW = []


def load_models():
    global clf, scl, le, rul_model, rul_scaler, meta
    clf = joblib.load(MODELS / "classifier.joblib")
    scl = joblib.load(MODELS / "scaler.joblib")
    le = joblib.load(MODELS / "label_encoder.joblib")
    rul_model = joblib.load(MODELS / "rul_model.joblib")
    rul_scaler = joblib.load(MODELS / "rul_scaler.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    print(f"Models loaded: {meta['fault_types']}")


def classify(sensors_vec):
    x = scl.transform([sensors_vec])
    proba = clf.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return le.classes_[idx], float(proba[idx]), {
        n: float(p) for n, p in zip(le.classes_, proba)
    }


def predict_rul(sensors_vec, t, window_len):
    global RUL_WINDOW
    RUL_WINDOW.append(sensors_vec)
    if len(RUL_WINDOW) > window_len:
        RUL_WINDOW.pop(0)
    # Rolling mean of the last window (matches training feature transform)
    xr = np.asarray(RUL_WINDOW, dtype=np.float32).mean(axis=0)
    feat = np.hstack([xr, [t]]).reshape(1, -1)
    feat_s = rul_scaler.transform(feat)
    n = float(rul_model.predict(feat_s)[0])
    return float(np.clip(n * meta["rul_max"], 0.0, meta["rul_max"]))


@app.route("/")
def index():
    return send_from_directory(HERE / "ui", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": clf is not None,
        "fault_types": meta.get("fault_types", []),
        "window": meta.get("window"),
        "backend": "sklearn",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "models not loaded"}), 500
    data = request.get_json()
    sensors = data.get("sensors")
    t = float(data.get("t", 0.0))
    if not sensors:
        return jsonify({"error": "missing sensors"}), 400

    keys = meta["sensor_keys"]
    vec = [float(sensors.get(k, 0)) for k in keys]
    vec = np.asarray(vec, dtype=np.float32)

    fault, conf, proba = classify(vec)
    rul = predict_rul(vec, t, meta["window"])

    return jsonify({
        "fault": fault,
        "confidence": conf,
        "probabilities": proba,
        "rul_h": rul,
        "window_filled": len(RUL_WINDOW) >= meta.get("window", 15),
    })


@app.route("/reset", methods=["POST"])
def reset_history():
    global RUL_WINDOW
    RUL_WINDOW.clear()
    return jsonify({"status": "ok", "cleared": True})


if __name__ == "__main__":
    load_models()
    print("Serving ML prediction playground at http://localhost:5000")
    print("  POST /predict with {sensors: {...}, t: seconds}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
