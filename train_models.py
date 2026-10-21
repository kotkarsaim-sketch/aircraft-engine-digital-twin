#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 AIRCRAFT ENGINE DIGITAL TWIN -- MODEL TRAINING
============================================================================
 Trains two models on the NASA C-MAPSS FD001 turbofan run-to-failure dataset
 (NASA Prognostics CoE data repository) and saves them for backend.py:

   models/autoencoder.h5  - dense autoencoder trained on HEALTHY early-life
                            samples. Anomaly score = reconstruction MSE.
   models/lstm_rul.h5     - LSTM regressor: sliding window of normalized
                            sensors -> Remaining Useful Life in cycles
                            (treated as hours by backend.py).
   models/scaler.json     - feature list, min/max normalization values,
                            window size, RUL cap and AE anomaly threshold.

 DATASET PLACEMENT
 -----------------
 data/ already contains FD001 (train/test/RUL) fetched from a public
 mirror (github.com/bishals098/NASA-CMAPSS-dataset); canonical source is
 the NASA Prognostics Data Repository (PCoE Data Sets). Sanity check:
 train_FD001.txt must have exactly 20,631 rows.

 If the file is absent you can still exercise the whole pipeline with
 synthetic data:

     python train_models.py --demo

 USAGE
 -----
     python train_models.py                 # train on real FD001
     python train_models.py --demo          # train on synthetic data
     python train_models.py --eval-only     # score saved models on the
                                            # official test_FD001 set
     python train_models.py --epochs-lstm 40 --window 30

 REQUIREMENTS: pip install numpy pandas tensorflow-cpu
============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    sys.exit("TensorFlow is required for training:  pip install tensorflow-cpu")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"

# C-MAPSS column layout: unit id, time cycle, 3 operational settings,
# 21 sensor channels.
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

N_FEATURES = 7            # keep in sync with backend.py SENSOR_DEFS count


# ===========================================================================
# Data loading
# ===========================================================================
def load_dataset(data_dir: Path, subset: str, demo: bool) -> pd.DataFrame:
    """Load train_FD00x.txt or synthesize a stand-in dataset (--demo)."""
    path = data_dir / f"train_{subset}.txt"
    if demo or not path.exists():
        if not demo:
            print(f"[WARN] {path} not found -> falling back to --demo "
                  "synthetic data.")
        return synth_fd001_like()
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df = df.dropna(axis=1, how="all")           # trailing-space artifacts
    print(f"Loaded {len(df)} rows, {df['unit'].nunique()} units from {path}")
    return df


MULTI_CONDITION = {"FD002", "FD004"}   # six operating regimes


def filter_dominant_condition(df: pd.DataFrame, subset: str):
    """
    FD002/FD004 mix six operating regimes. The digital twin's live input
    (a C172P at cruise) carries no operating-condition channels, so we
    train on the DOMINANT regime only -- the cruise-equivalent envelope --
    and record the cluster centre so test-set evaluation filters the same
    way. Returns (filtered_df, cluster_info_or_None).
    """
    if subset not in MULTI_CONDITION:
        return df, None
    from sklearn.cluster import KMeans
    ops = df[["op1", "op2", "op3"]].values
    km = KMeans(n_clusters=6, n_init=10, random_state=42).fit(ops)
    sizes = np.bincount(km.labels_, minlength=6)
    keep = int(np.argmax(sizes))
    kept = df[km.labels_ == keep].reset_index(drop=True)
    info = {"k": 6, "chosen": keep,
            "centers": km.cluster_centers_.tolist(),
            "sizes": sizes.tolist()}
    print(f"[CONDITION] {subset}: kept dominant regime {keep} "
          f"({int(sizes[keep])} of {len(df)} rows); regime sizes="
          f"{sizes.tolist()}")
    return kept, info


def nearest_center(ops: np.ndarray, centers: list) -> np.ndarray:
    """Index of the nearest recorded cluster centre for each row."""
    d = ((ops[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1)


def synth_fd001_like(n_units: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic run-to-failure data mimicking FD001's character: per-unit
    lifetime 150-230 cycles and smooth degradation trends with AR(1) noise
    across ~21 sensor channels.

    Degradation DIRECTIONS are drawn once per FEATURE (like real turbofan
    data, where e.g. EGT consistently rises toward failure), then scaled
    per unit -- this keeps the 'healthy manifold' coherent so the
    autoencoder learns a meaningful nominal envelope.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(400, 500, size=21)               # healthy baselines
    feat_slope = rng.uniform(-0.35, 0.35, size=21)      # fixed per feature
    feat_slope *= np.where(np.arange(21) % 3 == 0, 1.0, 0.35)
    frames = []
    for u in range(1, n_units + 1):
        n = int(rng.integers(150, 231))
        life = np.arange(1, n + 1)
        rul_norm = 1.0 - life / n                       # 1 -> 0 over life
        unit_gain = rng.uniform(0.7, 1.3)               # severity per unit
        rows = {"unit": u, "cycle": life}
        rows.update({"op1": 0.0, "op2": 0.0, "op3": 100.0})
        for i in range(21):
            drift = feat_slope[i] * unit_gain * rul_norm * base[i]
            ar = np.zeros(n)
            for t in range(1, n):                       # AR(1)-ish noise
                ar[t] = 0.85 * ar[t - 1] + rng.normal(0, 0.01 * base[i])
            rows[f"s{i + 1}"] = base[i] + drift + ar
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# Preprocessing
# ===========================================================================
def select_features(df: pd.DataFrame, k: int = N_FEATURES) -> list[str]:
    """
    Pick k informative channels: drop near-constant columns, keep the lowest
    numbered survivors (deterministic). Exactly k features are returned --
    backend.py feeds a matching 7-D vector from FlightGear sensors.
    """
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    variances = df[sensor_cols].var()
    varying = [c for c in sensor_cols if variances[c] > 1e-6]
    varying.sort(key=lambda c: int(c[1:]))
    feats = varying[:k]
    while len(feats) < k:                # extremely unlikely; keeps shapes OK
        feats.append(varying[-1])
    print(f"Selected features ({len(feats)}): {feats}")
    return feats


def add_labels(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """RUL label per row = max cycle of the unit - current cycle, clipped."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def minmax_fit(df: pd.DataFrame, feats: list[str]):
    mins = df[feats].min()
    maxs = df[feats].max()
    span = (maxs - mins).replace(0, 1.0)             # avoid divide-by-zero
    return mins.values.astype(np.float32), span.values.astype(np.float32)


def make_windows(unit_frames: dict[int, np.ndarray],
                 labels: dict[int, np.ndarray], w: int):
    """Sliding windows of w consecutive cycles; target = RUL at window end."""
    xs, ys = [], []
    for u, arr in unit_frames.items():
        y = labels[u]
        for end in range(w - 1, len(arr)):
            xs.append(arr[end - w + 1:end + 1])
            ys.append(y[end])
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y


# ===========================================================================
# NASA PHM'08-style asymmetric scoring (extra evaluation metric)
# ===========================================================================
def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true                      # late predictions punished more
    return float(np.sum(np.where(d < 0,
                                 np.exp(-d / 13.0) - 1,
                                 np.exp(d / 10.0) - 1)))


# ===========================================================================
# Model builders
# ===========================================================================
def build_autoencoder(n_feat: int) -> tf.keras.Model:
    """Small dense autoencoder: 7 -> 32 -> 16 -> latent 4 -> mirror -> 7."""
    model = models.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(4, activation="relu", name="latent"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_feat, activation="linear"),
    ], name="engine_autoencoder")
    model.compile(optimizer="adam", loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


def build_lstm(window: int, n_feat: int) -> tf.keras.Model:
    """Stacked LSTM regressor mapping (window x features) -> RUL."""
    model = models.Sequential([
        layers.Input(shape=(window, n_feat)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ], name="lstm_rul")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


# ===========================================================================
# Official test-set evaluation (test_FD001 + RUL_FD001)
# ===========================================================================
def evaluate_on_test_set(data_dir: Path, out_dir: Path, label: str) -> None:
    """
    Canonical C-MAPSS protocol: for each test engine, run the LSTM on the
    window ending at its LAST recorded cycle and compare the prediction
    against the ground-truth RUL file. Also checks that autoencoder
    reconstruction error correlates with the true degradation level (it
    should rise as true RUL falls). Subset/regime come from scaler.json.
    """
    meta = json.loads((out_dir / "scaler.json").read_text())
    subset = meta.get("subset", "FD001")
    cond = meta.get("op_condition")
    feats = meta["features"]
    mins = np.asarray(meta["min"], dtype=np.float32)
    span = np.asarray(meta["span"], dtype=np.float32)
    cap, w = float(meta["rul_cap"]), int(meta["window"])

    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"
    if not test_path.exists() or not rul_path.exists():
        print(f"[TEST-EVAL] skipped ({test_path.name} / {rul_path.name} "
              "not found)")
        return

    ae = tf.keras.models.load_model(out_dir / "autoencoder.h5", compile=False)
    lstm = tf.keras.models.load_model(out_dir / "lstm_rul.h5", compile=False)

    df = pd.read_csv(test_path, sep=r"\s+", header=None,
                     names=COLS).dropna(axis=1, how="all")
    if cond:
        ops = df[["op1", "op2", "op3"]].values
        before = df["unit"].nunique()
        df = df[nearest_center(ops, cond["centers"]) == cond["chosen"]]
        df = df.reset_index(drop=True)
        print(f"[TEST-EVAL] regime filter kept {df['unit'].nunique()}/{before}"
              " test engines")
    y_true_all = [float(v) for v in rul_path.read_text().split()]

    preds, trues, ae_errs = [], [], []
    padded = 0
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        arr = ((g[feats].values - mins) / span).astype(np.float32)
        if len(arr) < w:                       # pad short units at the front
            arr = np.vstack([np.repeat(arr[:1], w - len(arr), axis=0), arr])
            padded += 1
        win = arr[-w:].reshape(1, w, len(feats))
        preds.append(float(lstm.predict(win, verbose=0)[0][0]) * cap)
        trues.append(y_true_all[int(u) - 1])
        flat = win.reshape(-1, len(feats))          # AE is per-timestep
        ae_errs.append(float(np.mean(
            (ae.predict(flat, verbose=0) - flat) ** 2)))

    preds, trues = np.asarray(preds), np.asarray(trues)
    trues_cap = np.minimum(trues, cap)
    rmse = float(np.sqrt(np.mean((preds - trues_cap) ** 2)))
    mae = float(np.mean(np.abs(preds - trues_cap)))
    print(f"\n[TEST-EVAL:{label}] official {subset} test set "
          f"({len(preds)} engines, {padded} padded):")
    print(f"  LSTM  RMSE={rmse:.2f}  MAE={mae:.2f}  "
          f"PHM08-score={phm_score(trues_cap, preds):.0f}")
    corr = float(np.corrcoef(ae_errs, -trues)[0, 1])   # err vs degradation
    print(f"  AE    recon-error vs true-degradation correlation: "
          f"r={corr:+.3f}")
    show = min(8, len(preds))
    print("  sample pred vs true:",
          [(round(float(p), 1), round(float(t), 1))
           for p, t in zip(preds[:show], trues[:show])])

    # Optional visual artifact: prediction-vs-truth scatter + AE histogram.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.scatter(trues, preds, s=22, alpha=0.75, edgecolor="k",
                    linewidth=0.4)
        lim = max(cap, trues.max())
        ax1.plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
        ax1.set_xlabel("true RUL (cycles)"); ax1.set_ylabel("predicted RUL")
        ax1.set_title(f"LSTM on FD001 test set (RMSE {rmse:.1f})")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.hist(ae_errs, bins=25, color="#2b6cb0", alpha=0.85)
        ax2.set_xlabel("autoencoder reconstruction error")
        ax2.set_ylabel("engines")
        ax2.set_title(f"AE error vs degradation (r={corr:+.2f})")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        out_png = (HERE / "docs" /
                   f"{subset.lower()}_test_evaluation.png")
        out_png.parent.mkdir(exist_ok=True)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"  plot saved -> {out_png}")
    except Exception as exc:                       # noqa: BLE001
        print(f"  (plot skipped: {exc})")


# ===========================================================================
# Main training workflow
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Train AE + LSTM on C-MAPSS "
                                             "FD001")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--window", type=int, default=30,
                    help="LSTM sliding-window length in cycles")
    ap.add_argument("--rul-cap", type=float, default=125.0,
                    help="piecewise-linear RUL cap (standard: 125)")
    ap.add_argument("--epochs-ae", type=int, default=40)
    ap.add_argument("--epochs-lstm", type=int, default=40)
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic data instead of real FD001 files")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; evaluate saved models on "
                         "test_FD001 + RUL_FD001")
    ap.add_argument("--subset", default="FD001",
                    choices=["FD001", "FD002", "FD003", "FD004"],
                    help="C-MAPSS subset to train on (default FD001)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        evaluate_on_test_set(args.data_dir, args.out_dir, "saved models")
        return

    # ------------------------------------------------------------------ data
    df = load_dataset(args.data_dir, args.subset, args.demo)
    df, cond_info = filter_dominant_condition(df, args.subset)
    feats = select_features(df, N_FEATURES)
    df = add_labels(df, args.rul_cap)

    # Split by UNIT so validation sees engines unseen during training.
    val_units = {u for u in df["unit"].unique() if u % 5 == 0}
    tr = df[~df["unit"].isin(val_units)]
    va = df[df["unit"].isin(val_units)]

    mins, span = minmax_fit(tr, feats)

    def norm(frame: pd.DataFrame) -> np.ndarray:
        return ((frame[feats].values - mins) / span).astype(np.float32)

    def per_unit(frame: pd.DataFrame):
        arrays, labels = {}, {}
        for u, g in frame.groupby("unit"):
            g = g.sort_values("cycle")
            arrays[int(u)] = norm(g)
            labels[int(u)] = g["rul"].values.astype(np.float32)
        return arrays, labels

    tr_arr, tr_lab = per_unit(tr)
    va_arr, va_lab = per_unit(va)

    # ------------------------------------------------------- autoencoder
    # Healthy = early-life rows (RUL above 80 % of cap). Train only on
    # those so degraded patterns yield large reconstruction errors.
    healthy_mask = tr["rul"] >= 0.8 * args.rul_cap
    x_healthy = norm(tr[healthy_mask])
    print(f"\n[AUTOENCODER] training on {x_healthy.shape[0]} healthy "
          f"samples ...")

    ae = build_autoencoder(len(feats))
    ae.fit(x_healthy, x_healthy, epochs=args.epochs_ae, batch_size=64,
           validation_split=0.1, verbose=2)

    # Threshold = mean + 4*sigma of healthy sample-wise MSE.
    rec_h = ae.predict(x_healthy, verbose=0)
    err_h = np.mean((rec_h - x_healthy) ** 2, axis=1)
    threshold = float(max(err_h.mean() + 4.0 * err_h.std(), 1e-4))

    # Show separation against clearly degraded (late-life) samples.
    degr_mask = tr["rul"] <= 30
    sep_note = ""
    if degr_mask.any():
        x_deg = norm(tr[degr_mask])
        err_d = np.mean((ae.predict(x_deg, verbose=0) - x_deg) ** 2, axis=1)
        sep_note = (f" | degraded mean={err_d.mean():.5f} "
                    f"(separation x{err_d.mean() / max(err_h.mean(), 1e-9):.1f})")
    print(f"[AUTOENCODER] healthy MSE mean={err_h.mean():.5f} "
          f"std={err_h.std():.5f} -> threshold={threshold:.5f}{sep_note}")

    ae.save(args.out_dir / "autoencoder.h5")

    # --------------------------------------------------------------- LSTM
    x_tr, y_tr = make_windows(tr_arr, tr_lab, args.window)
    x_va, y_va = make_windows(va_arr, va_lab, args.window)
    # Scale targets to [0..1]: huge raw MSE (~1000s) makes Adam collapse to
    # a constant mean predictor very early. Normalized targets train far
    # better; the scale is stored so the backend can invert it.
    cap = float(args.rul_cap)
    print(f"\n[LSTM] windows: train={x_tr.shape}, val={x_va.shape}")

    lstm = build_lstm(args.window, len(feats))
    cbs = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)]
    lstm.fit(x_tr, y_tr / cap, validation_data=(x_va, y_va / cap),
             epochs=args.epochs_lstm, batch_size=64, callbacks=cbs,
             verbose=2)

    pred = lstm.predict(x_va, verbose=0).ravel() * cap
    rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
    mae = float(np.mean(np.abs(pred - y_va)))
    print(f"[LSTM] validation RMSE={rmse:.2f}  MAE={mae:.2f}  "
          f"PHM08-score={phm_score(y_va, pred):.0f}")
    print("[LSTM] sample predictions vs true:",
          [(round(float(p), 1), round(float(t), 1))
           for p, t in zip(pred[:8], y_va[:8])])

    lstm.save(args.out_dir / "lstm_rul.h5")

    # ------------------------------------------------------------ metadata
    meta = {
        "subset": args.subset,
        "op_condition": cond_info,
        "features": feats,
        "min": mins.tolist(),
        "span": span.tolist(),
        "max": (mins + span).tolist(),
        "window": int(args.window),
        "rul_cap": float(args.rul_cap),
        "label_scale": float(args.rul_cap),
        "ae_threshold": threshold,
        "note": ("min/max are C-MAPSS-domain values used at TRAINING time. "
                 "backend.py normalizes FlightGear sensors with its own "
                 "C172P envelope (SENSOR_DEFS); both map to [0..1]."),
    }
    (args.out_dir / "scaler.json").write_text(json.dumps(meta, indent=2))

    # Final gate: official FD001 test-set evaluation with fresh artifacts.
    evaluate_on_test_set(args.data_dir, args.out_dir, "just trained")

    print("\nSaved:")
    for f in sorted(args.out_dir.glob("*")):
        print(f"  {f}")
    print("\nNext: start backend.py  ->  it loads these models automatically.")


if __name__ == "__main__":
    main()
