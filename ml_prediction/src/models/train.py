"""
Model training pipeline with multiple baselines, cross-validation, and experiment tracking.
Supports fault classification and RUL regression.
"""
import sys
sys.path.insert(0, r"C:\pylibs")

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, cross_val_score, cross_validate,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore", category=UserWarning)


class ExperimentTracker:
    """Lightweight experiment tracking with JSONL logs."""

    def __init__(self, exp_dir: Path, exp_name: str):
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.log_file = exp_dir / f"{exp_name}_log.jsonl"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def log(self, entry: Dict):
        entry["timestamp"] = time.time()
        entry["exp_name"] = self.exp_name
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_logs(self) -> pd.DataFrame:
        if not self.log_file.exists():
            return pd.DataFrame()
        return pd.read_json(self.log_file, lines=True)


def build_classifier_models(config: dict) -> Dict[str, Any]:
    """Build classifier models from config."""
    models = {}
    for m_cfg in config["models"]["classifier"]["models"]:
        name = m_cfg["name"]
        cls_path = m_cfg["type"]
        params = m_cfg.get("params", {})

        if cls_path == "sklearn.ensemble.RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier
            models[name] = RandomForestClassifier(**params)
        elif cls_path == "xgboost.XGBClassifier" and HAS_XGB:
            models[name] = xgb.XGBClassifier(**params)
        elif cls_path == "lightgbm.LGBMClassifier" and HAS_LGB:
            models[name] = lgb.LGBMClassifier(**params)
        elif cls_path == "sklearn.neural_network.MLPClassifier":
            models[name] = MLPClassifier(**params)
        else:
            print(f"Skipping {name}: {cls_path} not available")
    return models


def build_rul_models(config: dict) -> Dict[str, Any]:
    """Build RUL regression models from config."""
    models = {}
    for m_cfg in config["models"]["rul"]["models"]:
        name = m_cfg["name"]
        cls_path = m_cfg["type"]
        params = m_cfg.get("params", {})

        if cls_path == "sklearn.ensemble.GradientBoostingRegressor":
            models[name] = GradientBoostingRegressor(**params)
        elif cls_path == "sklearn.ensemble.RandomForestRegressor":
            models[name] = RandomForestRegressor(**params)
        elif cls_path == "sklearn.linear_model.Ridge":
            models[name] = Ridge(**params)
        elif cls_path == "sklearn.neural_network.MLPRegressor":
            models[name] = MLPRegressor(**params)
        else:
            print(f"Skipping {name}: {cls_path} not available")
    return models


def evaluate_classifier(
    model, X: np.ndarray, y: np.ndarray,
    cv_splitter, groups: np.ndarray, label_names: List[str]
) -> Dict:
    """Comprehensive classifier evaluation with cross-validation."""
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr": "roc_auc_ovr",
        "balanced_accuracy": "balanced_accuracy",
    }

    cv_results = cross_validate(
        model, X, y, cv=cv_splitter, groups=groups, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )

    # Per-class metrics on full data (for reference)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    per_class = {}
    if y_proba is not None:
        for i, cls in enumerate(label_names):
            mask = (y == i)
            if mask.sum() > 0:
                per_class[cls] = {
                    "precision": precision_score(y == i, y_pred == i, zero_division=0),
                    "recall": recall_score(y == i, y_pred == i, zero_division=0),
                    "f1": f1_score(y == i, y_pred == i, zero_division=0),
                    "support": int(mask.sum()),
                }

    return {
        "cv_scores": {k: {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "train_mean": float(cv_results[f"train_{k}"].mean()) if f"train_{k}" in cv_results else None,
        } for k, v in cv_results.items() if k.startswith("test_")},
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def evaluate_rul(
    model, X: np.ndarray, y: np.ndarray,
    cv_splitter, groups: np.ndarray
) -> Dict:
    """Comprehensive RUL evaluation with cross-validation."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    cv_results = cross_validate(
        model, X, y, cv=cv_splitter, groups=groups, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )

    model.fit(X, y)
    y_pred = model.predict(X)

    # PHM08 scoring function
    def phm08_score(y_true, y_pred):
        d = y_pred - y_true
        score = np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1))
        return float(score)

    phm08 = phm08_score(y, y_pred)

    # Prediction interval coverage (using residuals)
    residuals = y - y_pred
    std_resid = np.std(residuals)
    coverage_90 = np.mean(np.abs(residuals) < 1.645 * std_resid)

    return {
        "cv_scores": {k: {
            "mean": float(-v.mean()) if "neg_" in scoring.get(k.replace("test_", ""), "") else float(v.mean()),
            "std": float(v.std()),
        } for k, v in cv_results.items() if k.startswith("test_")},
        "test_mae": float(mean_absolute_error(y, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "test_r2": float(r2_score(y, y_pred)),
        "test_phm08": phm08,
        "test_coverage_90": float(coverage_90),
        "residual_std": float(std_resid),
    }


def calibration_evaluation(model, X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict:
    """Evaluate probability calibration."""
    if not hasattr(model, "predict_proba"):
        return {}

    y_proba = model.predict_proba(X)
    n_classes = y_proba.shape[1]
    cal_results = {}

    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(y == i, y_proba[:, i], n_bins=n_bins)
        ece = np.mean(np.abs(prob_true - prob_pred))  # Expected Calibration Error
        cal_results[f"class_{i}"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "ece": float(ece),
        }

    # Overall ECE (weighted)
    overall_ece = np.mean([v["ece"] for v in cal_results.values()])
    return {"per_class": cal_results, "overall_ece": float(overall_ece)}


def train_with_cv(
    X: np.ndarray, y: np.ndarray,
    groups: np.ndarray,
    config: dict,
    task: str = "classifier"
) -> Dict:
    """Train multiple models with cross-validation and select best."""
    cv_cfg = config["training"]
    n_folds = cv_cfg["cv_folds"]
    cv_splitter = GroupKFold(n_splits=n_folds)

    if task == "classifier":
        models = build_classifier_models(config)
        eval_fn = evaluate_classifier
        label_names = config["data"]["fault_types"]
    else:
        models = build_rul_models(config)
        eval_fn = evaluate_rul
        label_names = None

    tracker = ExperimentTracker(Path(config["experiment"]["output_dir"]), config["experiment"]["name"])
    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ({task}) ===")
        t0 = time.time()

        try:
            # Pass groups to CV functions via cv_splitter.split
            if task == "classifier":
                eval_result = eval_fn(model, X, y, cv_splitter, groups, label_names)
            else:
                eval_result = eval_fn(model, X, y, cv_splitter, groups)
            train_time = time.time() - t0

            # Fit final model on all data
            model.fit(X, y)

            result = {
                "model": name,
                "task": task,
                "train_time": train_time,
                "evaluation": eval_result,
            }
            results[name] = result

            # Log
            tracker.log({
                "model": name,
                "task": task,
                "train_time": train_time,
                "cv_scores": eval_result.get("cv_scores", {}),
            })
            print(f"  Completed in {train_time:.1f}s")
            for metric, vals in eval_result.get("cv_scores", {}).items():
                print(f"    {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"model": name, "task": task, "error": str(e)}

    return results


def select_best_model(results: Dict, task: str) -> Tuple[str, Any]:
    """Select best model based on primary metric."""
    if task == "classifier":
        # Primary: f1_macro (key is test_f1_macro from cross_validate)
        best_name = max(results.keys(), key=lambda k: results[k].get("evaluation", {}).get("cv_scores", {}).get("test_f1_macro", {}).get("mean", 0))
        primary = results[best_name]["evaluation"]["cv_scores"]["test_f1_macro"]["mean"]
        print(f"\nBest classifier: {best_name} (test_f1_macro={primary:.4f})")
    else:
        # Primary: MAE (lower is better) - key is test_mae
        best_name = min(results.keys(), key=lambda k: results[k].get("evaluation", {}).get("cv_scores", {}).get("test_mae", {}).get("mean", float("inf")))
        primary = results[best_name]["evaluation"]["cv_scores"]["test_mae"]["mean"]
        print(f"\nBest RUL model: {best_name} (MAE={primary:.4f}h)")

    return best_name, results[best_name]


def save_model(model, name: str, path: Path, task: str):
    """Save model with metadata."""
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / f"{name}_{task}.joblib")
    print(f"Saved {name} ({task}) to {path}")


if __name__ == "__main__":
    # Quick test when run directly
    print("Model training module ready.")
    print(f"XGBoost available: {HAS_XGB}")
    print(f"LightGBM available: {HAS_LGB}")