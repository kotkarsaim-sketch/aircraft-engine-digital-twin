"""
Comprehensive evaluation suite with statistical significance testing,
ablation studies, and model card generation.
"""
import sys
sys.path.insert(0, r"C:\pylibs")

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    brier_score_loss, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
from scipy import stats

warnings.filterwarnings("ignore")


def mcnemar_test_manual(table):
    """Manual implementation of McNemar's test."""
    # table is [[n00, n01], [n10, n11]]
    n01 = table[0][1]
    n10 = table[1][0]
    if n01 + n10 > 0:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p = 1 - stats.chi2.cdf(chi2, 1)
    else:
        chi2 = 0
        p = 1.0
    return chi2, p


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM08 scoring function for RUL."""
    d = y_pred - y_true
    score = np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1))
    return float(score)


def evaluate_classifier_full(
    model, X: np.ndarray, y: np.ndarray,
    label_names: List[str]
) -> Dict:
    """Complete classifier evaluation on test set."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    # Basic metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, y_pred)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y, y_proba, multi_class="ovr"))
            metrics["roc_auc_ovo"] = float(roc_auc_score(y, y_proba, multi_class="ovo"))
        except ValueError:
            metrics["roc_auc_ovr"] = None
            metrics["roc_auc_ovo"] = None

    # Per-class metrics
    per_class = {}
    for i, cls in enumerate(label_names):
        mask = (y == i)
        if mask.sum() > 0:
            per_class[cls] = {
                "precision": float(precision_score(y == i, y_pred == i, zero_division=0)),
                "recall": float(recall_score(y == i, y_pred == i, zero_division=0)),
                "f1": float(f1_score(y == i, y_pred == i, zero_division=0)),
                "support": int(mask.sum()),
            }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Calibration
    calibration = {}
    if y_proba is not None:
        for i, cls in enumerate(label_names):
            prob_true, prob_pred = calibration_curve(y == i, y_proba[:, i], n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred))
            calibration[cls] = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
                "ece": float(ece),
            }
        overall_ece = np.mean([v["ece"] for v in calibration.values()])
        calibration["overall_ece"] = float(overall_ece)

    return {
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "calibration": calibration,
        "classification_report": classification_report(y, y_pred, target_names=label_names, output_dict=True),
    }


def evaluate_rul_full(
    model, X: np.ndarray, y: np.ndarray
) -> Dict:
    """Complete RUL evaluation on test set."""
    y_pred = model.predict(X)

    residuals = y - y_pred
    std_resid = np.std(residuals)

    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
        "mape": float(np.mean(np.abs(residuals / np.maximum(y, 1e-6))) * 100),
        "phm08": phm08_score(y, y_pred),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(std_resid),
        "residual_skew": float(stats.skew(residuals)),
        "coverage_90": float(np.mean(np.abs(residuals) < 1.645 * std_resid)),
        "max_error": float(np.max(np.abs(residuals))),
    }


def statistical_significance_test(
    model_a, model_b, X: np.ndarray, y: np.ndarray,
    task: str = "classifier", n_iterations: int = 1000
) -> Dict:
    """Paired statistical tests between two models."""
    # Bootstrap test
    n = len(X)
    np.random.seed(42)
    diffs = []

    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        X_bs, y_bs = X[idx], y[idx]

        if task == "classifier":
            pred_a = model_a.predict(X_bs)
            pred_b = model_b.predict(X_bs)
            diff = f1_score(y_bs, pred_a, average="macro") - f1_score(y_bs, pred_b, average="macro")
        else:
            pred_a = model_a.predict(X_bs)
            pred_b = model_b.predict(X_bs)
            diff = mean_absolute_error(y_bs, pred_a) - mean_absolute_error(y_bs, pred_b)

        diffs.append(diff)

    diffs = np.array(diffs)
    p_value = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))  # two-tailed
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

    return {
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "p_value": float(p_value),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "significant_at_05": p_value < 0.05,
    }


def mcnemar_test(model_a, model_b, X: np.ndarray, y: np.ndarray) -> Dict:
    """McNemar's test for classifier comparison."""
    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)

    # Contingency table
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)

    n00 = np.sum(~correct_a & ~correct_b)  # both wrong
    n01 = np.sum(~correct_a & correct_b)   # a wrong, b right
    n10 = np.sum(correct_a & ~correct_b)   # a right, b wrong
    n11 = np.sum(correct_a & correct_b)    # both right

    # McNemar test statistic
    if n01 + n10 > 0:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p = 1 - stats.chi2.cdf(chi2, 1)
    else:
        chi2 = 0
        p = 1.0

    return {
        "n00": int(n00), "n01": int(n01), "n10": int(n10), "n11": int(n11),
        "chi2": float(chi2), "p_value": float(p),
        "significant": p < 0.05,
    }


def ablation_study_sensor_subsets(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    sensor_names: List[str], config: dict,
    task: str = "classifier"
) -> Dict:
    """Evaluate model performance with different sensor subsets."""
    subsets = config["ablation"]["sensor_subsets"]
    results = {}

    cv_splitter = GroupKFold(n_splits=config["training"]["cv_folds"])

    for subset in subsets:
        name = subset["name"]
        sensors = subset["sensors"]
        idx = [sensor_names.index(s) for s in sensors]
        X_sub = X[:, idx]

        if task == "classifier":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            scoring = "f1_macro"
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
            scoring = "neg_mean_absolute_error"

        scores = cross_val_score(model, X_sub, y, groups=groups, cv=cv_splitter, scoring=scoring, n_jobs=-1)
        results[name] = {
            "sensors": sensors,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
        }
        print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f} ({sensors})")

    return results


def ablation_study_window_sizes(
    X_rul: np.ndarray, y_rul: np.ndarray, groups: np.ndarray,
    window_sizes: List[int], config: dict
) -> Dict:
    """Evaluate RUL model with different window sizes."""
    results = {}
    cv_splitter = GroupKFold(n_splits=config["training"]["cv_folds"])

    for w in window_sizes:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        scores = cross_val_score(model, X_rul, y_rul, groups=groups, cv=cv_splitter,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
        results[f"window_{w}"] = {
            "window": w,
            "mae_mean": float(-scores.mean()),
            "mae_std": float(scores.std()),
        }
        print(f"  window={w}: MAE={-scores.mean():.4f} ± {scores.std():.4f}")

    return results


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_calibration_curves(calibration: Dict, label_names: List[str], save_path: Path):
    """Plot calibration curves for all classes."""
    n_classes = len(label_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))

    if n_classes == 1:
        axes = [axes]

    for i, cls in enumerate(label_names):
        ax = axes[i]
        cal = calibration.get(cls, {})
        if "prob_true" in cal:
            ax.plot(cal["prob_pred"], cal["prob_true"], "o-", label=cls)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration: {cls}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Plot RUL predictions vs actual."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0].plot([0, max(y_true.max(), y_pred.max())], [0, max(y_true.max(), y_pred.max())], "r--")
    axes[0].set_xlabel("True RUL (h)")
    axes[0].set_ylabel("Predicted RUL (h)")
    axes[0].set_title("Predicted vs True")
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_true, residuals, alpha=0.5, s=10)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("True RUL (h)")
    axes[1].set_ylabel("Residual (h)")
    axes[1].set_title("Residuals")
    axes[1].grid(True, alpha=0.3)

    # Distribution
    axes[2].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    axes[2].axvline(0, color="r", linestyle="--")
    axes[2].set_xlabel("Residual (h)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_model_card(
    results: Dict,
    config: dict,
    eval_results: Dict,
    save_path: Path
) -> Dict:
    """Generate a model card following the Model Card Toolkit format."""
    card = {
        "model_details": {
            "name": f"C172P-{config['experiment']['name']}",
            "version": "1.0.0",
            "type": "Fault Classification + RUL Regression",
            "framework": "scikit-learn",
            "license": "MIT",
        },
        "intended_use": config["model_card"]["intended_use"],
        "limitations": config["model_card"]["limitations"],
        "ethical_considerations": config["model_card"]["ethical_considerations"],
        "training_data": config["model_card"]["training_data"],
        "evaluation": {
            "classifier": eval_results.get("classifier", {}),
            "rul": eval_results.get("rul", {}),
        },
        "statistical_tests": eval_results.get("statistical_tests", {}),
        "ablation_studies": eval_results.get("ablation", {}),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(card, f, indent=2)

    # Also save as markdown
    md_path = save_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(f"# Model Card: {card['model_details']['name']}\n\n")
        f.write(f"**Version:** {card['model_details']['version']}  \n")
        f.write(f"**Type:** {card['model_details']['type']}  \n")
        f.write(f"**Framework:** {card['model_details']['framework']}  \n\n")

        f.write("## Intended Use\n")
        f.write(f"{card['intended_use']}\n\n")

        f.write("## Limitations\n")
        for lim in card["limitations"]:
            f.write(f"- {lim}\n")
        f.write("\n")

        f.write("## Ethical Considerations\n")
        for eth in card["ethical_considerations"]:
            f.write(f"- {eth}\n")
        f.write("\n")

        f.write("## Training Data\n")
        for k, v in card["training_data"].items():
            f.write(f"- **{k}:** {v}\n")
        f.write("\n")

        f.write("## Evaluation Results\n")
        if "classifier" in card["evaluation"]:
            f.write("### Fault Classifier\n")
            clf = card["evaluation"]["classifier"]
            f.write(f"- Accuracy: {clf.get('metrics', {}).get('accuracy', 'N/A'):.4f}\n")
            f.write(f"- F1 Macro: {clf.get('metrics', {}).get('f1_macro', 'N/A'):.4f}\n")
            f.write(f"- Balanced Accuracy: {clf.get('metrics', {}).get('balanced_accuracy', 'N/A'):.4f}\n")
            f.write(f"- Cohen's Kappa: {clf.get('metrics', {}).get('cohen_kappa', 'N/A'):.4f}\n")
            f.write(f"- Overall ECE: {clf.get('calibration', {}).get('overall_ece', 'N/A'):.4f}\n\n")

        if "rul" in card["evaluation"]:
            f.write("### RUL Regressor\n")
            rul = card["evaluation"]["rul"]
            f.write(f"- MAE: {rul.get('mae', 'N/A'):.4f} h\n")
            f.write(f"- RMSE: {rul.get('rmse', 'N/A'):.4f} h\n")
            f.write(f"- R²: {rul.get('r2', 'N/A'):.4f}\n")
            f.write(f"- PHM08 Score: {rul.get('phm08', 'N/A'):.2f}\n")
            f.write(f"- 90% Coverage: {rul.get('coverage_90', 'N/A'):.4f}\n\n")

    print(f"Model card saved to {save_path} and {md_path}")
    return card


if __name__ == "__main__":
    print("Evaluation module ready.")