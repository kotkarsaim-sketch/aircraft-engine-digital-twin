#!/usr/bin/env python
"""
Main experiment runner for C172P ML Pipeline.
Executes the complete research pipeline:
1. Data generation with proper splits
2. Feature engineering
3. Multi-model training with cross-validation
4. Comprehensive evaluation
5. Statistical significance testing
6. Ablation studies
6. Model selection and serialization
7. Model card generation
"""
import sys
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml

# Add src to path - C:\pylibs MUST be first for TF/sklearn
import sys
sys.path.insert(0, r"C:\pylibs")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import C172PDataset, load_config
from models.train import (
    train_with_cv, select_best_model, save_model,
    build_classifier_models, build_rul_models
)
from evaluation.evaluate import (
    evaluate_classifier_full, evaluate_rul_full,
    statistical_significance_test, mcnemar_test,
    ablation_study_sensor_subsets, ablation_study_window_sizes,
    plot_confusion_matrix, plot_calibration_curves,
    plot_rul_predictions, generate_model_card
)

warnings.filterwarnings("ignore")


def main():
    # Load configuration
    config = load_config("config.yaml")
    exp_name = config["experiment"]["name"]
    output_dir = Path(config["experiment"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"C172P ML EXPERIMENT: {exp_name}")
    print("=" * 70)

    # Set random seeds for reproducibility
    seed = config["experiment"]["seed"]
    np.random.seed(seed)

    # 1. DATA GENERATION & PREPARATION
    print("\n[1/7] Generating and preparing data...")
    ds = C172PDataset(config)
    raw_df = ds.generate_raw_data()
    raw_df = ds.compute_rul(raw_df)
    raw_df = ds.engineer_features(raw_df)

    train_df, val_df, test_df = ds.split_by_seed(raw_df)

    # Prepare classifier data
    X_train_clf, y_train_clf = ds.prepare_classifier_data(train_df)
    X_val_clf, y_val_clf = ds.prepare_classifier_data(val_df)
    X_test_clf, y_test_clf = ds.prepare_classifier_data(test_df)

    # Prepare RUL data
    window = config["features"]["default_rul_window"]
    X_train_rul, y_train_rul = ds.create_rolling_features(train_df, window)
    X_val_rul, y_val_rul = ds.create_rolling_features(val_df, window)
    X_test_rul, y_test_rul = ds.create_rolling_features(test_df, window)

    # Get groups for CV (seed-based)
    train_groups = train_df["seed"].values
    # Map rolling features back to groups
    def get_groups_for_rolling(df, window):
        groups = []
        for seed, group in df.groupby("seed"):
            n = len(group)
            groups.extend([seed] * max(0, n - window + 1))
        return np.array(groups)

    train_groups_rul = get_groups_for_rolling(train_df, window)

    # Fit scalers on training data only
    ds.fit_scalers(train_df)

    # Transform classifier data
    X_train_clf_s = ds.scaler.transform(X_train_clf)
    X_val_clf_s = ds.scaler.transform(X_val_clf)
    X_test_clf_s = ds.scaler.transform(X_test_clf)

    # Transform RUL data
    X_train_rul_s = ds.rul_scaler.transform(X_train_rul)
    X_val_rul_s = ds.rul_scaler.transform(X_val_rul)
    X_test_rul_s = ds.rul_scaler.transform(X_test_rul)

    # Encode labels
    y_train_clf_enc = ds.label_encoder.transform(y_train_clf)
    y_val_clf_enc = ds.label_encoder.transform(y_val_clf)
    y_test_clf_enc = ds.label_encoder.transform(y_test_clf)

    label_names = ds.label_encoder.classes_.tolist()
    print(f"  Data shapes: clf_train={X_train_clf_s.shape}, rul_train={X_train_rul_s.shape}")
    print(f"  Labels: {label_names}")

    # 2. CLASSIFIER TRAINING WITH CV
    print("\n[2/7] Training fault classifiers with cross-validation...")
    clf_results = train_with_cv(
        X_train_clf_s, y_train_clf_enc, train_groups, config, task="classifier"
    )

    # 3. RUL TRAINING WITH CV
    print("\n[3/7] Training RUL regressors with cross-validation...")
    rul_results = train_with_cv(
        X_train_rul_s, y_train_rul, train_groups_rul, config, task="rul"
    )

    # 4. MODEL SELECTION
    print("\n[4/7] Selecting best models...")
    best_clf_name, best_clf_result = select_best_model(clf_results, "classifier")
    best_rul_name, best_rul_result = select_best_model(rul_results, "rul")

    # Retrain best models on full train+val
    print("\n  Retraining best models on train+val...")
    X_full_clf = np.vstack([X_train_clf_s, X_val_clf_s])
    y_full_clf = np.hstack([y_train_clf_enc, y_val_clf_enc])

    X_full_rul = np.vstack([X_train_rul_s, X_val_rul_s])
    y_full_rul = np.hstack([y_train_rul, y_val_rul])

    clf_models = build_classifier_models(config)
    rul_models = build_rul_models(config)

    best_clf = clf_models[best_clf_name]
    best_clf.fit(X_full_clf, y_full_clf)

    best_rul = rul_models[best_rul_name]
    best_rul.fit(X_full_rul, y_full_rul)

    # 5. COMPREHENSIVE TEST SET EVALUATION
    print("\n[5/7] Evaluating on held-out test set...")
    clf_eval = evaluate_classifier_full(best_clf, X_test_clf_s, y_test_clf_enc, label_names)
    rul_eval = evaluate_rul_full(best_rul, X_test_rul_s, y_test_rul)

    print("\n  CLASSIFIER TEST METRICS:")
    for k, v in clf_eval["metrics"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n  RUL TEST METRICS:")
    for k, v in rul_eval.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 6. STATISTICAL SIGNIFICANCE TESTING
    print("\n[6/7] Running statistical significance tests...")
    stat_tests = {}

    # Compare best classifier vs runner-up
    clf_names = [k for k in clf_results.keys() if "error" not in clf_results[k]]
    if len(clf_names) > 1:
        runner_up = max(
            [k for k in clf_names if k != best_clf_name],
            key=lambda k: clf_results[k]["evaluation"]["cv_scores"]["f1_macro"]["mean"]
        )
        model_a = clf_models[best_clf_name]
        model_a.fit(X_full_clf, y_full_clf)
        model_b = clf_models[runner_up]
        model_b.fit(X_full_clf, y_full_clf)

        stat_tests["classifier_bootstrap"] = statistical_significance_test(
            model_a, model_b, X_test_clf_s, y_test_clf_enc, task="classifier"
        )
        stat_tests["classifier_mcnemar"] = mcnemar_test(
            model_a, model_b, X_test_clf_s, y_test_clf_enc
        )
        print(f"  Classifier vs {runner_up}: p={stat_tests['classifier_bootstrap']['p_value']:.4f}, "
              f"significant={stat_tests['classifier_bootstrap']['significant_at_05']}")

    # Compare best RUL vs runner-up
    rul_names = [k for k in rul_results.keys() if "error" not in rul_results[k]]
    if len(rul_names) > 1:
        runner_up = min(
            [k for k in rul_names if k != best_rul_name],
            key=lambda k: rul_results[k]["evaluation"]["test_mae"]
        )
        model_a = rul_models[best_rul_name]
        model_a.fit(X_full_rul, y_full_rul)
        model_b = rul_models[runner_up]
        model_b.fit(X_full_rul, y_full_rul)

        stat_tests["rul_bootstrap"] = statistical_significance_test(
            model_a, model_b, X_test_rul_s, y_test_rul, task="rul"
        )
        print(f"  RUL vs {runner_up}: p={stat_tests['rul_bootstrap']['p_value']:.4f}, "
              f"significant={stat_tests['rul_bootstrap']['significant_at_05']}")

    # 7. ABLATION STUDIES
    print("\n[7/7] Running ablation studies...")
    ablation_results = {}

    # Sensor subsets for classifier
    print("  Sensor subset ablation (classifier)...")
    ablation_results["sensor_subsets_clf"] = ablation_study_sensor_subsets(
        X_train_clf_s, y_train_clf_enc, train_groups,
        config["features"]["sensor_keys"], config, task="classifier"
    )

    # Sensor subsets for RUL
    print("  Sensor subset ablation (RUL)...")
    ablation_results["sensor_subsets_rul"] = ablation_study_sensor_subsets(
        X_train_rul_s, y_train_rul, train_groups_rul,
        config["features"]["sensor_keys"], config, task="rul"
    )

    # Window sizes for RUL
    print("  Window size ablation (RUL)...")
    ablation_results["window_sizes"] = ablation_study_window_sizes(
        X_train_rul_s, y_train_rul, train_groups_rul,
        config["ablation"]["window_sizes"], config
    )

    # 8. SAVE ARTIFACTS
    print("\n[8/8] Saving models and artifacts...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    save_model(best_clf, best_clf_name, models_dir, "classifier")
    save_model(best_rul, best_rul_name, models_dir, "rul")

    # Save as "best" for serving
    joblib.dump(best_clf, models_dir / "best_classifier.joblib")
    joblib.dump(best_rul, models_dir / "best_rul.joblib")

    # Save artifacts
    ds.save_artifacts(ARTIFACTS_DIR)

    # Save metadata
    meta = {
        "experiment": exp_name,
        "best_classifier": best_clf_name,
        "best_rul": best_rul_name,
        "sensor_keys": config["features"]["sensor_keys"],
        "window": window,
        "rul_max": 1.0,
        "fault_types": label_names,
        "feature_config": {
            "add_deviation_features": config["features"].get("add_deviation_features", True),
            "add_cross_sensor_features": config["features"].get("add_cross_sensor_features", True),
            "add_temporal_features": config["features"].get("add_temporal_features", True),
            "scaler": config["features"].get("scaler", "standard"),
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
    }
    with open(models_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 9. PLOTS
    print("  Generating plots...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        np.array(clf_eval["confusion_matrix"]), label_names,
        plots_dir / "confusion_matrix.png"
    )
    plot_calibration_curves(clf_eval["calibration"], label_names,
                           plots_dir / "calibration_curves.png")
    plot_rul_predictions(y_test_rul, best_rul.predict(X_test_rul_s),
                         plots_dir / "rul_predictions.png")

    # 10. MODEL CARD
    eval_results = {
        "classifier": clf_eval,
        "rul": rul_eval,
        "statistical_tests": stat_tests,
        "ablation": ablation_results,
    }
    generate_model_card(eval_results, config, eval_results, output_dir / "model_card.json")

    # Save full results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "clf_results": {k: v for k, v in clf_results.items() if k != "model"},
            "rul_results": {k: v for k, v in rul_results.items() if k != "model"},
            "test_eval": eval_results,
        }, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Best Classifier: {best_clf_name}")
    print(f"Best RUL Model:  {best_rul_name}")
    print(f"Results saved to: {output_dir}")
    print(f"Models saved to:  {models_dir}")
    print(f"Model Card:       {output_dir / 'model_card.md'}")


if __name__ == "__main__":
    main()