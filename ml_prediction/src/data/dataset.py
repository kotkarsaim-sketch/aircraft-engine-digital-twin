"""
Data loading, splitting, and preprocessing for C172P ML pipeline.
Implements time-series aware splits to prevent data leakage.
"""
import sys
sys.path.insert(0, r"C:\pylibs")

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "sim"))

from sim_engine import SimEngine, FAULT_LIBRARY, SENSOR_KEYS


class C172PDataset:
    """Cessna 172P dataset with proper train/val/test splits by seed."""

    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.feat_cfg = config["features"]
        self.sensor_keys = self.feat_cfg["sensor_keys"]
        self.scaler = None
        self.label_encoder = None
        self.rul_scaler = None

    def generate_raw_data(self, force_regenerate: bool = False) -> pd.DataFrame:
        """Generate or load raw simulation data."""
        cache_path = Path("data/raw_data.parquet")
        if cache_path.exists() and not force_regenerate:
            return pd.read_parquet(cache_path)

        print("Generating simulation data...")
        eng = SimEngine(seed=self.config["experiment"]["seed"])
        rows = []

        # Healthy data: many seeds, only severity 0.0
        for seed_i in range(self.data_cfg["n_healthy_seeds"]):
            seed = hash(("none", 0.0, seed_i)) % 100000
            seq = eng.generate_sequence(
                duration=self.data_cfg["duration_s"],
                fault="none",
                severity=0.0,
                tick_hz=self.data_cfg["tick_hz"],
                seed=seed,
            )
            for row in seq:
                row["seed"] = seed
                row["severity"] = 0.0
            rows.extend(seq)

        # Fault data: all fault types, all severities
        fault_types = [f for f in self.data_cfg["fault_types"] if f != "none"]
        for fault in fault_types:
            for sev in self.data_cfg["severities"]:
                if sev == 0.0:
                    continue
                for seed_i in range(self.data_cfg["n_samples_per_fault_severity"]):
                    seed = hash((fault, sev, seed_i)) % 100000
                    seq = eng.generate_sequence(
                        duration=self.data_cfg["duration_s"],
                        fault=fault,
                        severity=sev,
                        tick_hz=self.data_cfg["tick_hz"],
                        seed=seed,
                    )
                    for row in seq:
                        row["seed"] = seed
                        row["severity"] = sev
                    rows.extend(seq)

        df = pd.DataFrame(rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Generated {len(df)} samples, cached to {cache_path}")
        return df

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RUL from sensor deviation (physics-informed)."""
        healthy = self.data_cfg["healthy_baseline"]
        ranges = self.data_cfg["sensor_ranges"]

        df = df.copy()
        rul_vals = []
        for _, row in df.iterrows():
            dev = 0.0
            for k in self.sensor_keys:
                dev += abs(row[k] - healthy[k]) / ranges[k]
            dev /= len(self.sensor_keys)
            # Max RUL 0.5h, decays with deviation + time
            max_rul = self.data_cfg["duration_s"] * 6.0 / 3600.0  # 0.5h
            rul = max(0.0, max_rul * (1.0 - min(dev, 1.5)) - row["t"] * 0.5 / 3600.0)
            rul_vals.append(rul)
        df["rul_h"] = rul_vals
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for RUL regression."""
        df = df.copy()
        healthy = self.data_cfg["healthy_baseline"]
        ranges = self.data_cfg["sensor_ranges"]

        if self.feat_cfg.get("add_deviation_features", True):
            for k in self.sensor_keys:
                df[f"{k}_dev"] = (df[k] - healthy[k]) / ranges[k]
            df["mean_abs_dev"] = df[[f"{k}_dev" for k in self.sensor_keys]].abs().mean(axis=1)
            df["max_abs_dev"] = df[[f"{k}_dev" for k in self.sensor_keys]].abs().max(axis=1)

        if self.feat_cfg.get("add_cross_sensor_features", True):
            # Key cross-sensor ratios
            df["egt_cht_ratio"] = df["egt"] / df["cht"]
            df["oil_temp_press_ratio"] = df["oil_temp"] / df["oil_press"].clip(lower=1)
            df["fuel_manifold_ratio"] = df["fuel_flow"] / df["manifold"].clip(lower=1)

        if self.feat_cfg.get("add_temporal_features", True):
            df["t_norm"] = df["t"] / self.data_cfg["duration_s"]
            df["t_sin"] = np.sin(2 * np.pi * df["t"] / self.data_cfg["duration_s"])
            df["t_cos"] = np.cos(2 * np.pi * df["t"] / self.data_cfg["duration_s"])

        return df

    def create_rolling_features(self, df: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create rolling-window features for RUL (per-seed to avoid leakage)."""
        X_list, y_list = [], []
        feature_cols = [c for c in df.columns if c not in
                       ["fault", "severity", "rul_h", "seed", "t"]]

        for seed, group in df.groupby("seed"):
            group = group.sort_values("t").reset_index(drop=True)
            X_seed = group[feature_cols].values
            y_seed = group["rul_h"].values

            for i in range(window - 1, len(X_seed)):
                # Rolling mean over window
                win_mean = X_seed[i - window + 1:i + 1].mean(axis=0)
                # Add current timestep
                X_list.append(np.hstack([win_mean, group.iloc[i]["t"]]))
                y_list.append(y_seed[i])

        return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

    def split_by_seed(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by unique seeds to prevent temporal leakage."""
        seeds = df["seed"].unique()
        np.random.seed(self.config["experiment"]["seed"])
        np.random.shuffle(seeds)

        n = len(seeds)
        n_train = int(n * self.data_cfg["train_split"])
        n_val = int(n * self.data_cfg["val_split"])

        train_seeds = set(seeds[:n_train])
        val_seeds = set(seeds[n_train:n_train + n_val])
        test_seeds = set(seeds[n_train + n_val:])

        train_df = df[df["seed"].isin(train_seeds)].copy()
        val_df = df[df["seed"].isin(val_seeds)].copy()
        test_df = df[df["seed"].isin(test_seeds)].copy()

        print(f"Split: train={len(train_seeds)} seeds ({len(train_df)} rows), "
              f"val={len(val_seeds)} seeds ({len(val_df)} rows), "
              f"test={len(test_seeds)} seeds ({len(test_df)} rows)")
        return train_df, val_df, test_df

    def prepare_classifier_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for fault classification."""
        X = df[self.sensor_keys].values.astype(np.float32)
        y = df["fault"].values
        return X, y

    def fit_scalers(self, train_df: pd.DataFrame):
        """Fit scalers on training data only."""
        scaler_type = self.feat_cfg.get("scaler", "standard")
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")

        X_train, _ = self.prepare_classifier_data(train_df)
        self.scaler.fit(X_train)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(train_df["fault"])

        # RUL scaler on rolling features
        window = self.feat_cfg["default_rul_window"]
        X_rul_train, _ = self.create_rolling_features(train_df, window)
        self.rul_scaler = StandardScaler()
        self.rul_scaler.fit(X_rul_train)

    def transform_classifier(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self.prepare_classifier_data(df)
        X_s = self.scaler.transform(X)
        y_enc = self.label_encoder.transform(y)
        return X_s, y_enc

    def transform_rul(self, df: pd.DataFrame, window: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if window is None:
            window = self.feat_cfg["default_rul_window"]
        X, y = self.create_rolling_features(df, window)
        X_s = self.rul_scaler.transform(X)
        return X_s, y

    def save_artifacts(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.label_encoder, path / "label_encoder.joblib")
        joblib.dump(self.rul_scaler, path / "rul_scaler.joblib")

    @classmethod
    def load_artifacts(cls, path: Path, config: dict):
        obj = cls(config)
        obj.scaler = joblib.load(path / "scaler.joblib")
        obj.label_encoder = joblib.load(path / "label_encoder.joblib")
        obj.rul_scaler = joblib.load(path / "rul_scaler.joblib")
        return obj


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_config()
    ds = C172PDataset(cfg)
    raw = ds.generate_raw_data()
    raw = ds.compute_rul(raw)
    raw = ds.engineer_features(raw)
    train, val, test = ds.split_by_seed(raw)
    ds.fit_scalers(train)
    ds.save_artifacts(Path("models/artifacts"))
    print("Data preparation complete.")