"""
Collect training data by running the C172P simulator with every fault type
at multiple severity levels. Outputs a CSV suitable for ML training.
"""
import csv
import sys
from pathlib import Path
from sim_engine import generate_sequence, SENSOR_KEYS, FAULT_LIBRARY

OUTPUT = Path(__file__).parent / "training_data.csv"
DURATION = 300       # seconds per scenario
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FAULT_TYPES = list(FAULT_LIBRARY.keys()) + ["none"]
SEEDS_PER_COMBO = 3  # different random seeds per (fault, severity)


def collect():
    rows = []
    total = len(FAULT_TYPES) * len(SEVERITIES) * SEEDS_PER_COMBO
    count = 0
    for fault in FAULT_TYPES:
        # For "none" (healthy) the severity is meaningless, so generate only
        # at sev=0 once, but with many more seeds for a rich healthy manifold.
        if fault == "none":
            sev_list, per_sev = [0.0], SEEDS_PER_COMBO * 8
        else:
            sev_list, per_sev = SEVERITIES, SEEDS_PER_COMBO
        for sev in sev_list:
            for seed_i in range(per_sev):
                count += 1
                seed = hash((fault, sev, seed_i)) % 100000
                seq = generate_sequence(DURATION, fault, sev, seed=seed)

                # Maximum RUL horizon (seconds) when perfectly healthy.
                start_rul = float(DURATION) * 6.0   # 0.5h horizon
                rul = start_rul
                # Healthy baseline for each sensor (used to derive RUL from
                # actual deviation, giving the regressor genuine signal).
                HEALTHY = {
                    "rpm": 2320.0, "egt": 1290.0, "cht": 372.0,
                    "oil_temp": 183.0, "oil_press": 61.0,
                    "fuel_flow": 8.8, "manifold": 27.3,
                }
                # Expected healthy operating range (half-width) for scaling
                RANGE = {
                    "rpm": 120.0, "egt": 60.0, "cht": 30.0,
                    "oil_temp": 8.0, "oil_press": 3.0,
                    "fuel_flow": 0.8, "manifold": 0.6,
                }
                for row in seq:
                    # Deviation index: normalized distance from healthy baseline
                    dev = 0.0
                    for k in SENSOR_KEYS:
                        dev += abs(row[k] - HEALTHY[k]) / RANGE[k]
                    dev /= len(SENSOR_KEYS)
                    # RUL ~ max at healthy, drops as deviation grows, plus
                    # a slow age decay with elapsed time
                    rul = max(0.0, start_rul * (1.0 - min(dev, 1.5)) -
                              row["t"] * 0.5)
                    row["rul_h"] = round(rul / 3600.0, 4)
                    rows.append(row)

                sys.stdout.write(f"\r  [{count}/{total}] fault={fault:8s} "
                                 f"sev={sev:.1f} seed={seed}")
                sys.stdout.flush()

    print(f"\n  Writing {len(rows)} rows to {OUTPUT}")
    fieldnames = ["t"] + SENSOR_KEYS + ["fault", "severity", "rul_h"]
    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("  Done.")


if __name__ == "__main__":
    collect()
