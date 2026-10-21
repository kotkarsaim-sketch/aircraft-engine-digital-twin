"""
Cessna 172P Engine Simulator — Research Grade
Procedural physics-informed simulation with configurable fault injection.
"""
import math
import random
import time
from typing import Dict, List, Optional
import numpy as np

SENSOR_KEYS = ["rpm", "egt", "cht", "oil_temp", "oil_press", "fuel_flow", "manifold"]

FAULT_LIBRARY = {
    "none": {"targets": {}},
    "egt": {
        "targets": {
            "egt": lambda s: 1290 + s * 560,
            "oil_temp": lambda s: 183 + s * 20,
        },
    },
    "cht": {
        "targets": {
            "cht": lambda s: 372 + s * 158,
            "oil_temp": lambda s: 183 + s * 18,
        },
    },
    "oil": {
        "targets": {
            "oil_press": lambda s: 61 - s * 51,
            "oil_temp": lambda s: 183 + s * 25,
        },
    },
    "oiltemp": {
        "targets": {
            "oil_temp": lambda s: 183 + s * 67,
        },
    },
    "ff": {
        "targets": {
            "fuel_flow": lambda s: 8.8 - s * 6.8,
        },
    },
    "mp": {
        "targets": {
            "manifold": lambda s: 27.3 - s * 15.3,
        },
    },
    "multi": {
        "targets": {
            "egt": lambda s: 1290 + s * 560,
            "oil_press": lambda s: 61 - s * 51,
            "oil_temp": lambda s: 183 + s * 55,
        },
    },
}

HEALTHY_BASELINE = {
    "rpm": 2320.0,
    "egt": 1290.0,
    "cht": 372.0,
    "oil_temp": 183.0,
    "oil_press": 61.0,
    "fuel_flow": 8.8,
    "manifold": 27.3,
}

SENSOR_RANGES = {
    "rpm": 120.0,
    "egt": 60.0,
    "cht": 30.0,
    "oil_temp": 8.0,
    "oil_press": 3.0,
    "fuel_flow": 0.8,
    "manifold": 0.6,
}


class SimEngine:
    """Cessna 172P engine simulator with realistic sensor dynamics."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._gauss_cache = None

    def _gauss(self) -> float:
        """Box-Muller transform for Gaussian noise."""
        if self._gauss_cache is not None:
            val = self._gauss_cache
            self._gauss_cache = None
            return val
        u1 = self._rng.random()
        u2 = self._rng.random()
        while u1 == 0:
            u1 = self._rng.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        self._gauss_cache = z1
        return z0

    def read(self, t: float, fault: str = "none", severity: float = 0.0) -> Dict[str, float]:
        """Return one tick of sensor data at elapsed time t (seconds)."""
        g = self._gauss
        s = max(0.0, min(1.0, severity))

        # Baseline with realistic coupling and noise
        vals = {
            "rpm": 2320 + 140 * math.sin(0.05 * t) + g() * 25,
            "egt": 1290 + 30 * math.sin(0.033 * t) + g() * 9,
            "cht": 372 + 14 * math.sin(0.021 * t) + g() * 5,
            "oil_temp": 183 + 4 * math.sin(0.008 * t) + g() * 2,
            "oil_press": 61 + 1.5 * math.sin(0.04 * t) + g() * 1.2,
            "manifold": 27.3 + 0.4 * math.sin(0.05 * t) + g() * 0.1,
        }
        # Fuel flow correlates with RPM
        vals["fuel_flow"] = 8.8 + (vals["rpm"] - 2320) / 140.0 * 0.5 + g() * 0.15

        # Apply fault override
        if fault in FAULT_LIBRARY and fault != "none":
            for key, fn in FAULT_LIBRARY[fault]["targets"].items():
                vals[key] = fn(s)

        return vals

    def generate_sequence(
        self,
        duration: float = 300.0,
        fault: str = "none",
        severity: float = 0.0,
        tick_hz: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Generate a full simulation sequence."""
        if seed is not None:
            self._rng = random.Random(seed)
            self._gauss_cache = None

        dt = 1.0 / tick_hz
        n_steps = int(duration * tick_hz)
        seq = []

        for i in range(n_steps):
            t = i * dt
            vals = self.read(t, fault, severity)
            row = {"t": round(t, 3)}
            row.update({k: round(v, 3) for k, v in vals.items()})
            row["fault"] = fault
            row["severity"] = round(severity, 3)
            seq.append(row)

        return seq

    @staticmethod
    def compute_deviation(vals: Dict[str, float]) -> float:
        """Compute normalized deviation from healthy baseline."""
        dev = 0.0
        for k in SENSOR_KEYS:
            dev += abs(vals[k] - HEALTHY_BASELINE[k]) / SENSOR_RANGES[k]
        return dev / len(SENSOR_KEYS)

    @staticmethod
    def compute_rul(vals: Dict[str, float], t: float, max_rul_h: float = 0.5) -> float:
        """Compute RUL from sensor deviation and elapsed time."""
        dev = SimEngine.compute_deviation(vals)
        rul = max(0.0, max_rul_h * (1.0 - min(dev, 1.5)) - t / 3600.0 * 0.5)
        return rul


if __name__ == "__main__":
    # Quick sanity check
    eng = SimEngine(seed=42)
    for fault in ["none", "egt", "oil", "multi"]:
        seq = eng.generate_sequence(duration=10, fault=fault, severity=0.5, tick_hz=1)
        print(f"{fault}: t=0 {seq[0]} | t=9 {seq[-1]}")