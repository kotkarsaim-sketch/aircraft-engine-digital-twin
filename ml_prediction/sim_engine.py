"""
Cessna 172P engine simulator — standalone version for ML data collection.
Generates realistic telemetry with sensor noise, cross-parameter coupling,
and configurable fault injection with severity scaling.
"""
import math
import random
import time

SENSOR_KEYS = ["rpm", "egt", "cht", "oil_temp", "oil_press", "fuel_flow", "manifold"]

FAULT_LIBRARY = {
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


class SimEngine:
    def __init__(self, seed=None):
        self.t0 = time.time()
        self._rng = random.Random(seed)

    def read(self, t: float, fault: str = "none", severity: float = 0.0) -> dict:
        """Return one tick of sensor data at elapsed time t (seconds)."""
        r = self._rng.gauss
        vals = {
            "rpm": 2320 + 140 * math.sin(0.05 * t) + r(0, 25),
            "egt": 1290 + 30 * math.sin(0.033 * t) + r(0, 9),
            "cht": 372 + 14 * math.sin(0.021 * t) + r(0, 5),
            "oil_temp": 183 + 4 * math.sin(0.008 * t) + r(0, 2),
            "oil_press": 61 + 1.5 * math.sin(0.04 * t) + r(0, 1.2),
            "fuel_flow": 8.8 + r(0, 0.15),
            "manifold": 27.3 + 0.4 * math.sin(0.05 * t) + r(0, 0.1),
        }
        # Fuel flow correlates with RPM
        vals["fuel_flow"] = 8.8 + (vals["rpm"] - 2320) / 140.0 * 0.5 + r(0, 0.15)

        # Apply fault override
        if fault in FAULT_LIBRARY:
            s = max(0.0, min(1.0, severity))
            for key, fn in FAULT_LIBRARY[fault]["targets"].items():
                vals[key] = fn(s)

        return vals


def generate_sequence(duration: int = 300, fault: str = "none",
                       severity: float = 0.0, tick_hz: float = 1.0,
                       seed: int = 42) -> list[dict]:
    """Generate a full simulation sequence as a list of dicts."""
    eng = SimEngine(seed=seed)
    dt = 1.0 / tick_hz
    rows = []
    for i in range(int(duration * tick_hz)):
        t = i * dt
        vals = eng.read(t, fault, severity)
        rows.append({
            "t": round(t, 2),
            **{k: round(v, 2) for k, v in vals.items()},
            "fault": fault,
            "severity": round(severity, 2),
        })
    return rows
