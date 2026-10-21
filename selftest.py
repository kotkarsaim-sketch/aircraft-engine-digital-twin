#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 AIRCRAFT ENGINE DIGITAL TWIN -- SELF TEST SUITE
============================================================================
 Regression tests for the backend's core logic. No FlightGear, no pytest,
 no TensorFlow REQUIRED (TF models are loaded if present, else the
 statistical fallback paths are exercised).

     python selftest.py

 Covers the failure modes discovered during system verification:
   T1  telnet reply parsing
   T2  detector: no CRITICAL from raw scores before calibration
   T3  detector: gross single-sensor deviation flags immediately
   T4  detector: post-calibration nominal data stays NOMINAL
   T5  RUL: synthetic health does NOT drain on healthy engine
       (domain-gap damage leak regression test)
   T6  RUL: LSTM offset anchoring completes and produces sane values
   T7  RUL: damage accrues only while an anomaly is active
   T8  fault manager: overrides ramp correctly and clear cleanly
   T9  simulated engine: all values inside normalization envelopes
============================================================================
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from backend import (AnomalyDetector, FaultManager, RulEstimator,
                     SimulatedEngine, SENSOR_DEFS, VALUE_RE, GROSS_DEV_LIMIT,
                     WINDOW_SIZE)

MODEL_DIR = Path(__file__).resolve().parent / "models"
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))


def norm(vectors) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
print("T1  telnet reply parsing")
samples = {
    "\r\n/engines/engine[0]/rpm = 2451.3 ()\r\n>": 2451.3,
    "\r\n/engines/engine/egt-degf = 1310 (degF)\r\n>": 1310.0,
    "\r\n/controls/engines/engine[0]/throttle = 0.25\r\n>": 0.25,
    "\r\n/engines/engine[0]/running = 1 ()\r\n>": 1.0,
}
for reply, expected in samples.items():
    m = VALUE_RE.search(reply)
    check(f"parse {expected}", bool(m) and float(m.group(1)) == expected)
m = VALUE_RE.search("\r\n# Unknown property\r\n>")
check("unknown property -> no match", m is None)

# ---------------------------------------------------------------------------
print("T2  detector: raw scores cannot read CRITICAL pre-calibration")
det = AnomalyDetector(MODEL_DIR)
det.model = None                       # force fallback statistic path
det.saved_threshold = 0.0096           # pretend a tight training threshold
det.mu = None
rng = np.random.default_rng(0)
base = np.array([0.77, 0.65, 0.62, 0.70, 0.53, 0.59, 0.85], dtype=np.float32)
statuses = set()
for _ in range(20):                    # far fewer than the 45-tick calibration
    x = np.clip(base + rng.normal(0, 0.01, 7), 0, 1).astype(np.float32)
    score, thr, is_anom, status = det.process(x, True, False)
    statuses.add(status)
check("pre-calibration status never CRITICAL", "CRITICAL" not in statuses,
      f"statuses={statuses}")

# ---------------------------------------------------------------------------
print("T3  detector: gross single-sensor deviation flags immediately")
x_bad = base.copy()
x_bad[1] = min(1.0, x_bad[1] + 0.30)   # EGT-style excursion
score, thr, is_anom, status = det.process(x_bad, True, False)
check("gross deviation flagged", is_anom, f"maxdev>={GROSS_DEV_LIMIT}")
check("flagged status is WARNING/CRITICAL", status in ("WARNING", "CRITICAL"))

# ---------------------------------------------------------------------------
print("T4  detector: post-calibration nominal data stays NOMINAL")
det2 = AnomalyDetector(MODEL_DIR)
det2.model = None
det2.mu = base.copy()
det2.calibrated_thr = 0.02             # simulate completed calibration
ok = True
for _ in range(10):
    x = np.clip(base + rng.normal(0, 0.01, 7), 0, 1).astype(np.float32)
    *_, st = det2.process(x, True, False)
    ok = ok and st == "NOMINAL"
check("nominal after calibration", ok)

# ---------------------------------------------------------------------------
print("T5  RUL: healthy engine does not drain synthetic health")
rul = RulEstimator(MODEL_DIR)
det5 = AnomalyDetector(MODEL_DIR)
sim = SimulatedEngine()
fm = FaultManager()
health_samples = []
lstm_seen = None
for tick in range(1, 96):
    ov = {}
    fm.apply(None, None, ov)
    raw = sim.read(ov)
    x01 = norm([np.clip((raw[k] - lo) / (hi - lo), 0, 1)
                for k, _s, lo, hi in SENSOR_DEFS])
    score, thr, is_anom, _st = det5.process(x01, True, False)
    _synth, lstm = rul.step(1.0, score, thr, True, x01,
                            anomaly_active=is_anom)
    health_samples.append(rul.health)
    if lstm is not None and lstm_seen is None:
        lstm_seen = lstm
check("health stays > 0.97 over 95 nominal ticks",
      min(health_samples) > 0.97, f"min={min(health_samples):.4f}")

# ---------------------------------------------------------------------------
print("T6  RUL: LSTM offset anchoring completes")
check("LSTM anchored (bias set)", rul.bias is not None,
      f"bias={rul.bias}")
check("LSTM RUL available and plausible",
      lstm_seen is not None and 0 < lstm_seen <= 125,
      f"first={lstm_seen}")
check("anchored LSTM near synthetic level",
      lstm_seen is not None and abs(lstm_seen - _synth) < 25,
      f"lstm={lstm_seen} synth={round(_synth, 1)}")

# ---------------------------------------------------------------------------
print("T7  RUL: damage accrues only while anomaly is active")
h_before = rul.health
for _ in range(10):                    # nominal ticks: negligible decay
    rul.step(1.0, 0.0, 0.05, True, base, anomaly_active=False)
h_nominal = rul.health
for _ in range(10):                    # anomalous ticks: visible decay
    rul.step(1.0, 0.20, 0.05, True, base, anomaly_active=True)
h_fault = rul.health
check("nominal decay negligible", (h_before - h_nominal) < 0.01,
      f"drop={h_before - h_nominal:.4f}")
check("fault decay significant", (h_nominal - h_fault) > 0.02,
      f"drop={h_nominal - h_fault:.4f}")

# ---------------------------------------------------------------------------
print("T8  fault manager")
fm8 = FaultManager()
check("activate known fault", fm8.activate("oil") is not None)
check("activate unknown fault rejected", fm8.activate("banana") is None)
ov = {}
fm8.apply(None, None, ov)
time.sleep(1.6)
ov.clear()
fm8.apply(None, None, ov)
check("oil-pressure override present and falling",
      "oil-pressure-psi" in ov and ov["oil-pressure-psi"] < 62,
      f"value={ov.get('oil-pressure-psi')}")
check("clear empties faults", fm8.clear() == 1 and not fm8.active)

print("T8b fault library sanity (all 8 fault modes)")
from backend import FAULT_LIBRARY
check("eight fault modes defined", len(FAULT_LIBRARY) == 8,
      f"found={sorted(FAULT_LIBRARY)}")
lib_ok, paths_ok = True, True
for name, spec in FAULT_LIBRARY.items():
    if not spec.get("desc") or not spec.get("targets"):
        lib_ok = False
    for target, fn in spec["targets"].items():
        if not (target.startswith("/") or target in
                {s for _k, s, _lo, _hi in SENSOR_DEFS} | {
                    "oil-temperature-degf", "cht-degf", "fuel-flow-gph"}):
            paths_ok = False
        for t in (0, 60, 300, 900):
            v = fn(t, 1.0)
            if not np.isfinite(v):
                lib_ok = False
check("every fault has desc + finite ramp functions", lib_ok)
check("targets are engine suffixes or absolute control paths", paths_ok)

# mixture fault must emulate real physics in simulation mode.
# Age the activation by 10 s so the lever has travelled to ~0.5.
fm8.activate("mixture")
fm8.active["mixture"] = (time.time() - 10.0, 1.0)
ov.clear()
fm8.apply(None, None, ov)
vals_mix = SimulatedEngine().read(ov)
check("mixture fault: EGT soars in sim mode",
      vals_mix["egt_degf"] > 1500, f"egt={vals_mix['egt_degf']:.0f}F")
check("mixture fault: RPM/FF fade in sim mode",
      vals_mix["rpm"] < 2200 and vals_mix["ff_gph"] < 6,
      f"rpm={vals_mix['rpm']:.0f} ff={vals_mix['ff_gph']:.1f}")

# ---------------------------------------------------------------------------
print("T9  simulated engine envelope sanity")
sim9 = SimulatedEngine()
vals = sim9.read({})
ok = all(lo <= vals[k] <= hi for k, _s, lo, hi in SENSOR_DEFS)
check("all sensors inside normalization envelopes", ok, f"{vals}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
