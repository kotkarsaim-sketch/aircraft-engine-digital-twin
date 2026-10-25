#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 AIRCRAFT ENGINE DIGITAL TWIN -- BACKEND SERVICE
============================================================================
 Connects to FlightGear (Cessna 172P) over its telnet property interface,
 polls engine sensors once per second, runs:

   1. An AUTOENCODER anomaly detector  (trained by train_models.py on the
      NASA C-MAPSS FD001 "healthy" data; falls back to an online statistical
      detector if TensorFlow or the model file is unavailable).
   2. An LSTM RUL regressor (sliding window of normalized sensors), fused
      with a SYNTHETIC degradation model driven by accumulated anomaly score
      so that Remaining Useful Life visibly decreases during a fault demo.

 and broadcasts a JSON snapshot every second over a WebSocket on port 8765
 to the browser dashboard (index.html + dashboard.js).

 MESSAGE SCHEMA (one JSON object per second):
 {
   "timestamp":     float,          unix time
   "mode":          "FLIGHTGEAR" | "SIMULATION",
   "engine_running" : bool,
   "sensors": { key: value, ... }   engineering units, see SENSOR_DEFS
   "normalized":    [7 floats 0..1] min-max normalized vector
   "anomaly_score": float,          reconstruction error / deviation energy
   "threshold":     float,          current anomaly threshold
   "is_anomaly":    bool,
   "status":        "STANDBY"|"NOMINAL"|"WARNING"|"CRITICAL",
   "health":        float 0..1,     synthetic health index
   "rul_lstm_h":    float | null,   LSTM estimate in hours (cycles)
   "rul_synth_h":   float,          synthetic-degradation estimate
   "rul_h":         float,          fused RUL displayed on dashboard
   "faults":        [names]         active injected faults
 }

 INTERACTIVE CONSOLE COMMANDS (type them into THIS terminal):
   start       - cheat-start the C172P engine (sets mixture/throttle/running)
   stop        - shut the engine down
   fault egt   - hot-section distress: EGT ramps toward 1850 F
   fault oil   - oil leak: oil pressure decays toward ~10 psi
   fault mp    - power loss: manifold pressure decays toward 12 inHg
   clear       - remove all injected faults
   heal        - reset synthetic health to 1.0 and re-calibrate detector
   status      - print one-line summary
   help        - list commands
   quit        - exit

 REQUIREMENTS:  pip install numpy websockets tensorflow-cpu
 FlightGear must run with:   --telnet=5500
 If FlightGear cannot be reached, this service automatically switches to a
 built-in SIMULATION mode so the whole pipeline stays demonstrable offline.
============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import socket
import sys
import threading
import urllib.request
import urllib.error

# TensorFlow installed via --target C:\pylibs on Windows Store Python
_pylibs = r"C:\pylibs"
if _pylibs not in sys.path:
    sys.path.insert(0, _pylibs)
import time
from collections import deque
from pathlib import Path

import numpy as np


def clamp(v, lo, hi):
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, v))

# ---------------------------------------------------------------------------
# Optional dependencies (fail loudly only when truly needed)
# ---------------------------------------------------------------------------
try:
    import websockets
except ImportError:
    print("FATAL: missing 'websockets'. Install with:  pip install websockets")
    sys.exit(1)

try:
    import tensorflow as tf                      # noqa: F401  (used lazily)
    TF_AVAILABLE = True
except Exception as _exc:                        # noqa: BLE001 - any TF failure
    TF_AVAILABLE = False
    _TF_IMPORT_ERROR = str(_exc)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

TELNET_HOST = "127.0.0.1"
TELNET_PORT = 5500            # FlightGear --telnet=5500
WS_PORT = 8765                # WebSocket port for the browser dashboard
POLL_INTERVAL_S = 1.0         # sensor polling period
CONNECT_RETRIES = 5           # telnet connection attempts at startup
CONNECT_RETRY_DELAY_S = 3.0

WINDOW_SIZE = 30              # sliding window length fed to the LSTM (~30 s)
RUL_MAX_HOURS = 125.0         # RUL scale; matches train_models.py rul_cap
BASELINE_AGING_PER_SEC = 0.00008   # tiny nominal wear so health decays slowly
DAMAGE_GAIN = 0.035                # extra health loss per sec per unit of
                                   # anomaly-score EXCESS above threshold
DEFAULT_THRESHOLD = 0.01      # fallback detector threshold (sum-of-squares
                              # scale) used until online calibration finishes
FAULT_DECAY_PER_SEC = 0.004   # extra health loss per second WHILE an anomaly
                              # is active (independent of score magnitude)
MU_ALPHA = 0.05               # nominal-profile adaptation rate per tick.
                              # Must track slow cruise wander (RPM sinusoid
                              # amplitude ~0.05 normalized) without being so
                              # fast that sustained faults get absorbed.

ENGINE_ON_RPM = 300           # below this we consider the engine stopped
GROSS_DEV_LIMIT = 0.08        # fallback detector: any normalized sensor
                              # deviating more than this from its adapted
                              # nominal profile flags immediately

# Regex extracting the first numeric value from a FlightGear telnet reply,
# e.g. "\r\n/engines/engine[0]/rpm = 2451.3 ()\r\n>" -> "2451.3"
VALUE_RE = re.compile(r"=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

# Engine sensor definitions.
#   key        -> JSON field name used on the dashboard
#   suffix     -> property under the resolved engine prefix, e.g.
#                 "/engines/engine[0]" + "/rpm"
#   min / max  -> normalization envelope for the ML models (C172P cruise)
SENSOR_DEFS = [
    # key          path suffix             lo      hi
    ("rpm",        "rpm",                  0.0,   3000.0),
    ("egt_degf",   "egt-degf",             0.0,   2000.0),
    ("cht_degf",   "cht-degf",             0.0,    600.0),
    ("oil_temp_f", "oil-temperature-degf", 0.0,    260.0),
    ("oil_psi",    "oil-pressure-psi",     0.0,    115.0),
    ("ff_gph",     "fuel-flow-gph",        0.0,     15.0),
    ("mp_inhg",    "mp-osi",               0.0,     32.0),
]
SENSOR_KEYS = [d[0] for d in SENSOR_DEFS]

# ---------------------------------------------------------------------------
# ML Prediction Integration
# ---------------------------------------------------------------------------
ML_API_URL = "http://localhost:5000/predict"
ML_SENSOR_MAP = {
    "rpm": "rpm",
    "egt_degf": "egt",
    "cht_degf": "cht",
    "oil_temp_f": "oil_temp",
    "oil_psi": "oil_press",
    "ff_gph": "fuel_flow",
    "mp_inhg": "manifold",
}
ML_SENSOR_KEYS = ["rpm", "egt", "cht", "oil_temp", "oil_press", "fuel_flow", "manifold"]


def call_ml_predict(raw_sensors: dict, t: float) -> dict | None:
    """Call the ML prediction API with mapped sensor values.
    Returns dict with fault, confidence, probabilities, rul_h or None on failure."""
    try:
        # Map backend sensor names to ML model sensor names
        ml_sensors = {}
        for backend_key, ml_key in ML_SENSOR_MAP.items():
            if backend_key in raw_sensors:
                ml_sensors[ml_key] = float(raw_sensors[backend_key])
        
        # Only call if we have all required sensors
        if len(ml_sensors) != len(ML_SENSOR_KEYS):
            log.warning("call_ml_predict: sensor count mismatch %d != %d. Keys: %s", len(ml_sensors), len(ML_SENSOR_KEYS), list(raw_sensors.keys()))
            return None
            
        payload = json.dumps({"sensors": ml_sensors, "t": t}).encode()
        req = urllib.request.Request(
            ML_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception as e:
        log.warning("ML prediction unavailable: %s", e)
    return None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("digital-twin")


# ===========================================================================
# FlightGear telnet client
# ===========================================================================
class TelnetError(Exception):
    """Raised when the telnet link fails mid-operation."""


class TelnetClient:
    """
    Minimal synchronous client for FlightGear's telnet property server.

    Protocol (normal/command mode):
      * send "<path>\\n"            -> replies "<path> = <value> (<unit>)>"
      * send "set <path> <value>\\n" -> writes the property
      * every reply is terminated by the "> " prompt, which we wait for.
    """

    def __init__(self, host: str = TELNET_HOST, port: int = TELNET_PORT,
                 connect_timeout: float = 4.0) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.sock: socket.socket | None = None
        self.buf = b""

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout)
        self.sock.settimeout(2.0)
        self.buf = b""
        self._drain_until_prompt(3.0)          # swallow greeting banner
        log.info("Telnet connected to %s:%d", self.host, self.port)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # -- low-level IO --------------------------------------------------------
    def _recv_more(self) -> bool:
        assert self.sock is not None
        try:
            chunk = self.sock.recv(4096)
        except socket.timeout:
            return False
        if not chunk:
            raise TelnetError("telnet connection closed by FlightGear")
        self.buf += chunk
        return True

    def _drain_until_prompt(self, timeout: float = 2.0) -> str:
        """Read from the socket until '>' appears or the deadline expires."""
        deadline = time.time() + timeout
        while b">" not in self.buf:
            if time.time() >= deadline:
                break
            if not self._recv_more():
                break
        head, sep, tail = self.buf.partition(b">")
        self.buf = tail
        return head.decode("utf-8", errors="replace")

    def command(self, line: str) -> str:
        """Send one command line, return the textual reply."""
        if self.sock is None:
            raise TelnetError("not connected")
        try:
            self.sock.sendall((line + "\n").encode("utf-8"))
        except OSError as exc:
            raise TelnetError(f"send failed: {exc}") from exc
        return self._drain_until_prompt(2.0)

    # -- high-level helpers ---------------------------------------------------
    def get_number(self, path: str) -> float | None:
        """Read one numeric property; None if unknown/non-numeric."""
        try:
            reply = self.command(path)
        except TelnetError as exc:
            log.debug("telnet get failed on %s: %s", path, exc)
            raise
        m = VALUE_RE.search(reply)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def set_property(self, path: str, value) -> bool:
        reply = self.command(f"set {path} {value}")
        return "Unknown" not in reply


def resolve_engine_prefix(tc: TelnetClient) -> str | None:
    """
    FlightGear versions differ: some expose '/engines/engine', newer ones
    '/engines/engine[0]'. Probe both once at startup.
    """
    for prefix in ("/engines/engine[0]", "/engines/engine"):
        try:
            if tc.get_number(prefix + "/rpm") is not None:
                return prefix
        except TelnetError:
            continue
    return None


# ===========================================================================
# Fault injection (simulated engine distress via property overrides)
# ===========================================================================
# Each fault defines target-property overrides as functions of elapsed time
# since activation. Because the FlightGear flight model re-writes these
# properties every frame, we simply re-write the forced value once per poll
# cycle -- immediately before reading -- which effectively pins it.
#
# Target keys are either ENGINE-relative suffixes ("egt-degf") or ABSOLUTE
# paths starting with "/" (e.g. control levers). Absolute control writes let
# FlightGear's own physics generate the degraded readings; in simulation
# mode SimulatedEngine emulates their effect on the sensors.
FAULT_LIBRARY = {
    "egt": {
        "desc": "hot-section distress: EGT ramps toward 1850 F",
        "targets": {
            "egt-degf": lambda t, s: min(1350 + s * 500, 1850),
            "oil-temperature-degf": lambda t, s: min(185 + s * 20, 250),
        },
    },
    "oil": {
        "desc": "oil leak: pressure decays toward 10 psi",
        "targets": {
            "oil-pressure-psi": lambda t, s: max(62 - s * 52, 10),
            "oil-temperature-degf": lambda t, s: min(185 + s * 25, 250),
        },
    },
    "mp": {
        "desc": "power loss: manifold pressure decays toward 12 inHg",
        "targets": {
            "mp-osi": lambda t, s: max(28 - s * 16, 12),
        },
    },
    "oiltemp": {
        "desc": "oil overheat: temperature ramps toward 250 F",
        "targets": {
            "oil-temperature-degf": lambda t, s: min(185 + s * 65, 250),
        },
    },
    "cht": {
        "desc": "cylinder overheat: CHT ramps toward 530 F",
        "targets": {
            "cht-degf": lambda t, s: min(380 + s * 150, 530),
            "oil-temperature-degf": lambda t, s: min(185 + s * 18, 250),
        },
    },
    "ff": {
        "desc": "fuel starvation: flow collapses toward 2 gph",
        "targets": {
            "fuel-flow-gph": lambda t, s: max(8.8 - s * 6.8, 2.0),
        },
    },
    "mixture": {
        "desc": "over-lean mixture: EGT soars, power fades "
                "(real control lever)",
        "targets": {"/controls/engines/engine[0]/mixture":
                    lambda t, s: round(max(1.0 - s * 0.70, 0.30), 3)},
    },
    "multi": {
        "desc": "multiple failure: EGT + oil leak + oil overheat",
        "targets": {
            "egt-degf": lambda t, s: min(1350 + s * 500, 1850),
            "oil-pressure-psi": lambda t, s: max(62 - s * 52, 10),
            "oil-temperature-degf": lambda t, s: min(185 + s * 55, 250),
        },
    },
}


class FaultManager:
    """Tracks which faults are active and applies their overrides."""

    def __init__(self) -> None:
        self.active: dict[str, tuple[float, float]] = {}
        # name -> (activation_timestamp, severity 0..1)
        self.manual_overrides: dict[str, float] = {}

    def activate(self, name: str, severity: float = 1.0) -> str | None:
        if name not in FAULT_LIBRARY:
            return None
        severity = clamp(severity, 0.0, 1.0)
        self.active[name] = (time.time(), severity)
        return FAULT_LIBRARY[name]["desc"]

    def deactivate(self, name: str) -> None:
        self.active.pop(name, None)

    def clear(self) -> int:
        n = len(self.active) + len(self.manual_overrides)
        self.active.clear()
        self.manual_overrides.clear()
        return n

    def get_durations(self) -> dict[str, float]:
        """Return elapsed seconds for each active fault."""
        return {n: round(now - t0, 1) for n, (t0, _s) in self.active.items()}

    def apply(self, tc: TelnetClient | None, prefix: str | None,
              sim_overrides: dict) -> None:
        """
        Force all active fault values (works in both FG & sim mode).
        Target keys starting with '/' are ABSOLUTE property paths (control
        levers); everything else is relative to the engine prefix.
        """
        for name, (t0, severity) in list(self.active.items()):
            elapsed = now - t0 if False else 0.0  # severity-only (no ramp)
            for target, fn in FAULT_LIBRARY[name]["targets"].items():
                value = round(fn(elapsed, severity), 3)
                sim_overrides[target] = value
                if tc is not None and prefix is not None:
                    path = target if target.startswith("/") \
                        else f"{prefix}/{target}"
                    try:
                        tc.set_property(path, value)
                    except TelnetError as exc:
                        log.warning("fault write failed (%s): %s", name, exc)
        for target, val in self.manual_overrides.items():
            sim_overrides[target] = val

    def summary(self) -> list[dict]:
        """Return list of {name, severity, duration_s} for the snapshot."""
        now = time.time()
        result = []
        for name, (t0, severity) in sorted(self.active.items()):
            result.append({
                "name": name,
                "severity": round(severity * 100),
                "duration_s": round(now - t0, 1),
            })
        return result


# ===========================================================================
# Built-in simulated engine (used only when FlightGear is unreachable)
# ===========================================================================
class SimulatedEngine:
    """Generates plausible C172P cruise values with gentle wander + noise."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.running = True

    def read(self, overrides: dict) -> dict[str, float]:
        t = time.time() - self.t0
        if not self.running:
            return {k: 0.0 for k in SENSOR_KEYS}
        rpm = 2320 + 140 * math.sin(0.05 * t) + np.random.normal(0, 25)
        egt = 1290 + 30 * math.sin(0.033 * t) + np.random.normal(0, 9)
        cht = 372 + 14 * math.sin(0.021 * t) + np.random.normal(0, 5)
        oil_temp = 183 + 4 * math.sin(0.008 * t) + np.random.normal(0, 2)
        oil_psi = 61 + 1.5 * math.sin(0.04 * t) + np.random.normal(0, 1.2)
        ff = 8.8 + (rpm - 2320) / 140.0 * 0.5 + np.random.normal(0, 0.15)
        mp = 27.3 + 0.4 * math.sin(0.05 * t) + np.random.normal(0, 0.1)
        vals = {
            "rpm": rpm, "egt_degf": egt, "cht_degf": cht,
            "oil_temp_f": oil_temp, "oil_psi": oil_psi,
            "ff_gph": ff, "mp_inhg": mp,
        }
        # Apply any active fault overrides on top of the healthy baseline.
        # Mixture-lever fault: emulate the physics FlightGear would produce
        # when the lever is pulled past peak EGT -- hotter combustion,
        # fading power, less fuel burned.
        mix = overrides.get("/controls/engines/engine[0]/mixture")
        if mix is not None:
            lean = max(0.0, min(1.0, 1.0 - float(mix)))    # 0 rich .. 0.7
            vals["egt_degf"] += lean * 480.0               # combustion soars
            vals["rpm"] *= 1.0 - 0.45 * lean               # power fades
            vals["mp_inhg"] *= 1.0 - 0.30 * lean
            vals["ff_gph"] *= 1.0 - 0.70 * lean            # fuel burn drops

        remap = {"egt-degf": "egt_degf", "oil-pressure-psi": "oil_psi",
                 "mp-osi": "mp_inhg", "oil-temperature-degf": "oil_temp_f",
                 "cht-degf": "cht_degf", "fuel-flow-gph": "ff_gph"}
        for suffix, v in overrides.items():
            key = remap.get(suffix)
            if key:
                vals[key] = v
        return vals


# ===========================================================================
# Anomaly detection
# ===========================================================================
class AnomalyDetector:
    """
    Primary path : TensorFlow autoencoder trained on healthy C-MAPSS data;
                   score = mean squared reconstruction error of the
                   min-max normalized 7-D sensor vector.
    Fallback     : online statistical detector -- squared deviation from an
                   exponentially-adapting nominal profile (no TF required).
    Threshold    : saved ae_threshold.json > online calibration > default.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model = None
        self.saved_threshold: float | None = None
        self.mu: np.ndarray | None = None          # fallback nominal profile
        self.calib: deque[float] = deque(maxlen=45)
        self.calibrated_thr: float | None = None

        # Load saved threshold produced by train_models.py (if present).
        thr_file = model_dir / "scaler.json"
        if thr_file.exists():
            try:
                meta = json.loads(thr_file.read_text())
                self.saved_threshold = float(meta.get("ae_threshold"))
                log.info("Loaded saved AE threshold: %.5f",
                         self.saved_threshold)
            except Exception as exc:               # noqa: BLE001
                log.warning("Could not parse scaler.json: %s", exc)

        # Load the trained autoencoder (TensorFlow only).
        ae_file = model_dir / "autoencoder.h5"
        if ae_file.exists():
            if not TF_AVAILABLE:
                log.warning("autoencoder.h5 found but TensorFlow is not "
                            "installed (%s). Using fallback detector.",
                            _TF_IMPORT_ERROR)
            else:
                try:
                    # compile=False: inference-only load. Avoids Keras 3
                    # metric/optimizer deserialization quirks with .h5.
                    self.model = tf.keras.models.load_model(ae_file,
                                                            compile=False)
                    n_in = self.model.input_shape[-1]
                    if n_in != len(SENSOR_DEFS):
                        log.warning("AE expects %d features but backend "
                                    "provides %d -> disabling model.",
                                    n_in, len(SENSOR_DEFS))
                        self.model = None
                    else:
                        log.info("Autoencoder loaded from %s", ae_file)
                except Exception as exc:           # noqa: BLE001
                    log.warning("Failed to load autoencoder.h5: %s", exc)

        if self.model is None and not ae_file.exists():
            log.info("No trained autoencoder found (%s). Using online "
                     "statistical fallback detector.", ae_file)

    # ------------------------------------------------------------------
    def effective_threshold(self) -> float:
        """
        Threshold policy: take the WIDEST of the available references --
        the AE's training-domain threshold and the online calibration from
        live data. This handles the C-MAPSS -> FlightGear domain gap, where
        baseline reconstruction error can exceed the training-domain value.
        """
        cands = []
        if self.model is not None and self.saved_threshold:
            cands.append(self.saved_threshold)
        if self.calibrated_thr:
            cands.append(self.calibrated_thr)
        return max(cands) if cands else DEFAULT_THRESHOLD

    def recalibrate(self) -> None:
        self.calib.clear()
        self.calibrated_thr = None
        self.mu = None
        log.info("Detector calibration reset")

    def _score(self, x01: np.ndarray) -> float:
        x = x01.reshape(1, -1)
        if self.model is not None:
            recon = self.model.predict(x, verbose=0)
            return float(np.mean((x - recon) ** 2))
        # Fallback: energy of deviation from slowly-adapting nominal
        # profile. SUM (not mean) of squared deviations gives better
        # contrast when a single sensor is driven far out of range.
        return float(np.sum((x01 - self.mu) ** 2))

    def process(self, x01: np.ndarray, engine_on: bool,
                faults_active: bool) -> tuple[float, float, bool, str]:
        """
        Returns (score, threshold, is_anomaly, status).
        Also performs online calibration while conditions are nominal.
        """
        if not engine_on:
            return 0.0, self.effective_threshold(), False, "STANDBY"

        if self.mu is None:                    # first live sample
            self.mu = x01.copy()
        score = self._score(x01)

        # Adapt the nominal input profile ONLY during healthy operation,
        # regardless of whether the autoencoder path is active.
        nominal_now = (not faults_active) and \
            float(np.max(np.abs(x01 - self.mu))) <= GROSS_DEV_LIMIT
        if nominal_now:
            self.mu = (1.0 - MU_ALPHA) * self.mu + MU_ALPHA * x01

        # Online calibration of the fallback statistic (also provides a
        # sanity reference when the AE is active).
        if self.calibrated_thr is None and not faults_active:
            self.calib.append(score)
            if len(self.calib) == self.calib.maxlen:
                arr = np.asarray(self.calib)
                thr = max(float(arr.mean() + 4.0 * arr.std()), 0.001)
                self.calibrated_thr = thr
                log.info("Online calibration done: threshold=%.5f", thr)

        thr = self.effective_threshold()
        # DUAL-PATH detection:
        #   1. detector statistic above threshold -- only trusted once the
        #      online calibration has seen ~45 s of live data (a threshold
        #      learned in the training domain may not transfer directly),
        #   2. any single sensor grossly outside its adapted nominal band,
        #      active from the very first tick.
        score_flag = (self.calibrated_thr is not None) and (score > thr)
        gross_dev = float(np.max(np.abs(x01 - self.mu))) > GROSS_DEV_LIMIT
        is_anom = score_flag or gross_dev

        # Status uses the SAME gating as is_anom so that a high raw score
        # in a not-yet-transferred domain cannot read as CRITICAL while
        # calibration is still running.
        if not is_anom:
            status = "NOMINAL"
        elif score > 3.0 * thr:
            status = "CRITICAL"
        else:
            status = "WARNING"
        return score, thr, is_anom, status


# ===========================================================================
# Remaining Useful Life estimation
# ===========================================================================
class RulEstimator:
    """
    * LSTM branch : sliding window (WINDOW_SIZE samples) of normalized
      sensors -> predicted RUL in hours (trained on C-MAPSS FD001 where
      1 cycle is treated as 1 hour for this demonstration).
    * Synthetic branch : health index h starts at 1.0 and decays with
      BASELINE_AGING_PER_SEC plus DAMAGE_GAIN per second for every unit of
      anomaly-score excess above the threshold -> RUL_synth = MAX_HOURS * h.

    Fusion: report the most conservative estimate, min(lstm, synth).
    """

    def __init__(self, model_dir: Path, window_size: int = WINDOW_SIZE) -> None:
        self.window = deque(maxlen=window_size)
        self.health = 1.0
        self.lstm = None
        self.last_lstm_rul: float | None = None
        self.label_scale = 1.0     # multiply raw LSTM output to get hours
        self.bias: float | None = None   # domain-transfer offset (hours)
        self._anchor_countdown = 45      # ticks before anchoring the bias

        # Training stores RUL labels scaled to [0..1]; scaler.json records
        # the factor so inference can convert back to hours.
        meta_file = model_dir / "scaler.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                self.label_scale = float(meta.get("label_scale", 1.0))
            except Exception as exc:               # noqa: BLE001
                log.warning("Could not parse scaler.json: %s", exc)

        lstm_file = model_dir / "lstm_rul.h5"
        if lstm_file.exists():
            if not TF_AVAILABLE:
                log.warning("lstm_rul.h5 found but TensorFlow is missing; "
                            "RUL will use the synthetic model only.")
            else:
                try:
                    self.lstm = tf.keras.models.load_model(lstm_file,
                                                           compile=False)
                    log.info("LSTM RUL model loaded from %s", lstm_file)
                except Exception as exc:           # noqa: BLE001
                    log.warning("Failed to load lstm_rul.h5: %s", exc)
        else:
            log.info("No LSTM model found (%s); synthetic RUL only.",
                     lstm_file)

    def reset_health(self) -> None:
        self.health = 1.0
        self.bias = None                     # re-anchor on next calibration
        self._anchor_countdown = 45
        log.info("Synthetic health reset to 1.0 (LSTM bias re-anchoring)")

    def step(self, dt: float, score: float, thr: float,
             engine_on: bool, x01: np.ndarray,
             anomaly_active: bool = False) -> tuple[float, float | None]:
        """Advance one poll cycle; returns (synth_rul, lstm_rul_or_None)."""
        if not engine_on:
            return RUL_MAX_HOURS * self.health, None

        # Excess damage only accrues when the detector has actually
        # declared an anomaly. This matters pre-calibration: the raw
        # score/threshold pair may not have transferred to the live domain
        # yet, and un-gated excess would drain health on a healthy engine.
        excess = max(0.0, score - thr) if anomaly_active else 0.0
        damage = dt * (BASELINE_AGING_PER_SEC
                       + (FAULT_DECAY_PER_SEC if anomaly_active else 0.0)
                       + DAMAGE_GAIN * excess)
        self.health = max(0.0, self.health - damage)

        synth_rul = RUL_MAX_HOURS * self.health

        lstm_rul = None
        if self.lstm is not None:
            self.window.append(x01.tolist())
            if len(self.window) == self.window.maxlen:
                try:
                    win = np.asarray(list(self.window),
                                     dtype=np.float32).reshape(
                        1, self.window.maxlen, -1)
                    raw = float(self.lstm.predict(win, verbose=0)[0][0])
                    raw_hours = raw * self.label_scale
                    if not math.isfinite(raw_hours):
                        raise ValueError("non-finite LSTM output")
                    # Domain transfer: the model was trained in the
                    # C-MAPSS domain, so its ABSOLUTE output on live
                    # FlightGear data is not meaningful. Anchor it once to
                    # the synthetic health model after a short nominal
                    # settling period, then keep its degradation TREND:
                    #   rul_lstm(t) = raw(t) + [synth(t0) - raw(t0)]
                    # Anomalous ticks PAUSE the countdown (they neither
                    # advance nor reset it) so intermittent transients
                    # cannot starve anchoring forever.
                    if self.bias is None:
                        if not anomaly_active and self.health >= 0.95:
                            self._anchor_countdown -= 1
                        log.debug("anchor countdown=%d anomaly_active=%s "
                                  "raw_hours=%.1f", self._anchor_countdown,
                                  anomaly_active, raw_hours)
                        if self._anchor_countdown <= 0:
                            self.bias = synth_rul - raw_hours
                            log.info("LSTM anchored to synthetic model "
                                     "(bias=%.1f h)", self.bias)
                    else:
                        lstm_rul = float(np.clip(raw_hours + self.bias,
                                                 0.0, RUL_MAX_HOURS))
                except Exception as exc:           # noqa: BLE001
                    log.warning("LSTM predict failed: %s", exc)
        self.last_lstm_rul = lstm_rul
        return synth_rul, lstm_rul


# ===========================================================================
# WebSocket hub
# ===========================================================================
class DashboardHub:
    """Keeps the set of connected dashboards and broadcasts snapshots."""

    def __init__(self) -> None:
        self.clients: set = set()
        self.latest: dict | None = None
        self._cmd_queue: asyncio.Queue | None = None

    def set_command_queue(self, q: asyncio.Queue) -> None:
        self._cmd_queue = q

    async def handler(self, ws, path=None):          # noqa: D401, ANN001
        """One connection: push latest snapshot, then handle commands."""
        self.clients.add(ws)
        log.info("Dashboard connected (%d total)", len(self.clients))
        try:
            if self.latest is not None:
                await ws.send(json.dumps(self.latest))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if self._cmd_queue is not None:
                    await self._cmd_queue.put(msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("Dashboard disconnected (%d total)", len(self.clients))

    async def publish(self, snapshot: dict) -> None:
        self.latest = snapshot
        if not self.clients:
            return
        msg = json.dumps(snapshot)
        results = await asyncio.gather(
            *(c.send(msg) for c in list(self.clients)),
            return_exceptions=True,
        )
        dead = {c for c, r in zip(list(self.clients), results)
                if isinstance(r, Exception)}
        for c in dead:
            self.clients.discard(c)


# ===========================================================================
# Console command interpreter (runs in a background thread reading stdin)
# ===========================================================================
def stdin_command_loop(loop: asyncio.AbstractEventQueue,
                       queue: "asyncio.Queue[str]") -> None:
    for raw in sys.stdin:
        cmd = raw.strip().lower()
        if cmd:
            loop.call_soon_threadsafe(queue.put_nowait, cmd)
        if cmd == "quit":
            break


async def handle_ws_command(msg: dict, fm: FaultManager,
                            detector: AnomalyDetector,
                            rul: RulEstimator) -> None:
    """Handle a JSON command received from the browser dashboard."""
    
    def reset_ml_server():
        try:
            req = urllib.request.Request(
                "http://localhost:5000/reset", method="POST"
            )
            urllib.request.urlopen(req, timeout=1.0)
        except Exception as e:
            log.debug("Failed to reset ML server: %s", e)

    cmd = msg.get("cmd", "")
    if cmd == "fault":
        fault_name = msg.get("type", "")
        severity = float(msg.get("severity", 100)) / 100.0
        desc = fm.activate(fault_name, severity)
        if desc:
            log.info("WS FAULT -> %s (%.0f%%) %s",
                     fault_name, severity * 100, desc)
        else:
            log.warning("WS unknown fault '%s'", fault_name)
    elif cmd == "clear":
        n = fm.clear()
        log.info("WS FAULTS CLEARED (%d active)", n)
        asyncio.get_running_loop().run_in_executor(None, reset_ml_server)
    elif cmd == "mitigate":
        n = fm.clear()
        rul.health = min(rul.health + 0.15, 1.0) # Boost synthetic health slightly
        log.info("WS MITIGATE EXECUTED. Faults cleared. Health boosted.")
        asyncio.get_running_loop().run_in_executor(None, reset_ml_server)
    elif cmd == "heal":
        detector.recalibrate()
        rul.reset_health()
        fm.clear()
        asyncio.get_running_loop().run_in_executor(None, reset_ml_server)
        log.info("WS HEAL: health restored, faults cleared")
    elif cmd == "severity":
        name = msg.get("type", "")
        sev = float(msg.get("severity", 100)) / 100.0
        if name in fm.active:
            t0 = fm.active[name][0]
            fm.active[name] = (t0, sev)
    elif cmd == "override":
        sensor = msg.get("sensor")
        val = msg.get("value")
        if sensor:
            if val is None or val == "":
                fm.manual_overrides.pop(sensor, None)
                log.info("WS OVERRIDE CLEARED: %s", sensor)
            else:
                fm.manual_overrides[sensor] = float(val)
                log.info("WS OVERRIDE SET: %s = %.1f", sensor, float(val))
    else:
        log.debug("WS unknown command: %s", cmd)


async def handle_command(cmd: str, fm: FaultManager, sim: SimulatedEngine | None,
                         tc: TelnetClient | None, prefix: str | None,
                         detector: AnomalyDetector,
                         rul: RulEstimator) -> bool:
    """Execute one console command. Returns False to request shutdown."""
    parts = cmd.split()
    verb = parts[0] if parts else ""

    if verb == "quit":
        return False
    if verb == "help":
        log.info("Commands: start | stop | fault <name> [severity] | "
                 "multi [severity] | clear | heal | status | quit")
    elif verb == "start":
        if sim is not None:
            sim.running = True
            log.info("[SIM] engine started")
        elif tc and prefix:
            for p, v in (("mixture", 1.0), ("throttle", 0.25)):
                tc.set_property(f"/controls/engines/engine[0]/{p}", v)
            ok = tc.set_property(prefix + "/running", True)
            log.info("Engine start command sent (running=%s)", ok)
    elif verb == "stop":
        if sim is not None:
            sim.running = False
            log.info("[SIM] engine stopped")
        elif tc and prefix:
            tc.set_property(prefix + "/running", False)
            log.info("Engine stop command sent")
    elif verb == "fault":
        fault_name = parts[1] if len(parts) > 1 else ""
        severity = float(parts[2]) / 100.0 if len(parts) > 2 else 1.0
        desc = fm.activate(fault_name, severity)
        if desc:
            log.info("FAULT INJECTED -> %s (severity %.0f%%)",
                     desc, severity * 100)
        else:
            log.warning("Unknown fault '%s' (use: egt, oil, mp, oiltemp, "
                        "cht, ff, mixture, multi)", fault_name or "?")
    elif verb == "multi":
        severity = float(parts[1]) / 100.0 if len(parts) > 1 else 1.0
        desc = fm.activate("multi", severity)
        if desc:
            log.info("MULTI-FAULT INJECTED -> %s (severity %.0f%%)",
                     desc, severity * 100)
    elif verb == "clear":
        n = fm.clear()
        log.info("Faults cleared (%d active before)", n)
    elif verb == "heal":
        detector.recalibrate()
        rul.reset_health()
        fm.clear()
        log.info("Health restored, faults cleared, detector recalibrating")
    elif verb == "status":
        snap = hub_latest(globals())
        if snap:
            log.info("status=%s score=%.4f/%.4f health=%.2f rul=%.1fh "
                     "faults=%s", snap["status"], snap["anomaly_score"],
                     snap["threshold"], snap["health"], snap["rul_h"],
                     snap.get("faults") or "-")
        else:
            log.info("No telemetry yet")
    else:
        log.warning("Unknown command '%s' (try 'help')", cmd)
    return True


def hub_latest(ns: dict) -> dict | None:
    hub = ns.get("_hub_ref")
    return hub.latest if hub else None


# ===========================================================================
# Per-tick processing (executed in a worker thread via run_in_executor)
# ===========================================================================
class TickContext:
    """Bundle of mutable state shared across poll cycles."""

    def __init__(self) -> None:
        self.tc: TelnetClient | None = None
        self.prefix: str | None = None
        self.sim: SimulatedEngine | None = None
        self.fm = FaultManager()
        self.detector = AnomalyDetector(MODEL_DIR)
        self.rul = RulEstimator(MODEL_DIR)
        self.last_values: dict[str, float] = {}
        self.overrides: dict[str, float] = {}


def process_tick(ctx: TickContext, dt: float) -> dict:
    """Read sensors, run ML, build the JSON snapshot. May raise TelnetError."""
    engine_on_prev = ctx.last_values.get("rpm", 0.0) >= ENGINE_ON_RPM

    # 1) apply active fault overrides (before reading, so reads reflect them)
    ctx.overrides.clear()
    ctx.fm.apply(ctx.tc, ctx.prefix, ctx.overrides)

    # 2) read sensors
    raw: dict[str, float] = {}
    if ctx.sim is not None:
        raw = ctx.sim.read(ctx.overrides)
    else:
        assert ctx.tc is not None and ctx.prefix is not None
        for key, suffix, lo, hi in SENSOR_DEFS:
            try:
                val = ctx.tc.get_number(ctx.prefix + "/" + suffix)
            except TelnetError:
                val = None
            if val is None:                       # hold last good value
                val = ctx.last_values.get(key, 0.0)
            raw[key] = val

    engine_on = raw.get("rpm", 0.0) >= ENGINE_ON_RPM
    ctx.last_values.update(raw)

    # 3) normalize into the training envelope [0..1]
    x01 = np.array([
        float(np.clip((raw[k] - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
        for k, _s, lo, hi in SENSOR_DEFS
    ], dtype=np.float32)

    # 4) anomaly detection + RUL update (skipped while engine is off)
    if engine_on:
        score, thr, is_anom, status = ctx.detector.process(
            x01, True, bool(ctx.fm.active))
        synth_rul, lstm_rul = ctx.rul.step(dt, score, thr, True, x01,
                                           anomaly_active=is_anom)
    else:
        score, thr, is_anom = 0.0, ctx.detector.effective_threshold(), False
        status = "STANDBY"
        synth_rul, lstm_rul = ctx.rul.step(dt, score, thr, False, x01)

    fused_rul = synth_rul
    if lstm_rul is not None:
        fused_rul = min(fused_rul, lstm_rul)      # conservative fusion

    # 5) compute per-sensor explainability (deviation from nominal)
    explain = {}
    for i, (k, _s, _lo, _hi) in enumerate(SENSOR_DEFS):
        dev = abs(float(x01[i]) - float(ctx.detector.mu[i])) \
            if ctx.detector.mu is not None else 0.0
        explain[k] = round(dev, 4)

    # 6) assemble snapshot for the dashboard
    return {
        "timestamp": round(time.time(), 3),
        "mode": "SIMULATION" if ctx.sim is not None else "FLIGHTGEAR",
        "engine_running": bool(engine_on),
        "sensors": {k: round(float(raw[k]), 1) for k in SENSOR_KEYS},
        "normalized": [round(float(v), 4) for v in x01],
        "anomaly_score": round(score, 5),
        "threshold": round(thr, 5),
        "is_anomaly": bool(is_anom),
        "status": status,
        "health": round(ctx.rul.health, 4),
        "rul_lstm_h": None if lstm_rul is None else round(lstm_rul, 1),
        "rul_synth_h": round(synth_rul, 1),
        "rul_h": round(fused_rul, 1),
        "faults": ctx.fm.summary(),
        "explain": explain,
    }


# ===========================================================================
# SQLite persistent storage (lightweight, synchronous writes)
# ===========================================================================
try:
    import sqlite3 as _sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class DataStore:
    """Writes telemetry + events to a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._conn = None
        if not SQLITE_AVAILABLE:
            return
        try:
            self._conn = _sqlite3.connect(str(db_path), check_same_thread=False)
            c = self._conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS telemetry (
                ts REAL PRIMARY KEY,
                rpm REAL, egt REAL, cht REAL, oil_temp REAL,
                oil_press REAL, fuel_flow REAL, manifold REAL,
                anomaly_score REAL, health REAL, rul_h REAL, status TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS events (
                ts REAL, etype TEXT, detail TEXT
            )""")
            self._conn.commit()
            log.info("SQLite storage: %s", db_path)
        except Exception as exc:
            log.warning("SQLite init failed: %s", exc)
            self._conn = None

    def write_snapshot(self, snap: dict) -> None:
        if self._conn is None:
            return
        try:
            s = snap.get("sensors", {})
            c = self._conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (snap["timestamp"],
                 s.get("rpm"), s.get("egt_degf"), s.get("cht_degf"),
                 s.get("oil_temp_f"), s.get("oil_psi"),
                 s.get("ff_gph"), s.get("mp_inhg"),
                 snap.get("anomaly_score"), snap.get("health"),
                 snap.get("rul_h"), snap.get("status")),
            )
            self._conn.commit()
        except Exception:
            pass

    def write_event(self, ts: float, etype: str, detail: str) -> None:
        if self._conn is None:
            return
        try:
            c = self._conn.cursor()
            c.execute("INSERT INTO events VALUES (?,?,?)", (ts, etype, detail))
            self._conn.commit()
        except Exception:
            pass

    def close(self) -> None:
        if self._conn:
            self._conn.close()


# ===========================================================================
# Main asynchronous service
# ===========================================================================
async def amain(args: argparse.Namespace) -> None:
    ctx = TickContext()
    hub = DashboardHub()
    globals()["_hub_ref"] = hub
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[str]" = asyncio.Queue()
    ws_cmd_queue: asyncio.Queue = asyncio.Queue()
    hub.set_command_queue(ws_cmd_queue)

    # ---- persistent storage -----------------------------------------------
    store = DataStore(HERE / "twin_data.db")

    # ---- connect to FlightGear (or fall back to simulation) --------------
    if not args.simulation:
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                tc = TelnetClient(port=args.telnet_port)
                tc.connect()
                ctx.tc = tc
                break
            except OSError as exc:
                log.warning("FlightGear telnet not reachable (%s). "
                            "Retry %d/%d ...", exc, attempt, CONNECT_RETRIES)
                await asyncio.sleep(CONNECT_RETRY_DELAY_S)
        if ctx.tc is not None:
            ctx.prefix = resolve_engine_prefix(ctx.tc)
            if ctx.prefix is None:
                log.error("Connected to FlightGear but no engine RPM "
                          "property found; falling back to SIMULATION.")
                ctx.tc.close()
                ctx.tc = None
    else:
        log.info("--simulation flag set: skipping FlightGear connection.")

    if ctx.tc is None:
        ctx.sim = SimulatedEngine()
        log.info("=" * 66)
        log.info("  RUNNING IN SIMULATION MODE (no live FlightGear data)")
        log.info("  Launch FlightGear with --telnet=%d to use real data.",
                 args.telnet_port)
        log.info("=" * 66)

    # ---- console command thread ------------------------------------------
    threading.Thread(target=stdin_command_loop,
                     args=(loop, queue), daemon=True).start()

    # ---- WebSocket server --------------------------------------------------
    server = await websockets.serve(hub.handler, "0.0.0.0", args.ws_port)
    log.info("WebSocket server listening on ws://localhost:%d", args.ws_port)
    log.info("Type 'help' for console commands.")

    last_t = time.time()
    t0 = time.time()
    last_status = "STANDBY"
    try:
        while True:
            try:
                # drain pending console commands
                while not queue.empty():
                    cmd = queue.get_nowait()
                    cont = await handle_command(cmd, ctx.fm, ctx.sim, ctx.tc,
                                                ctx.prefix, ctx.detector,
                                                ctx.rul)
                    if not cont:
                        return

                # drain pending WS commands from the dashboard
                while not ws_cmd_queue.empty():
                    msg = ws_cmd_queue.get_nowait()
                    await handle_ws_command(msg, ctx.fm, ctx.detector, ctx.rul)

            except Exception as e:
                log.error("Error processing commands: %s", e)
            now = time.time()
            dt = float(np.clip(now - last_t, 0.5, 2.0))
            last_t = now

            try:
                snapshot = await loop.run_in_executor(None, process_tick,
                                                      ctx, dt)
            except TelnetError as exc:
                log.error("Lost FlightGear link (%s); reconnecting...", exc)
                try:
                    if ctx.tc:
                        ctx.tc.close()
                        ctx.tc.connect()
                        ctx.prefix = resolve_engine_prefix(ctx.tc)
                except OSError as exc2:
                    log.error("Reconnect failed: %s", exc2)
                await asyncio.sleep(2.0)
                continue

            # --- ML Prediction Integration ---
            try:
                ml_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        call_ml_predict,
                        snapshot.get("sensors", {}),
                        now - t0,
                    ),
                    timeout=3.0,
                )
                if ml_result:
                    snapshot["ml_fault"] = ml_result.get("fault", "unknown")
                    snapshot["ml_confidence"] = ml_result.get("confidence", 0)
                    snapshot["ml_probabilities"] = ml_result.get("probabilities", {})
                    snapshot["ml_rul_h"] = ml_result.get("rul_h")
                    snapshot["ml_window_filled"] = ml_result.get("window_filled", False)
                else:
                    snapshot["ml_fault"] = None
            except Exception as ml_exc:
                log.warning("ML prediction skipped: %s", ml_exc)
                snapshot["ml_fault"] = None

            # log status transitions
            if snapshot["status"] != last_status:
                store.write_event(now, "status_change",
                                  f"{last_status} -> {snapshot['status']}")
                last_status = snapshot["status"]

            # persist telemetry every tick
            store.write_snapshot(snapshot)

            await hub.publish(snapshot)
            await asyncio.sleep(max(0.0, POLL_INTERVAL_S -
                                    (time.time() - now)))
    except asyncio.CancelledError:
            raise
    except Exception as e:
            log.error("Error in main loop: %s", e, exc_info=True)
            await asyncio.sleep(1.0)
    finally:
        server.close()
        await server.wait_closed()
        store.close()
        if ctx.tc:
            ctx.tc.close()
        log.info("Backend stopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aircraft Engine Digital Twin "
                                         "backend")
    ap.add_argument("--ws-port", type=int, default=WS_PORT,
                help="WebSocket port for the dashboard (default 8765)")
    ap.add_argument("--telnet-port", type=int, default=TELNET_PORT,
                help="FlightGear telnet port (default 5500)")
    ap.add_argument("--simulation", action="store_true",
                help="skip FlightGear and use the internal simulator")
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as e:
        log.error("Fatal error in backend: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
