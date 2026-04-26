"""
emotion_inference.py — Realtime EEG emotion classification using dual XGBoost models.

Pipeline:
  1. Push 14-channel EEG samples via push_sample() or push_chunk()
  2. Calibration phase (~30s): collects baseline windows, computes per-user z-score stats
  3. Prediction phase: extract 84 DE+DASM features → calibration z-score → global scaler → XGBoost
  4. Both valence (pos/neg) and arousal (high/low) are predicted independently
  5. 4-quadrant mood + focused override → app emotion label → EmotionSmoother → broadcast

Quick start:
  # Synthetic board test
  python emotion_inference.py --board-id -1 --debug

  # Emotiv via LSL bridge
  python emotion_inference.py --board-id -11 --ip-address 127.0.0.1 --ip-port 6998

  # Custom model paths
  python emotion_inference.py \\
      --model-valence C:/Users/Cyberhell/eeg_emotion/models/valence_xgb.joblib \\
      --model-arousal C:/Users/Cyberhell/eeg_emotion/models/arousal_xgb.joblib \\
      --scaler        C:/Users/Cyberhell/eeg_emotion/artifacts/scaler.joblib

Feature layout (84 total, must match training order exactly):
  [DE_Theta x14, DE_Alpha x14, DE_Beta x14, DE_Gamma x14,
   DASM_Theta x7, DASM_Alpha x7, DASM_Beta x7, DASM_Gamma x7]

DASM left-right pairs (index into CHANNELS list):
  AF3(0)-AF4(13), F7(1)-F8(12), F3(2)-F4(11), FC5(3)-FC6(10),
  T7(4)-T8(9),    P7(5)-P8(8),  O1(6)-O2(7)
"""

import argparse
import collections
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import signal


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

FS = 128
WINDOW_SECONDS = 4
WINDOW_SIZE = FS * WINDOW_SECONDS  # 512 samples

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2",  "P8", "T8", "FC6", "F4", "F8", "AF4",
]
N_CHANNELS = len(CHANNELS)  # 14

# DASM pairs: (left_index, right_index) — must match training
DASM_PAIRS = [
    (0, 13),  # AF3 - AF4
    (1, 12),  # F7  - F8
    (2, 11),  # F3  - F4
    (3, 10),  # FC5 - FC6
    (4,  9),  # T7  - T8
    (5,  8),  # P7  - P8
    (6,  7),  # O1  - O2
]
N_PAIRS = len(DASM_PAIRS)  # 7

# Band order matters — must match training feature layout
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]
BANDS = {
    "theta": (4,  8),
    "alpha": (8,  12),
    "beta":  (12, 30),
    "gamma": (30, 45),
}

# Default artifact paths (Windows local, overridable via CLI or env)
DEFAULT_VALENCE_PATH = Path(r"C:\Users\Cyberhell\eeg_emotion\models\valence_xgb.joblib")
DEFAULT_AROUSAL_PATH = Path(r"C:\Users\Cyberhell\eeg_emotion\models\arousal_xgb.joblib")
DEFAULT_SCALER_PATH  = Path(r"C:\Users\Cyberhell\eeg_emotion\artifacts\scaler.joblib")

# Hysteresis thresholds for valence/arousal state
THRESHOLD_HIGH = 0.55   # prob must exceed this to enter "high" state
THRESHOLD_LOW  = 0.45   # prob must drop below this to enter "low" state
DEAD_ZONE      = 0.08   # |prob - 0.5| < this → "uncertain"

# 4-quadrant → app emotion mapping
QUADRANT_TO_EMOTION = {
    ("positive", "high"): "happy",
    ("positive", "low"):  "calm",
    ("negative", "high"): "angry",
    ("negative", "low"):  "sad",
}


# ═══════════════════════════════════════════════════════════════════════════
#  EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════

class MissingArtifactsError(RuntimeError):
    pass

class NotCalibratedError(RuntimeError):
    pass

class ChannelCountError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  ARTIFACT PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

def resolve_artifact_paths(
    valence_xgb_path=None,
    arousal_xgb_path=None,
    scaler_path=None,
):
    """
    Resolve model and scaler paths with the following priority:
      1. Explicit argument passed in
      2. Environment variable
      3. Known candidate directories
      4. Default Windows local path

    Raises MissingArtifactsError if any file cannot be found.
    """
    def _find(explicit, env_var, candidates, default):
        if explicit:
            return Path(explicit)
        env_val = os.getenv(env_var)
        if env_val:
            return Path(env_val)
        for p in candidates:
            if p.exists():
                return p
        return default  # may not exist — checked below

    base_dir     = Path(__file__).resolve().parent
    artifact_dir = os.getenv("EEG_EMOTION_ARTIFACT_DIR")

    search_roots = []
    if artifact_dir:
        search_roots.append(Path(artifact_dir))
    search_roots.extend([
        base_dir / "emotion_artifacts",
        base_dir.parent / "emotion_artifacts",
        Path.home() / "eeg_emotion",
    ])

    # Candidate model dirs: structured subdirs AND flat repo models/ folder
    model_candidates_v = [d / "models" / "valence_xgb.joblib" for d in search_roots]
    model_candidates_a = [d / "models" / "arousal_xgb.joblib" for d in search_roots]
    scaler_candidates  = [d / "artifacts" / "scaler.joblib"   for d in search_roots]

    # Also search repo-relative models/ and artifacts/ directly
    for repo_models in [base_dir / "models", base_dir.parent / "models"]:
        model_candidates_v.append(repo_models / "valence_xgb.joblib")
        model_candidates_a.append(repo_models / "arousal_xgb.joblib")
        scaler_candidates.append( repo_models / "scaler.joblib")

    for repo_artifacts in [base_dir / "artifacts", base_dir.parent / "artifacts"]:
        scaler_candidates.append(repo_artifacts / "scaler.joblib")

    resolved_valence = _find(
        valence_xgb_path,
        "EEG_VALENCE_MODEL_PATH",
        model_candidates_v,
        DEFAULT_VALENCE_PATH,
    )
    resolved_arousal = _find(
        arousal_xgb_path,
        "EEG_AROUSAL_MODEL_PATH",
        model_candidates_a,
        DEFAULT_AROUSAL_PATH,
    )
    resolved_scaler = _find(
        scaler_path,
        "EEG_SCALER_PATH",
        scaler_candidates,
        DEFAULT_SCALER_PATH,
    )

    missing = []
    for name, path in [
        ("valence_xgb.joblib", resolved_valence),
        ("arousal_xgb.joblib", resolved_arousal),
        ("scaler.joblib",      resolved_scaler),
    ]:
        if not path or not Path(path).exists():
            missing.append(name)

    if missing:
        raise MissingArtifactsError(
            "Missing model artifacts: " + ", ".join(missing) + ". "
            "Set EEG_EMOTION_ARTIFACT_DIR, or pass explicit paths via CLI / constructor."
        )

    print(f"[ARTIFACTS] valence → {resolved_valence}")
    print(f"[ARTIFACTS] arousal → {resolved_arousal}")
    print(f"[ARTIFACTS] scaler  → {resolved_scaler}")
    return str(resolved_valence), str(resolved_arousal), str(resolved_scaler)


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_band_filters(fs=FS):
    """Pre-compute 4th-order Butterworth bandpass filters for each band."""
    return {
        name: signal.butter(4, [low, high], btype="band", fs=fs)
        for name, (low, high) in BANDS.items()
    }


def extract_features(window, band_filters):
    """
    Extract 84 features from a (N_CHANNELS, WINDOW_SIZE) array.

    Returns:
        features       np.ndarray shape (84,)  — [DE x56, DASM x28]
        band_power_rel dict  — relative band power per band (for UI)
    """
    # window shape: (N_CHANNELS, WINDOW_SIZE)
    de_blocks   = []   # 4 blocks of 14 = 56
    dasm_blocks = []   # 4 blocks of 7  = 28
    raw_power   = {}

    for band_name in BAND_ORDER:
        b, a = band_filters[band_name]
        filtered = signal.filtfilt(b, a, window, axis=1)          # (14, 512)
        variance  = np.var(filtered, axis=1) + 1e-12              # (14,)
        raw_power[band_name] = float(np.mean(variance))

        de_band = 0.5 * np.log(2.0 * np.pi * np.e * variance)    # (14,)
        de_blocks.append(de_band)

        dasm_band = np.array(
            [de_band[l] - de_band[r] for l, r in DASM_PAIRS],
            dtype=np.float64,
        )  # (7,)
        dasm_blocks.append(dasm_band)

    features = np.concatenate(de_blocks + dasm_blocks)  # (84,)

    total = sum(raw_power.values()) or 1.0
    band_power_rel = {name: float(v / total) for name, v in raw_power.items()}

    return features, band_power_rel


# ═══════════════════════════════════════════════════════════════════════════
#  DUAL-AXIS RECOGNIZER
# ═══════════════════════════════════════════════════════════════════════════

class DualAxisRecognizer:
    """
    Buffers incoming EEG samples, extracts features, and runs
    both valence and arousal XGBoost models.

    Internal predict() return (consumed by EmotionSmoother):
      During calibration:
        {"type": "status", "status": "calibrating", "progress": float}
      After calibration:
        {"type": "dual_axis", "status": "predicting",
         "valence_prob": float, "arousal_prob": float, "features": dict}
    """

    def __init__(
        self,
        valence_xgb_path=None,
        arousal_xgb_path=None,
        scaler_path=None,
        calibration_seconds=30,
    ):
        v_path, a_path, s_path = resolve_artifact_paths(
            valence_xgb_path=valence_xgb_path,
            arousal_xgb_path=arousal_xgb_path,
            scaler_path=scaler_path,
        )
        self.valence_model = joblib.load(v_path)
        self.arousal_model = joblib.load(a_path)
        self.scaler        = joblib.load(s_path)

        self._band_filters = _build_band_filters(FS)

        # Rolling sample buffers — one deque per channel
        self._buffers = [
            collections.deque(maxlen=WINDOW_SIZE) for _ in range(N_CHANNELS)
        ]

        # Calibration state
        self._calib_target = max(8, int(calibration_seconds / WINDOW_SECONDS))
        self._calib_features: list = []
        self._calib_mean = None
        self._calib_std  = None

    # ── Public API ──────────────────────────────────────────────────────

    def push_sample(self, sample):
        """Append one 14-channel sample. Raises ChannelCountError on wrong size."""
        if len(sample) != N_CHANNELS:
            raise ChannelCountError(
                f"Expected {N_CHANNELS} channels, got {len(sample)}"
            )
        for i, v in enumerate(sample):
            self._buffers[i].append(float(v))

    def push_chunk(self, chunk):
        """
        Append a chunk of samples.
        chunk: array-like of shape (n_samples, 14) or (14, n_samples).
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ChannelCountError("chunk must be 2-dimensional")
        # Accept both (n, 14) and (14, n)
        if chunk.shape[0] == N_CHANNELS and chunk.shape[1] != N_CHANNELS:
            chunk = chunk.T
        if chunk.shape[1] != N_CHANNELS:
            raise ChannelCountError(
                f"Expected {N_CHANNELS} channels, got {chunk.shape[1]}"
            )
        for sample in chunk:
            self.push_sample(sample)

    def is_ready(self):
        return all(len(b) == WINDOW_SIZE for b in self._buffers)

    def is_calibrated(self):
        return self._calib_mean is not None

    def calibration_progress(self):
        if self.is_calibrated():
            return 1.0
        return min(len(self._calib_features) / self._calib_target, 1.0)

    def predict(self):
        """
        Returns a prediction dict, or None if the buffer isn't full yet.
        """
        if not self.is_ready():
            return None

        window   = np.array([list(b) for b in self._buffers], dtype=np.float64)
        features, band_rel = extract_features(window, self._band_filters)

        # ── Calibration phase ──────────────────────────────────────────
        if not self.is_calibrated():
            self._calib_features.append(features.copy())
            if len(self._calib_features) >= self._calib_target:
                arr = np.array(self._calib_features)
                self._calib_mean = arr.mean(axis=0)
                self._calib_std  = arr.std(axis=0) + 1e-8
            return {
                "type":     "status",
                "status":   "calibrating",
                "progress": round(self.calibration_progress(), 3),
            }

        # ── Prediction phase ───────────────────────────────────────────
        normalized = self._normalize(features)

        valence_prob = float(self.valence_model.predict_proba(normalized)[0, 1])
        arousal_prob = float(self.arousal_model.predict_proba(normalized)[0, 1])

        return {
            "type":        "dual_axis",
            "status":      "predicting",
            "valence_prob": valence_prob,
            "arousal_prob": arousal_prob,
            "features":    band_rel,
        }

    # ── Internal ────────────────────────────────────────────────────────

    def _normalize(self, features):
        """Apply per-user calibration z-score, then global scaler."""
        calibrated = (features - self._calib_mean) / self._calib_std
        return self.scaler.transform(calibrated.reshape(1, -1))


# ═══════════════════════════════════════════════════════════════════════════
#  EMOTION SMOOTHER  (same output contract as before — app.js unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class EmotionSmoother:
    """
    Converts raw dual-axis predictions into stable app-level emotion labels.

    Stability mechanisms:
      - EWMA smoothing on valence_prob and arousal_prob
      - Hysteresis thresholds (enter_high / enter_low) to avoid thrashing
      - Dead-zone: if either axis is uncertain, skip
      - Majority vote over rolling history
      - Cooldown between different emotion switches
      - Silence window before re-emitting the same emotion
    """

    APP_EMOTIONS = ("calm", "happy", "angry", "sad", "focused")

    def __init__(
        self,
        min_confidence=0.62,
        history_size=5,
        min_stable_votes=3,
        cooldown_seconds=10.0,
        duplicate_silence_seconds=3.0,
        ewma_alpha=0.2,
        enter_high=THRESHOLD_HIGH,
        enter_low=THRESHOLD_LOW,
        dead_zone=DEAD_ZONE,
    ):
        self.min_confidence            = min_confidence
        self.history                   = collections.deque(maxlen=history_size)
        self.min_stable_votes          = min_stable_votes
        self.cooldown_seconds          = cooldown_seconds
        self.duplicate_silence_seconds = duplicate_silence_seconds
        self.ewma_alpha                = ewma_alpha
        self.enter_high                = enter_high
        self.enter_low                 = enter_low
        self.dead_zone                 = dead_zone

        # EWMA state
        self._ewma_valence = None
        self._ewma_arousal = None

        # Hysteresis state ("high" / "low" / None)
        self._valence_state = None
        self._arousal_state = None

        # Emit tracking
        self.last_emitted_emotion = None
        self.last_emitted_at      = 0.0
        self.last_status          = None

    def process(self, prediction, timestamp):
        """
        Accepts a raw prediction dict from DualAxisRecognizer.predict().
        Returns a broadcast-ready emotion dict or None.
        """
        if not prediction:
            return None

        # Pass calibration status through unchanged
        if prediction.get("type") == "status":
            if prediction.get("status") == "calibrating":
                progress = round(prediction.get("progress", 0.0), 2)
                if progress != self.last_status:
                    self.last_status = progress
                    return {
                        "type":      "status",
                        "status":    "calibrating",
                        "progress":  progress,
                        "timestamp": timestamp,
                    }
            return None

        if prediction.get("type") != "dual_axis":
            return None

        # ── EWMA smoothing ─────────────────────────────────────────────
        vp = prediction["valence_prob"]
        ap = prediction["arousal_prob"]

        if self._ewma_valence is None:
            self._ewma_valence = vp
            self._ewma_arousal = ap
        else:
            a = self.ewma_alpha
            self._ewma_valence = a * vp + (1 - a) * self._ewma_valence
            self._ewma_arousal = a * ap + (1 - a) * self._ewma_arousal

        sv = self._ewma_valence
        sa = self._ewma_arousal

        # ── Hysteresis state update ────────────────────────────────────
        self._valence_state = self._update_state(self._valence_state, sv)
        self._arousal_state = self._update_state(self._arousal_state, sa)

        # ── Dead-zone: skip if either axis is ambiguous ────────────────
        if self._valence_state is None or self._arousal_state is None:
            return None

        # ── Map to emotion ─────────────────────────────────────────────
        candidate = self._map_to_emotion(sv, sa, prediction["features"])
        candidate["timestamp"] = timestamp
        self.history.append(candidate)

        # ── Majority vote ──────────────────────────────────────────────
        recent = [
            item for item in self.history
            if item["confidence"] >= self.min_confidence
        ]
        if not recent:
            return None

        counts = collections.Counter(item["emotion"] for item in recent)
        winner, votes = counts.most_common(1)[0]
        if votes < self.min_stable_votes:
            return None

        # ── Cooldown / duplicate silence ───────────────────────────────
        now = timestamp
        if winner == self.last_emitted_emotion:
            if now - self.last_emitted_at < self.duplicate_silence_seconds:
                return None
        elif now - self.last_emitted_at < self.cooldown_seconds:
            return None

        # ── Build output (same shape app.js expects) ───────────────────
        winner_items   = [i for i in recent if i["emotion"] == winner]
        avg_confidence = float(np.mean([i["confidence"] for i in winner_items]))
        latest         = winner_items[-1]

        self.last_emitted_emotion = winner
        self.last_emitted_at      = now

        return {
            "type":       "emotion",
            "emotion":    winner,
            "confidence": round(avg_confidence, 3),
            "timestamp":  timestamp,
            "features": {
                name: round(float(v), 3)
                for name, v in latest["features"].items()
            },
            # Extra fields for debugging — app.js ignores unknown keys
            "valence_prob":  round(sv, 3),
            "arousal_prob":  round(sa, 3),
            "valence_label": self._valence_state,
            "arousal_label": self._arousal_state,
            "mood_quadrant": QUADRANT_TO_EMOTION.get(
                (self._valence_state, self._arousal_state), winner
            ),
            "source": "eeg_model",
        }

    # ── Internal ────────────────────────────────────────────────────────

    def _update_state(self, current_state, prob):
        """
        Hysteresis: only change state when prob crosses a threshold.
        Returns "high", "low", or None (dead-zone).
        """
        if abs(prob - 0.5) < self.dead_zone:
            return None  # uncertain
        if current_state == "high":
            return "low" if prob < self.enter_low else "high"
        if current_state == "low":
            return "high" if prob > self.enter_high else "low"
        # First reading — pick based on threshold
        if prob > self.enter_high:
            return "high"
        if prob < self.enter_low:
            return "low"
        return None

    def _map_to_emotion(self, valence_prob, arousal_prob, features):
        """
        Convert smoothed dual-axis probabilities → app emotion label.

        Priority:
          1. "focused" override: high beta dominance regardless of quadrant
          2. 4-quadrant QUADRANT_TO_EMOTION lookup
        """
        alpha = float(features.get("alpha", 0.0))
        beta  = float(features.get("beta",  0.0))
        theta = float(features.get("theta", 0.0))

        focus_ratio = beta / (alpha + theta + 1e-6)
        is_focused  = (
            valence_prob >= 0.45        # not strongly negative
            and focus_ratio >= 0.34
            and beta >= theta * 0.9
        )

        # Confidence: distance of both probs from 0.5, averaged
        valence_conf = 0.5 + abs(valence_prob - 0.5)
        arousal_conf = 0.5 + abs(arousal_prob - 0.5)
        confidence   = min(0.99, (valence_conf + arousal_conf) / 2.0)

        if is_focused:
            emotion = "focused"
        else:
            v_label = "positive" if valence_prob >= 0.5 else "negative"
            a_label = "high"     if arousal_prob >= 0.5 else "low"
            emotion = QUADRANT_TO_EMOTION.get((v_label, a_label), "calm")

        return {
            "emotion":    emotion,
            "confidence": confidence,
            "features":   features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  EMOTION INFERENCE ENGINE  (public API — used by eeg_server.py)
# ═══════════════════════════════════════════════════════════════════════════

class EmotionInferenceEngine:
    """
    Drop-in replacement for the old single-axis engine.

    eeg_server.py calls:
      engine = EmotionInferenceEngine()
      engine.push_sample(sample_14)
      msg = engine.predict(timestamp=t)   # returns broadcast-ready dict or None
    """

    CHANNELS = CHANNELS

    def __init__(
        self,
        valence_xgb_path=None,
        arousal_xgb_path=None,
        scaler_path=None,
        calibration_seconds=30,
        ewma_alpha=0.2,
        **kwargs,   # absorb any legacy kwargs without crashing
    ):
        self.recognizer = DualAxisRecognizer(
            valence_xgb_path=valence_xgb_path,
            arousal_xgb_path=arousal_xgb_path,
            scaler_path=scaler_path,
            calibration_seconds=calibration_seconds,
        )
        self.smoother = EmotionSmoother(ewma_alpha=ewma_alpha)

    def push_sample(self, sample):
        self.recognizer.push_sample(sample)

    def push_chunk(self, chunk):
        self.recognizer.push_chunk(chunk)

    def is_ready(self):
        return self.recognizer.is_ready()

    def is_calibrated(self):
        return self.recognizer.is_calibrated()

    def calibration_progress(self):
        return self.recognizer.calibration_progress()

    def predict(self, timestamp=None):
        ts = float(timestamp if timestamp is not None else time.time())
        raw = self.recognizer.predict()
        return self.smoother.process(raw, ts)


# ═══════════════════════════════════════════════════════════════════════════
#  ARTIFACT EXPORT  (called from notebook after training)
# ═══════════════════════════════════════════════════════════════════════════

def export_runtime_artifacts(valence_model, arousal_model, scaler, artifact_dir=None):
    """
    Save both models + scaler to disk in the expected directory layout.
    Called once from the training notebook after fitting.
    """
    artifact_root = Path(
        artifact_dir
        or os.getenv("EEG_EMOTION_ARTIFACT_DIR")
        or (Path(__file__).resolve().parent / "emotion_artifacts")
    )
    models_dir    = artifact_root / "models"
    artifacts_dir = artifact_root / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    valence_path = models_dir    / "valence_xgb.joblib"
    arousal_path = models_dir    / "arousal_xgb.joblib"
    scaler_path  = artifacts_dir / "scaler.joblib"
    meta_path    = artifact_root / "metadata.json"

    joblib.dump(valence_model, valence_path)
    joblib.dump(arousal_model, arousal_path)
    joblib.dump(scaler,        scaler_path)

    metadata = {
        "sample_rate":    FS,
        "window_seconds": WINDOW_SECONDS,
        "window_size":    WINDOW_SIZE,
        "channels":       CHANNELS,
        "n_channels":     N_CHANNELS,
        "feature_bands":  {k: list(v) for k, v in BANDS.items()},
        "band_order":     BAND_ORDER,
        "dasm_pairs":     DASM_PAIRS,
        "n_features":     N_CHANNELS * len(BANDS) + N_PAIRS * len(BANDS),
        "model_outputs": {
            "valence": {"type": "binary", "labels": ["negative", "positive"]},
            "arousal": {"type": "binary", "labels": ["low",      "high"]},
        },
        "quadrant_map":  {str(k): v for k, v in QUADRANT_TO_EMOTION.items()},
        "app_emotions":  list(EmotionSmoother.APP_EMOTIONS),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "valence_model_path": str(valence_path),
        "arousal_model_path": str(arousal_path),
        "scaler_path":        str(scaler_path),
        "metadata_path":      str(meta_path),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BRAINFLOW STREAM HELPER
# ═══════════════════════════════════════════════════════════════════════════

def run_brainflow_stream(
    board_id=-1,
    ip_address="",
    ip_port=0,
    serial_port="",
    predict_every=1.0,
    calibration_seconds=30,
    ewma_alpha=0.2,
    valence_xgb_path=None,
    arousal_xgb_path=None,
    scaler_path=None,
    debug=False,
):
    """
    Connect to a BrainFlow board and run live emotion inference.

    board_id -1   → SYNTHETIC_BOARD (no hardware needed, for testing)
    board_id -11  → STREAMING_BOARD (Emotiv via LSL bridge)

    Prints status lines:
      [BUFFERING] n/512 samples
      [CALIBRATING] 42%
      [EMOTION] happy  valence=0.73  arousal=0.61  conf=0.81
    """
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        from brainflow.data_filter import DataFilter
    except ImportError:
        raise ImportError(
            "BrainFlow not installed. Run: pip install brainflow"
        )

    params = BrainFlowInputParams()
    if ip_address:
        params.ip_address = ip_address
    if ip_port:
        params.ip_port = ip_port
    if serial_port:
        params.serial_port = serial_port

    BoardShim.enable_dev_board_logger() if debug else BoardShim.disable_board_logger()

    board      = BoardShim(board_id, params)
    eeg_ch     = BoardShim.get_eeg_channels(board_id)
    n_board_ch = len(eeg_ch)

    board.prepare_session()
    board.start_stream()
    print(f"[BRAINFLOW] Board {board_id} streaming. EEG channels: {n_board_ch}")

    engine = EmotionInferenceEngine(
        valence_xgb_path=valence_xgb_path,
        arousal_xgb_path=arousal_xgb_path,
        scaler_path=scaler_path,
        calibration_seconds=calibration_seconds,
        ewma_alpha=ewma_alpha,
    )

    last_predict = time.time()

    try:
        while True:
            time.sleep(0.04)  # ~25 Hz polling

            data = board.get_board_data()
            if data.shape[1] == 0:
                continue

            eeg_data = data[eeg_ch, :]  # (n_board_ch, n_samples)

            # Map board channels → model 14 channels
            n_use = min(n_board_ch, N_CHANNELS)
            chunk = eeg_data[:n_use, :].T  # (n_samples, n_use)

            # Pad with zeros if board has fewer than 14 channels
            if n_use < N_CHANNELS:
                pad   = np.zeros((chunk.shape[0], N_CHANNELS - n_use))
                chunk = np.hstack([chunk, pad])

            try:
                engine.push_chunk(chunk)
            except ChannelCountError as e:
                print(f"[WARN] {e}")
                continue

            now = time.time()
            if now - last_predict < predict_every:
                continue
            last_predict = now

            if not engine.is_ready():
                total  = sum(len(b) for b in engine.recognizer._buffers) // N_CHANNELS
                print(f"[BUFFERING] {total}/{WINDOW_SIZE} samples")
                continue

            result = engine.predict(timestamp=now)

            if result is None:
                pct = int(engine.calibration_progress() * 100)
                if not engine.is_calibrated():
                    print(f"[CALIBRATING] {pct}%")
                continue

            if result.get("type") == "status":
                pct = int(result.get("progress", 0) * 100)
                print(f"[CALIBRATING] {pct}%")
                continue

            if result.get("type") == "emotion":
                em   = result["emotion"]
                vp   = result.get("valence_prob", "?")
                ap   = result.get("arousal_prob", "?")
                conf = result["confidence"]
                print(
                    f"[EMOTION] {em:<12} "
                    f"valence={vp:.3f}  arousal={ap:.3f}  conf={conf:.3f}"
                )
                if debug:
                    print(f"          features={result['features']}")

    except KeyboardInterrupt:
        print("\n[BRAINFLOW] Stopped.")
    finally:
        board.stop_stream()
        board.release_session()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        description="Realtime EEG dual-axis emotion classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-valence",        default=None,  help="Path to valence_xgb.joblib")
    p.add_argument("--model-arousal",        default=None,  help="Path to arousal_xgb.joblib")
    p.add_argument("--scaler",               default=None,  help="Path to scaler.joblib")
    p.add_argument("--board-id",             type=int,   default=-1,   help="BrainFlow board ID (-1 = synthetic)")
    p.add_argument("--ip-address",           default="",               help="Board IP address")
    p.add_argument("--ip-port",              type=int,   default=0,    help="Board IP port")
    p.add_argument("--serial-port",          default="",               help="Board serial port")
    p.add_argument("--predict-every",        type=float, default=1.0,  help="Seconds between predictions")
    p.add_argument("--calibration-seconds",  type=int,   default=30,   help="Baseline calibration duration (s)")
    p.add_argument("--alpha",                type=float, default=0.2,  help="EWMA smoothing alpha")
    p.add_argument("--debug",                action="store_true",       help="Print feature stats and thresholds")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_brainflow_stream(
        board_id=args.board_id,
        ip_address=args.ip_address,
        ip_port=args.ip_port,
        serial_port=args.serial_port,
        predict_every=args.predict_every,
        calibration_seconds=args.calibration_seconds,
        ewma_alpha=args.alpha,
        valence_xgb_path=args.model_valence,
        arousal_xgb_path=args.model_arousal,
        scaler_path=args.scaler,
        debug=args.debug,
    )
