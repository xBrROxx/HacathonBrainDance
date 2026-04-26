"""
realtime_emotion.py  —  real-time valence inference from Emotiv EPOC X
=======================================================================
Reads the preprocessed EEG stream from ws://localhost:8765 (published by
eeg_server.py), extracts the same 84 DE+DASM features used during training,
runs the saved XGBoost model, and prints the live valence prediction.

Preprocessing note
------------------
eeg_server.py already handles:
    DC removal → bandpass 0.5–45 Hz → notch 50/60 Hz → blink/jaw rejection
    → average re-reference → clip ±100 µV → z-score → smooth

This script therefore applies NO additional signal cleaning — it only does
the SAME per-band filtfilt that the notebook's compute_features() does to
derive Differential Entropy (DE) and Differential Asymmetry (DASM) features.

Run order
---------
1. EmotivPRO (or Cortex)        — provides raw EEG
2. eeg_server.py                — cleans & broadcasts on ws://8765
3. THIS FILE                    — reads clean EEG, predicts emotion
4. website backend              — receives predictions (extend as needed)

Usage
-----
    python realtime_emotion.py
    python realtime_emotion.py --ws ws://localhost:8765  (custom server URL)
    python realtime_emotion.py --test                    (synthetic data, no headset)
"""

import os
import sys
import time
import threading
import collections
import argparse
import asyncio
import json
import numpy as np
import joblib
import websockets
from scipy import signal

# ─────────────────────────── MODEL PATHS ──────────────────────────────────
# Step 10 of the notebook saves to C:\Users\Cyberhell\eeg_emotion on Windows
# and /content/eeg_emotion on Colab.  Adjust BASE_DIR if you moved the files.
if os.path.isdir('/home'):
    BASE_DIR = '/home/gav/hackayon/project'
else:
    BASE_DIR = r'C:\Users\Cyberhell\eeg_emotion'

MODELS_DIR    = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

VALENCE_MODEL_PATH = os.path.join(MODELS_DIR,    'valence_xgb.joblib')
SCALER_PATH        = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')

# ─────────────────────────── CONSTANTS ────────────────────────────────────
# ALL values below MUST match the training notebook exactly.

FS             = 128
WINDOW_SECONDS = 4
WINDOW_SAMPLES = WINDOW_SECONDS * FS    # 512

CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
N_CH = len(CHANNELS)                    # 14

BANDS = {
    'Theta': (4,  8),
    'Alpha': (8,  12),
    'Beta':  (12, 30),
    'Gamma': (30, 45),
}

# Left-right channel pairs for DASM (must match LEFT_IDX/RIGHT_IDX in notebook)
LEFT_IDX  = [0, 1, 2, 3, 4, 5, 6]     # AF3  F7  F3  FC5  T7  P7  O1
RIGHT_IDX = [13, 12, 11, 10, 9, 8, 7]  # AF4  F8  F4  FC6  T8  P8  O2
N_PAIRS   = len(LEFT_IDX)              # 7
N_FEATURES = 4 * (N_CH + N_PAIRS)      # 84


# ══════════════════════════════════════════════════════════════════════════
#  REAL-TIME EMOTION RECOGNIZER
#  This is an updated version of the class from notebook Step 11 with:
#    • WebSocket input instead of BrainFlow (to consume eeg_server output)
#    • Improved calibration logging
#    • Thread-safe push_chunk from async receive loop
# ══════════════════════════════════════════════════════════════════════════

class RealTimeEmotionRecognizer:
    """
    Sliding-window valence recognizer.

    Lifecycle
    ---------
    1. Call push_chunk(chunk) as new data arrives.
    2. Call predict() every second or so.
       - Returns None while the 512-sample buffer fills (first ~4 s).
       - Returns {'status': 'calibrating', 'progress': 0..1} for ~30 s.
       - Returns {'status': 'predicting', 'valence': ..., 'valence_prob': ...}
         once calibration is complete.
    """

    def __init__(self,
                 valence_model_path: str = VALENCE_MODEL_PATH,
                 scaler_path: str = SCALER_PATH):

        if not os.path.exists(valence_model_path):
            raise FileNotFoundError(
                f"Model not found: {valence_model_path}\n"
                "Download valence_xgb.joblib from Colab (Step 10 output) "
                f"and place it in {MODELS_DIR}"
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}\n"
                "Download scaler.joblib from Colab (Step 10 output) "
                f"and place it in {ARTIFACTS_DIR}"
            )

        self.valence_model = joblib.load(valence_model_path)
        self.scaler        = joblib.load(scaler_path)
        print(f"[MODEL] Loaded XGBoost ({self.valence_model.n_estimators} trees)")
        print(f"[MODEL] Loaded StandardScaler from {scaler_path}")

        # Sliding buffer: one deque per channel, holds WINDOW_SAMPLES samples
        self._buffers = [
            collections.deque(maxlen=WINDOW_SAMPLES) for _ in range(N_CH)
        ]
        self._lock = threading.Lock()

        # Pre-build bandpass filters once (same as notebook _band_filters dict)
        self._band_filters = {
            name: signal.butter(4, [lo, hi], btype='band', fs=FS)
            for name, (lo, hi) in BANDS.items()
        }

        # Per-user calibration (mirrors per-subject z-score in notebook Step 3)
        self._calib_target = max(8, int(30 / WINDOW_SECONDS))  # ~8 windows = 30 s
        self._calib_feats  = []
        self._calib_mean   = None
        self._calib_std    = None

    # ── buffer management ──────────────────────────────────────────────────

    def push_chunk(self, chunk: np.ndarray):
        """
        Append a chunk of shape (N_CH, n_samples) to the sliding buffer.
        eeg_server.py sends (WINDOW_SIZE=256, N_CH=14), so transpose first.

        chunk: ndarray of shape (n_samples, N_CH)  or  (N_CH, n_samples)
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        # Accept either orientation
        if chunk.ndim == 2 and chunk.shape[1] == N_CH:
            chunk = chunk.T                 # → (N_CH, n_samples)
        if chunk.shape[0] != N_CH:
            raise ValueError(
                f"push_chunk: expected {N_CH} channels, got {chunk.shape}"
            )
        with self._lock:
            for ch_i in range(N_CH):
                self._buffers[ch_i].extend(chunk[ch_i].tolist())

    def _buffer_full(self) -> bool:
        return all(len(b) == WINDOW_SAMPLES for b in self._buffers)

    def _get_window(self) -> np.ndarray:
        """Return current window as (N_CH, WINDOW_SAMPLES) array."""
        with self._lock:
            return np.array([list(b) for b in self._buffers])

    # ── feature extraction — identical to compute_features() in notebook ──

    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        window : (N_CH, WINDOW_SAMPLES)
        Returns: 84-element feature vector [56 DE | 28 DASM]
        """
        de   = np.empty(N_CH   * 4, dtype=np.float64)
        dasm = np.empty(N_PAIRS * 4, dtype=np.float64)

        for bi, (b, a) in enumerate(self._band_filters.values()):
            filtered = signal.filtfilt(b, a, window, axis=1)
            var = np.var(filtered, axis=1) + 1e-12
            de_band = 0.5 * np.log(2 * np.pi * np.e * var)

            de[bi * N_CH:(bi + 1) * N_CH] = de_band
            dasm[bi * N_PAIRS:(bi + 1) * N_PAIRS] = (
                de_band[LEFT_IDX] - de_band[RIGHT_IDX]
            )

        return np.concatenate([de, dasm])

    # ── calibration ────────────────────────────────────────────────────────

    def is_calibrated(self) -> bool:
        return self._calib_mean is not None

    def calibration_progress(self) -> float:
        if self.is_calibrated():
            return 1.0
        return len(self._calib_feats) / self._calib_target

    def reset_calibration(self):
        """Restart per-user baseline collection."""
        self._calib_feats = []
        self._calib_mean  = None
        self._calib_std   = None
        print("[CALIB] Calibration reset.")

    def _normalize(self, feats: np.ndarray) -> np.ndarray:
        """Per-user z-score then global StandardScaler (matches Step 3)."""
        feats_z = (feats - self._calib_mean) / self._calib_std
        return self.scaler.transform(feats_z.reshape(1, -1))

    # ── inference ──────────────────────────────────────────────────────────

    def predict(self) -> dict | None:
        """
        Run one inference step.

        Returns
        -------
        None   — buffer not yet full
        dict   — one of:
            {'status': 'calibrating', 'progress': float, 'timestamp': float}
            {'status': 'predicting',  'valence': str, 'valence_prob': float,
             'timestamp': float}
        """
        if not self._buffer_full():
            return None

        feats = self._extract_features(self._get_window())

        # Phase 1: calibration
        if not self.is_calibrated():
            self._calib_feats.append(feats.copy())
            if len(self._calib_feats) >= self._calib_target:
                arr = np.array(self._calib_feats)
                self._calib_mean = arr.mean(axis=0)
                self._calib_std  = arr.std(axis=0) + 1e-8
                print("[CALIB] Baseline established. Starting predictions.")
            return {
                'status':    'calibrating',
                'progress':  self.calibration_progress(),
                'timestamp': time.time(),
            }

        # Phase 2: prediction
        feats_norm = self._normalize(feats)
        v_prob = float(self.valence_model.predict_proba(feats_norm)[0, 1])
        return {
            'status':       'predicting',
            'valence':      'positive' if v_prob > 0.5 else 'negative',
            'valence_prob': v_prob,
            'timestamp':    time.time(),
        }


# ══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET READER  (consumes eeg_server.py broadcast)
# ══════════════════════════════════════════════════════════════════════════

async def stream_from_server(recognizer: RealTimeEmotionRecognizer,
                             ws_url: str = "ws://localhost:8765",
                             predict_every: float = 1.0):
    """Connect to eeg_server.py, feed windows into the recognizer, print predictions."""
    print(f"[STREAM] Connecting to {ws_url} …")
    async with websockets.connect(ws_url) as ws:
        print("[STREAM] Connected. Filling buffer …")
        last_predict = 0.0

        async for raw in ws:
            msg  = json.loads(raw)
            data = np.array(msg["data"])        # (256, 14)
            recognizer.push_chunk(data)

            now = time.time()
            if now - last_predict >= predict_every:
                result = recognizer.predict()
                if result is None:
                    filled = len(recognizer._buffers[0])
                    print(f"[BUFFER] {filled}/{WINDOW_SAMPLES} samples")
                elif result['status'] == 'calibrating':
                    pct = result['progress'] * 100
                    print(f"[{time.strftime('%H:%M:%S')}] CALIBRATING … {pct:.0f}%")
                else:
                    bar_len = int(result['valence_prob'] * 20)
                    bar = '█' * bar_len + '░' * (20 - bar_len)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{result['valence'].upper():>8s}  "
                        f"[{bar}]  p={result['valence_prob']:.3f}"
                    )
                last_predict = now


# ══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TEST MODE  (no headset, no eeg_server)
# ══════════════════════════════════════════════════════════════════════════

def run_synthetic_test(recognizer: RealTimeEmotionRecognizer,
                       total_seconds: int = 90,
                       predict_every: float = 1.0):
    """
    Feed Gaussian noise at 128 Hz to verify the pipeline without hardware.
    Predictions will be random (noise input), but calibration → predicting
    state transitions confirm the plumbing is correct.
    """
    print("[TEST] Synthetic test mode — feeding Gaussian noise @ 128 Hz")
    print("[TEST] Predictions will be random (this is expected for fake data)")

    rng      = np.random.default_rng(42)
    end_time = time.time() + total_seconds
    last_predict = 0.0

    while time.time() < end_time:
        # Simulate 0.1 s of data arriving (12 samples @ 128 Hz ≈ 13)
        chunk = rng.normal(0, 20, size=(N_CH, 13)).astype(np.float64)
        recognizer.push_chunk(chunk)
        time.sleep(0.1)

        now = time.time()
        if now - last_predict >= predict_every:
            result = recognizer.predict()
            if result is None:
                filled = len(recognizer._buffers[0])
                print(f"[BUFFER] {filled}/{WINDOW_SAMPLES}")
            elif result['status'] == 'calibrating':
                print(f"[{time.strftime('%H:%M:%S')}] CALIBRATING … "
                      f"{result['progress']*100:.0f}%")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{result['valence'].upper():>8s}  "
                      f"p={result['valence_prob']:.3f}")
            last_predict = now

    print("[TEST] Done.")


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real-time EEG emotion recognition")
    parser.add_argument(
        "--ws",
        default="ws://localhost:8765",
        help="WebSocket URL of eeg_server.py (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with synthetic noise (no headset / eeg_server needed)"
    )
    parser.add_argument(
        "--model",
        default=VALENCE_MODEL_PATH,
        help=f"Path to valence_xgb.joblib (default: {VALENCE_MODEL_PATH})"
    )
    parser.add_argument(
        "--scaler",
        default=SCALER_PATH,
        help=f"Path to scaler.joblib (default: {SCALER_PATH})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Real-Time Emotion Recognizer")
    print(f"  Window: {WINDOW_SECONDS}s @ {FS} Hz ({WINDOW_SAMPLES} samples)")
    print(f"  Features: {N_FEATURES} (56 DE + 28 DASM)")
    print(f"  Channels: {CHANNELS}")
    print("=" * 60)

    recognizer = RealTimeEmotionRecognizer(
        valence_model_path=args.model,
        scaler_path=args.scaler,
    )

    if args.test:
        run_synthetic_test(recognizer)
    else:
        try:
            asyncio.run(stream_from_server(recognizer, ws_url=args.ws))
        except KeyboardInterrupt:
            print("\n[EXIT] Stopped by user.")


if __name__ == "__main__":
    main()