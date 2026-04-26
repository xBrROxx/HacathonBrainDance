"""
eeg_server.py  —  EEG acquisition, preprocessing & WebSocket broadcast server
==============================================================================
Connects to the Emotiv Cortex SDK (wss://localhost:6868), preprocesses each
256-sample window, and rebroadcasts the clean data on ws://localhost:8765.

Preprocessing chain (matches the DEAP/notebook pipeline exactly):
  1. DC offset removal
  2. Bandpass  0.5 – 45 Hz  (4th-order Butterworth, zero-phase)
  3. Notch     50 Hz  (power-line, Q=30)
  4. Notch     60 Hz  (US power-line, Q=30)  ← harmonic guard
  5. EOG / blink rejection  (threshold on Fp/AF channels)
  6. EMG / jaw clamp        (threshold on temporal channels T7, T8)
  7. Average re-reference
  8. Amplitude clip  ±100 µV
  9. Z-score normalisation
 10. Smoothing  (5-sample moving average)
 11. Final NaN/Inf guard

Author: your-project
"""

import asyncio
import websockets
import json
import threading
import time
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import websocket as ws_client

# ─────────────────────────── CONFIG ───────────────────────────
CLIENT_ID     = "com.alumni_neuro.BrainDance"
CLIENT_SECRET = (
    "BgvspaFB7Ab5hWVjg6PpVoT5p7EELLj72Wm5obI9MLt7LhUVMoib8yBsVd7Idhlth33w2oJrMXCS1gD2z9egPaUeneCCdHbLtUG6aVlX6Shhgli0FJjdPaxFSOca2zUw"
)
CORTEX_URL    = "wss://localhost:6868"
WS_SERVER_PORT = 8765

FS            = 128    # Emotiv EPOC X sampling rate (Hz)  ← MUST match model
WINDOW_SIZE   = 256    # samples per processing window  (2 s @ 128 Hz)
OVERLAP       = 128    # samples to discard after each window (50 % overlap)

# Emotiv EPOC X channel order — MUST match training order in the notebook
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# Frontal / near-eye channels → blink & eye-movement artefacts (EOG)
# Higher index = the column index inside a (samples × 14) window matrix
EOG_CHANNEL_INDICES = [
    CHANNELS.index(c) for c in ['AF3', 'AF4', 'F7', 'F8']
]
# Temporal channels → jaw-clench / muscle artefacts (EMG)
EMG_CHANNEL_INDICES = [
    CHANNELS.index(c) for c in ['T7', 'T8']
]

# ─────────────────────────── GLOBAL STATE ─────────────────────
clients    = set()
buffer     = []
session_id = None
auth_token = None


# ══════════════════════════════════════════════════════════════
#  FILTER HELPERS
#  All filters use filtfilt (zero-phase) to avoid phase distortion,
#  matching scipy.signal.filtfilt calls in the notebook's feature
#  extraction cell (compute_features / extract_band_power).
# ══════════════════════════════════════════════════════════════

def _bandpass_filter(data: np.ndarray,
                     low: float = 0.5,
                     high: float = 45.0,
                     fs: int = 128) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass (matches notebook butter(4,...))."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def _notch_filter(data: np.ndarray,
                  freq: float,
                  fs: int = 128,
                  Q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at `freq` Hz (matches notebook iirnotch usage)."""
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return filtfilt(b, a, data, axis=0)


def _remove_dc(data: np.ndarray) -> np.ndarray:
    """Subtract per-channel mean (removes DC / slow drift)."""
    return data - np.mean(data, axis=0)


def _average_rereference(data: np.ndarray) -> np.ndarray:
    """Subtract common average reference across channels per time-point."""
    mean = np.mean(data, axis=1, keepdims=True)  # shape (samples, 1)
    return data - mean


def _clip_artifacts(data: np.ndarray,
                    threshold: float = 100.0) -> np.ndarray:
    """Hard-clip to ±threshold µV (removes extreme voltage spikes)."""
    return np.clip(data, -threshold, threshold)


def _normalize(data: np.ndarray) -> np.ndarray:
    """Per-channel Z-score normalisation (mean=0, std=1)."""
    std = np.std(data, axis=0)
    std[std == 0] = 1.0           # guard against dead channels
    return (data - np.mean(data, axis=0)) / std


def _smooth(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Simple moving-average smoothing along the time axis."""
    kernel = np.ones(kernel_size) / kernel_size
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data
    )


# ══════════════════════════════════════════════════════════════
#  ARTEFACT REJECTION
#
#  Blink / EOG:
#    Blinks produce large (100–300 µV) positive deflections primarily at
#    frontal and near-eye sites (Fp, AF, F channels).  When the peak-to-peak
#    amplitude on any EOG channel exceeds `eog_threshold` (default 150 µV)
#    the window is flagged as contaminated.
#
#  Jaw-clench / facial EMG:
#    Jaw clenching creates broadband high-frequency bursts (70–200 µV p-p)
#    on temporal channels (T7, T8).  We check peak-to-peak on those channels
#    against `emg_threshold` (default 80 µV).
#
#  The DEAP dataset was pre-processed by the authors using automatic artefact
#  removal (ICA + thresholding).  Applying the same amplitude thresholds
#  before feature extraction keeps the real-time pipeline consistent with
#  the training distribution and prevents the model from receiving data it
#  never saw during training.
# ══════════════════════════════════════════════════════════════

def _reject_eog_blink(data: np.ndarray,
                      channel_indices: list,
                      threshold: float = 150.0) -> bool:
    """
    Return True (→ reject window) if any EOG-prone channel has peak-to-peak
    amplitude > threshold (µV).
    data : (samples, channels) after DC removal but BEFORE clipping.
    """
    for idx in channel_indices:
        ch_data = data[:, idx]
        ptp = np.ptp(ch_data)          # peak-to-peak = max - min
        if ptp > threshold:
            return True
    return False


def _reject_jaw_emg(data: np.ndarray,
                    channel_indices: list,
                    threshold: float = 80.0) -> bool:
    """
    Return True (→ reject window) if any temporal / EMG-prone channel has
    peak-to-peak amplitude > threshold (µV).
    """
    for idx in channel_indices:
        ch_data = data[:, idx]
        if np.ptp(ch_data) > threshold:
            return True
    return False


# ══════════════════════════════════════════════════════════════
#  MAIN PREPROCESSING PIPELINE
#  Matches the DEAP preprocessing assumed by the notebook:
#   • DEAP's authors applied a 4–45 Hz bandpass + 50 Hz notch + downsampling.
#   • The notebook then runs filtfilt bandpass per band inside compute_features.
#   • We replicate the same approach here so the model receives data
#     it was trained on.
# ══════════════════════════════════════════════════════════════

def preprocess(window: list) -> np.ndarray | None:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    window : list of lists, shape (WINDOW_SIZE, n_channels)
        Raw EEG samples from the buffer.

    Returns
    -------
    np.ndarray of shape (WINDOW_SIZE, n_channels) or None if rejected.
    """
    try:
        print("[PROCESS] Starting preprocessing pipeline...")

        data = np.array(window, dtype=np.float64)   # (samples, channels)

        # ── Guard: invalid values ──────────────────────────────
        if not np.isfinite(data).all():
            print("[WARN] NaN/Inf detected → replacing with 0")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Guard: window too short ────────────────────────────
        if data.shape[0] < 10:
            print("[WARN] Window has < 10 samples → skipping")
            return None

        # ── Step 1: DC offset removal ──────────────────────────
        print("[STEP] 1/10  DC offset removal")
        data = _remove_dc(data)

        # ── Step 2: Bandpass 0.5–45 Hz ────────────────────────
        # Matches butter(4, [low,high], btype='band', fs=fs) in the notebook.
        print("[STEP] 2/10  Bandpass 0.5–45 Hz")
        data = _bandpass_filter(data, low=0.5, high=45.0, fs=FS)

        # ── Step 3: Notch 50 Hz (power-line Europe/Asia) ──────
        print("[STEP] 3/10  Notch 50 Hz")
        data = _notch_filter(data, freq=50.0, fs=FS, Q=30.0)

        # ── Step 4: Notch 60 Hz (US power-line harmonic guard) ─
        print("[STEP] 4/10  Notch 60 Hz")
        data = _notch_filter(data, freq=60.0, fs=FS, Q=30.0)

        # ── Step 5: Blink / EOG rejection ─────────────────────
        # Must happen BEFORE clipping so the raw amplitude is visible.
        print("[STEP] 5/10  EOG blink rejection (AF3, AF4, F7, F8 — threshold 150 µV)")
        if _reject_eog_blink(data, EOG_CHANNEL_INDICES, threshold=150.0):
            print("[REJECT] Window discarded: eye-blink / EOG artefact detected")
            return None

        # ── Step 6: Jaw / EMG rejection ───────────────────────
        print("[STEP] 6/10  EMG jaw rejection (T7, T8 — threshold 80 µV)")
        if _reject_jaw_emg(data, EMG_CHANNEL_INDICES, threshold=80.0):
            print("[REJECT] Window discarded: jaw-clench / EMG artefact detected")
            return None

        # ── Step 7: Average re-reference ──────────────────────
        print("[STEP] 7/10  Average re-reference")
        data = _average_rereference(data)

        # ── Step 8: Hard-clip ±100 µV ─────────────────────────
        print("[STEP] 8/10  Amplitude clip ±100 µV")
        data = _clip_artifacts(data, threshold=100.0)

        # ── Step 9: Z-score normalisation ─────────────────────
        print("[STEP] 9/10  Z-score normalisation")
        data = _normalize(data)

        # ── Step 10: Smoothing ────────────────────────────────
        print("[STEP] 10/10 5-sample moving-average smoothing")
        data = _smooth(data, kernel_size=5)

        # ── Final sanity check ────────────────────────────────
        if not np.isfinite(data).all():
            print("[ERROR] Output contains NaN/Inf after pipeline → dropping")
            return None

        print(f"[OK] Preprocessing done | shape={data.shape}\n")
        return data

    except Exception as exc:
        print(f"[ERROR] Preprocessing crashed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════
#  WEBSOCKET SERVER
# ══════════════════════════════════════════════════════════════

async def handler(websocket):
    print("WebSocket client connected")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)
        print("WebSocket client disconnected")


async def broadcast(message: str):
    if not clients:
        return
    dead = set()
    for c in clients:
        try:
            await c.send(message)
        except Exception:
            dead.add(c)
    clients -= dead


# ══════════════════════════════════════════════════════════════
#  EEG BUFFER & WINDOWING
# ══════════════════════════════════════════════════════════════

def handle_eeg(eeg: list, loop: asyncio.AbstractEventLoop):
    """Called for every raw EEG packet from the Cortex stream."""
    global buffer

    # Guard: malformed packet
    if not isinstance(eeg, list) or len(eeg) < 2:
        print("[WARN] Malformed EEG packet, skipping")
        return

    sample = eeg[1:]           # index 0 is the Cortex COUNTER/timestamp

    # Guard: wrong channel count
    if len(sample) != len(CHANNELS):
        print(f"[WARN] Expected {len(CHANNELS)} channels, got {len(sample)}")
        return

    buffer.append(sample)

    if len(buffer) >= WINDOW_SIZE:
        window = list(buffer[:WINDOW_SIZE])
        del buffer[:OVERLAP]   # slide forward by OVERLAP samples

        clean = preprocess(window)
        if clean is not None:
            message = json.dumps({
                "timestamp": time.time(),
                "channels":  CHANNELS,
                "data":      clean.tolist(),   # (256, 14) list-of-lists
            })
            asyncio.run_coroutine_threadsafe(broadcast(message), loop)


# ══════════════════════════════════════════════════════════════
#  CORTEX SDK CLIENT
# ══════════════════════════════════════════════════════════════

def start_cortex(loop: asyncio.AbstractEventLoop):
    global session_id, auth_token

    request_id = 0

    def send(ws, method: str, params: dict):
        nonlocal request_id
        request_id += 1
        ws.send(json.dumps({
            "id":       request_id,
            "jsonrpc":  "2.0",
            "method":   method,
            "params":   params,
        }))

    def on_message(ws, message: str):
        global session_id, auth_token

        data = json.loads(message)

        if "error" in data:
            print(f"[CORTEX ERROR] {data['error']}")
            return

        # ── Auth response ──────────────────────────────────────
        if "result" in data and isinstance(data["result"], dict) \
                and "cortexToken" in data["result"]:
            auth_token = data["result"]["cortexToken"]
            print("[CORTEX] Authorised — querying headsets")
            send(ws, "queryHeadsets", {})

        # ── Headset list ───────────────────────────────────────
        elif "result" in data and isinstance(data["result"], list):
            headsets = data["result"]
            if not headsets:
                print("[CORTEX] No headset found. Is the Emotiv dongle plugged in?")
                return
            headset_id = headsets[0]["id"]
            print(f"[CORTEX] Found headset: {headset_id}")
            send(ws, "controlDevice", {"command": "connect", "headset": headset_id})
            time.sleep(1)
            send(ws, "createSession", {
                "cortexToken": auth_token,
                "headset":     headset_id,
                "status":      "active",
            })

        # ── Session created ────────────────────────────────────
        elif "result" in data and isinstance(data["result"], dict) \
                and "id" in data["result"]:
            session_id = data["result"]["id"]
            print(f"[CORTEX] Session created: {session_id}")
            send(ws, "subscribe", {
                "cortexToken": auth_token,
                "session":     session_id,
                "streams":     ["eeg"],
            })

        # ── EEG data ───────────────────────────────────────────
        elif "eeg" in data:
            handle_eeg(data["eeg"], loop)

    def on_open(ws):
        print("[CORTEX] Connected — authorising …")
        send(ws, "authorize", {
            "clientId":     CLIENT_ID,
            "clientSecret": CLIENT_SECRET,
        })

    def on_error(ws, error):
        print(f"[CORTEX] WebSocket error: {error}")

    def on_close(ws, code, msg):
        print(f"[CORTEX] Connection closed ({code}: {msg})")

    app = ws_client.WebSocketApp(
        CORTEX_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.run_forever(sslopt={"cert_reqs": 0})   # Cortex uses self-signed cert


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def main():
    loop = asyncio.get_running_loop()

    # Start Cortex client in a background daemon thread
    threading.Thread(target=start_cortex, args=(loop,), daemon=True).start()

    async with websockets.serve(handler, "0.0.0.0", WS_SERVER_PORT):
        print(f"[SERVER] Broadcasting clean EEG on ws://localhost:{WS_SERVER_PORT}")
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())