"""
eeg_server.py — EEG acquisition, preprocessing, emotion inference, and WebSocket broadcast
==========================================================================================

Pipeline:
  1. EEG source (Cortex WebSocket OR LSL)
  2. Per-window preprocessing (filter, reject artifacts, normalize)
  3. Emotion inference (XGBoost valence → 5-way emotion mapping)
  4. WebSocket broadcast (separate "eeg_window" and "emotion" message types)
  5. Optional JSONL recording for audit/reprocessing

Environment variables:
  CORTEX_CLIENT_ID           — Emotiv Cortex client ID
  CORTEX_CLIENT_SECRET       — Emotiv Cortex client secret
  CORTEX_URL                 — Emotiv Cortex WebSocket endpoint (default: wss://localhost:6868)
  USE_LSL                    — If set, read from EmotivPRO LSL stream instead of Cortex WebSocket
  EEG_WS_HOST                — WebSocket server host (default: 127.0.0.1)
  EEG_WS_PORT                — WebSocket server port (default: 8765)
  EEG_POWERLINE_HZ           — Powerline frequency for notch filter (default: 50)
  EEG_EMOTION_ARTIFACT_DIR   — Path to model artifacts directory (auto-detected)
  EEG_VALENCE_MODEL_PATH     — Direct path to valence_xgb.joblib
  EEG_AROUSAL_MODEL_PATH     — Direct path to arousal_xgb.joblib
  EEG_SCALER_PATH            — Direct path to scaler.joblib
  PREDICTIONS_JSONL          — Override path to predictions log file
  RECORD_PREPROCESSED        — If set, write preprocessed windows to JSONL file
"""

import os
import asyncio
import websockets
import json
import threading
import time
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import websocket as ws_client


try:
    from emotion_inference import EmotionInferenceEngine, MissingArtifactsError
except Exception as exc:
    EmotionInferenceEngine = None
    MissingArtifactsError = RuntimeError
    EMOTION_INFERENCE_IMPORT_ERROR = exc
else:
    EMOTION_INFERENCE_IMPORT_ERROR = None

try:
    import pylsl
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv()

def _env_flag(name, default=False):
    """Parse common env boolean forms safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(env_name, default_path):
    """Resolve env path; relative paths are anchored at this file's directory."""
    raw = os.getenv(env_name)
    if not raw:
        return os.path.abspath(default_path)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(_SERVER_DIR, expanded))

CLIENT_ID           = os.getenv("CORTEX_CLIENT_ID", "braindance")
CLIENT_SECRET       = os.getenv("CORTEX_CLIENT_SECRET")
CORTEX_URL          = os.getenv("CORTEX_URL", "wss://localhost:6868")
USE_LSL             = _env_flag("USE_LSL", default=False)
WS_SERVER_HOST      = os.getenv("EEG_WS_HOST", "127.0.0.1")
WS_SERVER_PORT      = int(os.getenv("EEG_WS_PORT", "8765"))

FS                  = 128
WINDOW_SIZE         = 256
OVERLAP             = 128
POWERLINE_HZ        = float(os.getenv("EEG_POWERLINE_HZ", "50"))

MODEL_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# EOG channels (blink detection)
EOG_CHANNELS = ["AF3", "AF4", "F7", "F8"]
EMG_CHANNELS = ["T7", "T8"]

# Record options
RECORD_PREPROCESSED = os.getenv("RECORD_PREPROCESSED")

# Predictions JSONL — always written, path unified so tailer finds same file.
# Resolves to HacathonBrainDance/emotion_predictions.jsonl regardless of CWD.
_SERVER_DIR      = os.path.dirname(os.path.abspath(__file__))   # eeg/
_REPO_ROOT       = os.path.dirname(_SERVER_DIR)                  # HacathonBrainDance/
PREDICTIONS_JSONL = _resolve_path(
    "PREDICTIONS_JSONL",
    os.path.join(_REPO_ROOT, "emotion_predictions.jsonl")
)
PREPROCESSED_JSONL = _resolve_path(
    "PREPROCESSED_JSONL",
    os.path.join(_SERVER_DIR, "eeg_preprocessed.jsonl")
)

# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════

clients             = set()
buffer              = []
session_id          = None
auth_token          = None
eeg_columns         = None
eeg_channel_indices = None
emotion_engine      = None

# Recording file handles
prediction_file     = None
preprocessed_file   = None

# ═══════════════════════════════════════════════════════════════════════════
#  FILE RECORDING
# ═══════════════════════════════════════════════════════════════════════════

def init_recording():
    global prediction_file, preprocessed_file

    # Always open predictions file — it is the live feed for the browser tailer
    prediction_file = open(PREDICTIONS_JSONL, "a", buffering=1)
    print(f"[RECORD] Emotions → {PREDICTIONS_JSONL}")

    if RECORD_PREPROCESSED:
        preprocessed_file = open(PREPROCESSED_JSONL, "a", buffering=1)
        print(f"[RECORD] Windows → {PREPROCESSED_JSONL}")


def record_emotion(msg):
    if prediction_file and msg:
        prediction_file.write(json.dumps(msg) + "\n")


def record_preprocessed(msg):
    if preprocessed_file and msg:
        preprocessed_file.write(json.dumps(msg) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  EMOTION ENGINE INIT
# ═══════════════════════════════════════════════════════════════════════════

def create_emotion_engine():
    if EmotionInferenceEngine is None:
        print(f"[WARN] Emotion inference unavailable: {EMOTION_INFERENCE_IMPORT_ERROR}")
        return None

    try:
        engine = EmotionInferenceEngine(
            valence_xgb_path=os.getenv("EEG_VALENCE_MODEL_PATH"),
            arousal_xgb_path=os.getenv("EEG_AROUSAL_MODEL_PATH"),
            scaler_path=os.getenv("EEG_SCALER_PATH"),
        )
        print("[MODEL] Emotion inference ready (valence + arousal)")
        return engine
    except MissingArtifactsError as exc:
        print(f"[WARN] Emotion inference disabled: {exc}")
    except Exception as exc:
        print(f"[WARN] Failed to initialize emotion inference: {exc}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def _bandpass_filter(data, low=0.5, high=45.0, fs=128):
    """4th-order zero-phase Butterworth bandpass."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def _notch_filter(data, freq=POWERLINE_HZ, fs=128, Q=30.0):
    """Zero-phase IIR notch filter."""
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return filtfilt(b, a, data, axis=0)


def _remove_dc(data):
    """Subtract per-channel mean."""
    return data - np.mean(data, axis=0)


def _average_rereference(data):
    """Subtract common average reference."""
    mean = np.mean(data, axis=1, keepdims=True)
    return data - mean


def _clip_artifacts(data, threshold=100.0):
    """Hard-clip to ±threshold µV."""
    return np.clip(data, -threshold, threshold)


def _normalize(data):
    """Per-channel Z-score normalisation."""
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    return (data - np.mean(data, axis=0)) / std


def _smooth(data, kernel_size=5):
    """Simple moving-average smoothing."""
    kernel = np.ones(kernel_size) / kernel_size
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ARTIFACT REJECTION — Blink & Jaw
#  (CRITICAL: matches DEAP preprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def _reject_eog_blink(data, eog_indices, threshold=150.0):
    """
    Return True (reject window) if any EOG channel has peak-to-peak > threshold.
    data: (samples, channels) after DC removal.
    """
    for idx in eog_indices:
        ch_data = data[:, idx]
        ptp = np.ptp(ch_data)  # peak-to-peak
        if ptp > threshold:
            return True
    return False


def _reject_jaw_emg(data, emg_indices, threshold=80.0):
    """
    Return True (reject window) if any EMG channel has peak-to-peak > threshold.
    """
    for idx in emg_indices:
        ch_data = data[:, idx]
        if np.ptp(ch_data) > threshold:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(window, eog_ch_indices, emg_ch_indices):
    """
    Full preprocessing matching DEAP + notebook pipeline.
    
    window: list of (WINDOW_SIZE, n_channels)
    eog_ch_indices, emg_ch_indices: channel indices for artifact rejection
    
    Returns: np.ndarray of shape (WINDOW_SIZE, n_channels) or None if rejected
    """
    try:
        data = np.array(window, dtype=np.float64)

        # Guard: invalid values
        if not np.isfinite(data).all():
            print("[WARN] NaN/Inf detected → replacing with 0")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Guard: too small
        if data.shape[0] < 10:
            print("[WARN] Window < 10 samples → skipping")
            return None

        # Step 1: DC removal
        data = _remove_dc(data)

        # Step 2: Bandpass 0.5–45 Hz
        data = _bandpass_filter(data, low=0.5, high=45.0, fs=FS)

        # Step 3: Notch powerline (50 or 60 Hz)
        data = _notch_filter(data, freq=POWERLINE_HZ, fs=FS, Q=30.0)

        # Step 4: Additional 60 Hz harmonic guard (if main is 50)
        if POWERLINE_HZ == 50:
            data = _notch_filter(data, freq=60.0, fs=FS, Q=30.0)

        # Step 5: EOG blink rejection (must be BEFORE clipping to see raw amplitude)
        if _reject_eog_blink(data, eog_ch_indices, threshold=150.0):
            print("[REJECT] EOG blink artefact detected")
            return None

        # Step 6: EMG jaw rejection
        if _reject_jaw_emg(data, emg_ch_indices, threshold=80.0):
            print("[REJECT] EMG jaw artefact detected")
            return None

        # Step 7: Average re-reference
        data = _average_rereference(data)

        # Step 8: Amplitude clip
        data = _clip_artifacts(data, threshold=100.0)

        # Step 9: Z-score normalise
        data = _normalize(data)

        # Step 10: Smoothing
        data = _smooth(data, kernel_size=5)

        # Final guard
        if not np.isfinite(data).all():
            print("[ERROR] Output corrupted → dropping")
            return None

        print(f"[OK] Preprocessed | shape={data.shape}")
        return data

    except Exception as e:
        print(f"[ERROR] Preprocessing crashed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET SERVER
# ═══════════════════════════════════════════════════════════════════════════

async def handler(websocket):
    """Accept incoming WebSocket connections (for future UI clients)."""
    print("[WS] Client connected")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)
        print("[WS] Client disconnected")


async def broadcast(message):
    """Broadcast message to all connected clients."""
    global clients
    if not clients:
        return
    dead = set()
    for c in clients:
        try:
            await c.send(message)
        except Exception:
            dead.add(c)
    clients -= dead

async def tail_jsonl_and_broadcast(filepath):
    """Watch emotion_predictions.jsonl and forward every new line to browser clients."""
    import os
    while not os.path.exists(filepath):
        print(f"[TAIL] Waiting for {filepath}…")
        await asyncio.sleep(1)
    print(f"[TAIL] Watching {filepath}")
    with open(filepath, "r") as f:
        f.seek(0, 2)  # jump to end, skip old entries
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                if line:
                    try:
                        payload = json.loads(line)
                        await broadcast(json.dumps(payload))
                        print(f"[TAIL] → {payload.get('type')} / {payload.get('emotion', payload.get('status', ''))}")
                    except json.JSONDecodeError:
                        pass
            else:
                await asyncio.sleep(0.1)
                
def broadcast_payload(payload, loop):
    """Threadsafe broadcast from EEG thread."""
    asyncio.run_coroutine_threadsafe(
        broadcast(json.dumps(payload)), loop
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CHANNEL MAPPING (Cortex → Model)
# ═══════════════════════════════════════════════════════════════════════════

def set_eeg_columns(cols):
    """Extract channel indices from Cortex message."""
    global eeg_columns, eeg_channel_indices

    if not isinstance(cols, list) or not cols:
        return

    eeg_columns = cols
    if all(name in cols for name in MODEL_CHANNELS):
        eeg_channel_indices = [cols.index(name) for name in MODEL_CHANNELS]
        print(f"[MODEL] Channel mapping: {MODEL_CHANNELS}")
    else:
        missing = [name for name in MODEL_CHANNELS if name not in cols]
        eeg_channel_indices = None
        print(f"[WARN] Missing channels for model: {missing}")


def get_eog_emg_indices(cols):
    """Get channel indices for artifact rejection."""
    eog_indices, emg_indices = [], []
    for idx, name in enumerate(cols):
        if name in EOG_CHANNELS:
            eog_indices.append(idx)
        if name in EMG_CHANNELS:
            emg_indices.append(idx)
    return eog_indices, emg_indices


def maybe_extract_eeg_columns(message):
    """Try to extract channel metadata from Cortex message."""
    if not isinstance(message, dict):
        return

    # Option 1: EEG stream payload
    eeg_payload = message.get("eeg")
    if isinstance(eeg_payload, dict) and isinstance(eeg_payload.get("cols"), list):
        set_eeg_columns(eeg_payload["cols"])
        return

    # Option 2: Subscribe response
    result = message.get("result")
    if isinstance(result, dict):
        success = result.get("success")
        if isinstance(success, list):
            for item in success:
                if item.get("streamName") == "eeg" and isinstance(item.get("cols"), list):
                    set_eeg_columns(item["cols"])
                    return


def extract_model_sample(eeg):
    """Extract 14 brain channels from Cortex EEG packet."""
    if not isinstance(eeg, list):
        return None

    try:
        # Use explicit channel mapping if available
        if eeg_channel_indices:
            return [float(eeg[index]) for index in eeg_channel_indices]

        # Fallback: try last 14 channels (common in Cortex SDK versions)
        if len(eeg) >= 16:
            return [float(value) for value in eeg[2:16]]
        if len(eeg) >= 15:
            return [float(value) for value in eeg[1:15]]
        if len(eeg) == 14:
            return [float(value) for value in eeg[:14]]
    except (TypeError, ValueError, IndexError):
        return None

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  EEG BUFFER & WINDOWING
# ═══════════════════════════════════════════════════════════════════════════

def handle_eeg(eeg, loop):
    """Process incoming EEG sample."""
    global buffer, emotion_engine

    # Guard: malformed
    if not isinstance(eeg, list) or len(eeg) < 5:
        return

    # Keep full Cortex packet for recording
    raw_packet = eeg.copy()
    sample = eeg[1:]  # remove Cortex counter
    buffer.append(sample)

    # Feed model if available
    if emotion_engine is not None:
        model_sample = extract_model_sample(eeg)
        if model_sample is not None:
            try:
                emotion_engine.push_sample(model_sample)
            except Exception as exc:
                print(f"[WARN] Emotion sample rejected: {exc}")

    # Process window
    if len(buffer) >= WINDOW_SIZE:
        window = list(buffer[:WINDOW_SIZE])
        del buffer[:OVERLAP]
        timestamp = time.time()

        # Get channel indices for artifact rejection
        if eeg_columns:
            eog_indices, emg_indices = get_eog_emg_indices(eeg_columns)
        else:
            # Fallback: assume standard Emotiv order
            eog_indices = [
                MODEL_CHANNELS.index(ch) for ch in ["AF3", "AF4", "F7", "F8"]
            ]
            emg_indices = [
                MODEL_CHANNELS.index(ch) for ch in ["T7", "T8"]
            ]

        # Preprocess with artifact rejection
        clean = preprocess(window, eog_indices, emg_indices)

        if clean is not None:
            # Broadcast EEG window
            broadcast_payload({
                "type": "eeg_window",
                "timestamp": timestamp,
                "data": clean.tolist(),
            }, loop)

            # Record if enabled
            record_preprocessed({
                "timestamp": timestamp,
                "data": clean.tolist(),
            })

        # Inference
        if emotion_engine is not None:
            try:
                emotion_message = emotion_engine.predict(timestamp=timestamp)
            except Exception as exc:
                print(f"[WARN] Inference failed: {exc}")
                emotion_message = None

            if emotion_message is not None:
                broadcast_payload(emotion_message, loop)
                record_emotion(emotion_message)


# ═══════════════════════════════════════════════════════════════════════════
#  CORTEX CLIENT (WebSocket)
# ═══════════════════════════════════════════════════════════════════════════

def start_cortex(loop):
    """Connect to Emotiv Cortex via WebSocket."""
    global session_id, auth_token

    request_id = 0

    def send(ws, method, params):
        nonlocal request_id
        request_id += 1
        ws.send(json.dumps({
            "id": request_id,
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))

    def on_message(ws, message):
        global session_id, auth_token

        data = json.loads(message)
        maybe_extract_eeg_columns(data)

        # Auth response
        if "result" in data and "cortexToken" in data["result"]:
            auth_token = data["result"]["cortexToken"]
            send(ws, "queryHeadsets", {})

        # Headset list
        elif "result" in data and isinstance(data["result"], list):
            headsets = data["result"]
            if not headsets:
                print("[CORTEX] No headset found")
                return
            headset_id = headsets[0]["id"]
            print(f"[CORTEX] Found: {headset_id}")
            send(ws, "controlDevice", {"command": "connect", "headset": headset_id})
            time.sleep(1)
            send(ws, "createSession", {
                "cortexToken": auth_token,
                "headset": headset_id,
                "status": "active",
            })

        # Session created
        elif "result" in data and isinstance(data["result"], dict) and "id" in data["result"]:
            session_id = data["result"]["id"]
            print(f"[CORTEX] Session: {session_id}")
            send(ws, "subscribe", {
                "cortexToken": auth_token,
                "session": session_id,
                "streams": ["eeg"],
            })

        # EEG data
        elif "eeg" in data:
            eeg_payload = data["eeg"]
            if isinstance(eeg_payload, list):
                handle_eeg(eeg_payload, loop)

    def on_open(ws):
        print("[CORTEX] Connecting…")
        send(ws, "authorize", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET,
        })

    def on_error(ws, error):
        print(f"[CORTEX] Error: {error}")

    def on_close(ws, code, msg):
        print(f"[CORTEX] Closed ({code}: {msg})")

    app = ws_client.WebSocketApp(
        CORTEX_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.run_forever(sslopt={"cert_reqs": 0})


# ═══════════════════════════════════════════════════════════════════════════
#  LSL CLIENT (Lab Streaming Layer)
# ═══════════════════════════════════════════════════════════════════════════

def start_lsl(loop):
    """Connect to EmotivPRO LSL stream."""
    if not HAS_LSL:
        print("[LSL] pylsl not installed. Install with: pip install pylsl")
        return

    print("[LSL] Searching for EEG stream…")
    streams = pylsl.resolve_byprop('type', 'EEG', timeout=5)

    if not streams:
        print("[LSL] No EEG stream found. Enable LSL in EmotivPRO.")
        return

    inlet = pylsl.StreamInlet(streams[0])
    info = inlet.info()
    ch_count = info.channel_count()
    print(f"[LSL] Connected to '{info.name()}' ({ch_count} channels)")

# Extract channel names from metadata
    ch_list = []
    ch = info.desc().child("channels").child("channel")
    for i in range(ch_count):
        label = ch.child_value("label")
        ch_list.append(label if label else f"Ch{i}")
        ch = ch.next_sibling("channel")
    set_eeg_columns(ch_list)

    # Read stream
    while True:
        sample, timestamp = inlet.pull_sample()
        if sample:
            # LSL sample is raw values (no counter), so prepend 0
            eeg_packet = [0] + list(sample)
            handle_eeg(eeg_packet, loop)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    global emotion_engine

    loop = asyncio.get_running_loop()
    init_recording()
    emotion_engine = create_emotion_engine()

    # Start EEG source
    if USE_LSL:
        print("[MAIN] Using LSL source")
        if not HAS_LSL:
            print("[ERROR] LSL requested but pylsl not available. Install: pip install pylsl")
            return
        threading.Thread(target=start_lsl, args=(loop,), daemon=True).start()
    else:
        if not CLIENT_SECRET:
            print("[WARN] CORTEX_CLIENT_SECRET not set; EEG ingestion disabled")
        else:
            print("[MAIN] Using Cortex WebSocket")
            threading.Thread(target=start_cortex, args=(loop,), daemon=True).start()

    # Start WebSocket server
    async with websockets.serve(handler, WS_SERVER_HOST, WS_SERVER_PORT):
        print(f"[SERVER] Broadcasting on ws://{WS_SERVER_HOST}:{WS_SERVER_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())