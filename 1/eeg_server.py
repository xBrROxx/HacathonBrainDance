import os
import asyncio
import websockets
import json
import threading
import time
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import websocket as ws_client
from scipy.signal import butter, filtfilt, iirnotch

try:
    from emotion_inference import EmotionInferenceEngine, MissingArtifactsError
except Exception as exc:
    # Keep the EEG relay usable even when model dependencies/artifacts aren't installed.
    EmotionInferenceEngine = None
    MissingArtifactsError = RuntimeError
    EMOTION_INFERENCE_IMPORT_ERROR = exc
else:
    EMOTION_INFERENCE_IMPORT_ERROR = None

# ================= CONFIG =================
CLIENT_ID = os.getenv("CORTEX_CLIENT_ID", "braindance")
CLIENT_SECRET = os.getenv("CORTEX_CLIENT_SECRET")
CORTEX_URL = os.getenv("CORTEX_URL", "wss://localhost:6868")
WS_SERVER_HOST = os.getenv("EEG_WS_HOST", "127.0.0.1")
WS_SERVER_PORT = int(os.getenv("EEG_WS_PORT", "8765"))

FS = 128
WINDOW_SIZE = 256
OVERLAP = 128
POWERLINE_HZ = float(os.getenv("EEG_POWERLINE_HZ", "50"))
MODEL_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# ================= GLOBAL STATE =================
clients = set()
buffer = []
session_id = None
auth_token = None
eeg_columns = None
eeg_channel_indices = None


def create_emotion_engine():
    if EmotionInferenceEngine is None:
        print(f"[WARN] Emotion inference unavailable: {EMOTION_INFERENCE_IMPORT_ERROR}")
        return None

    try:
        # The model path is optional at startup; we fall back to EEG-only streaming if missing.
        engine = EmotionInferenceEngine()
        print("[MODEL] Emotion inference ready")
        return engine
    except MissingArtifactsError as exc:
        print(f"[WARN] Emotion inference disabled: {exc}")
    except Exception as exc:
        print(f"[WARN] Failed to initialize emotion inference: {exc}")
    return None


emotion_engine = create_emotion_engine()

# ================= FILTERS =================
def bandpass(data, low=0.5, high=45, fs=128):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=0)  # zero-phase

def notch(data, freq=POWERLINE_HZ, fs=128):
    b, a = iirnotch(freq/(fs/2), Q=30)
    return filtfilt(b, a, data, axis=0)

def rereference(data):
    # average reference
    mean = np.mean(data, axis=1, keepdims=True)
    return data - mean

def remove_dc(data):
    return data - np.mean(data, axis=0)

def clip_artifacts(data, threshold=100):
    # remove extreme spikes (µV range assumption)
    return np.clip(data, -threshold, threshold)

def normalize(data):
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - np.mean(data, axis=0)) / std

def smooth(data, kernel_size=5):
    kernel = np.ones(kernel_size) / kernel_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)

# ================= MAIN PIPELINE =================

def preprocess(window):
    try:
        print("[PROCESS] Starting preprocessing pipeline...")

        data = np.array(window, dtype=np.float64)

        # ===== EDGE CASE: invalid values =====
        if not np.isfinite(data).all():
            print("[WARN] Found NaN/Inf → cleaning")
            data = np.nan_to_num(data)

        # ===== EDGE CASE: too small window =====
        if data.shape[0] < 10:
            print("[WARN] Window too small → skipping")
            return None

        print("[STEP] Removing DC offset...")
        data = remove_dc(data)

        print("[STEP] Bandpass 0.5–45 Hz...")
        data = bandpass(data)

        print("[STEP] Notch 50 Hz...")
        data = notch(data)

        print("[STEP] Re-referencing...")
        data = rereference(data)

        print("[STEP] Clipping artifacts...")
        data = clip_artifacts(data)

        print("[STEP] Normalizing...")
        data = normalize(data)

        print("[STEP] Smoothing...")
        data = smooth(data)

        # ===== FINAL SANITY =====
        if not np.isfinite(data).all():
            print("[ERROR] Output corrupted → dropping window")
            return None

        print(f"[SUCCESS] Done | shape={data.shape}\n")
        return data

    except Exception as e:
        print("[ERROR] Preprocessing failed:", e)
        return None

# ================= SERVER =================
async def handler(websocket):
    print("Client connected")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)
        print("Client disconnected")

async def broadcast(message):
    if not clients:
        return
    dead = []
    for c in clients:
        try:
            await c.send(message)
        except:
            dead.append(c)
    for d in dead:
        clients.discard(d)


def broadcast_payload(payload, loop):
    asyncio.run_coroutine_threadsafe(
        broadcast(json.dumps(payload)), loop
    )


def set_eeg_columns(cols):
    global eeg_columns, eeg_channel_indices

    if not isinstance(cols, list) or not cols:
        return

    eeg_columns = cols
    if all(name in cols for name in MODEL_CHANNELS):
        # Build the exact channel remap the notebook-trained model expects.
        eeg_channel_indices = [cols.index(name) for name in MODEL_CHANNELS]
        print(f"[MODEL] Using Cortex channel mapping: {MODEL_CHANNELS}")
    else:
        missing = [name for name in MODEL_CHANNELS if name not in cols]
        eeg_channel_indices = None
        print(f"[WARN] Missing expected EEG channels for model mapping: {missing}")


def maybe_extract_eeg_columns(message):
    if not isinstance(message, dict):
        return

    # Cortex can expose EEG labels either in the first stream payload or in
    # the subscribe response, depending on the SDK/version.
    eeg_payload = message.get("eeg")
    if isinstance(eeg_payload, dict) and isinstance(eeg_payload.get("cols"), list):
        set_eeg_columns(eeg_payload["cols"])
        return

    result = message.get("result")
    if isinstance(result, dict):
        success = result.get("success")
        if isinstance(success, list):
            for item in success:
                if item.get("streamName") == "eeg" and isinstance(item.get("cols"), list):
                    set_eeg_columns(item["cols"])
                    return


def extract_model_sample(eeg):
    if not isinstance(eeg, list):
        return None

    try:
        if eeg_channel_indices:
            return [float(eeg[index]) for index in eeg_channel_indices]

        # Cortex commonly prefixes counter/interpolation columns before the
        # 14 Emotiv channels. Use explicit labels when available; this is only
        # a fallback for local hackathon demos.
        if len(eeg) >= 16:
            return [float(value) for value in eeg[2:16]]
        if len(eeg) >= 15:
            return [float(value) for value in eeg[1:15]]
        if len(eeg) == 14:
            return [float(value) for value in eeg[:14]]
    except (TypeError, ValueError, IndexError):
        return None

    return None

# ================= BUFFER =================
def handle_eeg(eeg, loop):
    global buffer

    # Edge case: malformed packet
    if not isinstance(eeg, list) or len(eeg) < 5:
        return

    sample = eeg[1:]  # remove timestamp
    buffer.append(sample)

    if emotion_engine is not None:
        model_sample = extract_model_sample(eeg)
        if model_sample is not None:
            try:
                # Feed the inference buffer sample-by-sample while keeping the
                # existing processed-window stream untouched for other clients.
                emotion_engine.push_sample(model_sample)
            except Exception as exc:
                print(f"[WARN] Emotion sample rejected: {exc}")

    if len(buffer) >= WINDOW_SIZE:
        window = np.array(buffer[:WINDOW_SIZE])
        del buffer[:OVERLAP]
        timestamp = time.time()

        clean = preprocess(window)

        if clean is not None:
            broadcast_payload({
                "type": "eeg_window",
                "timestamp": timestamp,
                "data": clean.tolist()
            }, loop)

        if emotion_engine is not None:
            try:
                # Emotion messages are emitted on the same socket, but typed so
                # old EEG-window consumers can ignore them safely.
                emotion_message = emotion_engine.predict(timestamp=timestamp)
            except Exception as exc:
                print(f"[WARN] Emotion inference failed: {exc}")
                emotion_message = None

            if emotion_message is not None:
                broadcast_payload(emotion_message, loop)

# ================= CORTEX CLIENT =================
def start_cortex(loop):

    global session_id, auth_token

    request_id = 0

    def send(ws, method, params):
        nonlocal request_id
        request_id += 1
        req = {
            "id": request_id,
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        ws.send(json.dumps(req))

    def on_message(ws, message):
        global session_id, auth_token

        data = json.loads(message)
        maybe_extract_eeg_columns(data)

        # AUTH RESPONSE
        if "result" in data and "cortexToken" in data["result"]:
            auth_token = data["result"]["cortexToken"]

            send(ws, "queryHeadsets", {})

        # HEADSET FOUND
        elif "result" in data and isinstance(data["result"], list):
            if len(data["result"]) == 0:
                print("No headset found")
                return

            headset_id = data["result"][0]["id"]

            send(ws, "controlDevice", {
                "command": "connect",
                "headset": headset_id
            })

            time.sleep(1)

            send(ws, "createSession", {
                "cortexToken": auth_token,
                "headset": headset_id,
                "status": "active"
            })

        # SESSION CREATED
        elif "result" in data and "id" in data["result"]:
            session_id = data["result"]["id"]

            send(ws, "subscribe", {
                "cortexToken": auth_token,
                "session": session_id,
                "streams": ["eeg"]
            })

        # EEG DATA
        elif "eeg" in data:
            eeg_payload = data["eeg"]
            if isinstance(eeg_payload, list):
                handle_eeg(eeg_payload, loop)

    def on_open(ws):
        print("Connected to Cortex")

        send(ws, "authorize", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET
        })

    ws = ws_client.WebSocketApp(
        CORTEX_URL,
        on_message=on_message,
        on_open=on_open
    )

    ws.run_forever()

# ================= MAIN =================
async def main():
    loop = asyncio.get_running_loop()

    if CLIENT_SECRET:
        threading.Thread(
            target=start_cortex,
            args=(loop,),
            daemon=True
        ).start()
    else:
        print("[WARN] CORTEX_CLIENT_SECRET not set; EEG ingestion disabled")

    async with websockets.serve(handler, WS_SERVER_HOST, WS_SERVER_PORT):
        print(f"Server running on ws://{WS_SERVER_HOST}:{WS_SERVER_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
