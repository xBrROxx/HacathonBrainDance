import asyncio
import websockets
import json
import threading
import time
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import websocket as ws_client
from scipy.signal import butter, filtfilt, iirnotch

# ================= CONFIG =================
CLIENT_ID = "braindance"
CLIENT_SECRET = "w0gT6A9v8NOLpHO8usSYwly2Kan5UdaPmy3RzQCzhDLhVyI1AoXWwusb4Oh2LRQ6BTaQCIU5ywqQA5PBHtHtPoQQXOI3KF4WSYCM44ZD6Di3UFjVWhIAKcjCZkCiZYqO"
CORTEX_URL = "wss://localhost:6868"
WS_SERVER_PORT = 8765

FS = 128
WINDOW_SIZE = 256
OVERLAP = 128

# ================= GLOBAL STATE =================
clients = set()
buffer = []
session_id = None
auth_token = None

# ================= FILTERS =================
def bandpass(data, low=0.5, high=45, fs=128):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=0)  # zero-phase

def notch(data, freq=50, fs=128):
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
        clients.remove(websocket)
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
        clients.remove(d)

# ================= BUFFER =================
def handle_eeg(eeg, loop):
    global buffer

    # Edge case: malformed packet
    if not isinstance(eeg, list) or len(eeg) < 5:
        return

    sample = eeg[1:]  # remove timestamp
    buffer.append(sample)

    if len(buffer) >= WINDOW_SIZE:
        window = np.array(buffer[:WINDOW_SIZE])
        del buffer[:OVERLAP]

        clean = preprocess(window)

        if clean is not None:
            message = json.dumps({
                "timestamp": time.time(),
                "data": clean.tolist()
            })

            asyncio.run_coroutine_threadsafe(
                broadcast(message), loop
            )

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
            handle_eeg(data["eeg"], loop)

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

    threading.Thread(
        target=start_cortex,
        args=(loop,),
        daemon=True
    ).start()

    async with websockets.serve(handler, "0.0.0.0", WS_SERVER_PORT):
        print(f"Server running on ws://localhost:{WS_SERVER_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())