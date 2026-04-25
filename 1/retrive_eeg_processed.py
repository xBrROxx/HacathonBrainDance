import asyncio
import websockets
import json
import numpy as np

async def listen():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to filtered EEG stream")
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            if data.get("type") != "eeg_window":
                continue
            print(f"t={data['timestamp']:.2f}, data shape={np.array(data['data']).shape}")

asyncio.run(listen())
