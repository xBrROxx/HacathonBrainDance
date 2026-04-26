"""
retrive_eeg_processed.py  —  test listener for the clean EEG WebSocket feed
============================================================================
Run this AFTER eeg_server.py is running to verify the broadcast is working.
It prints the timestamp and data shape of every incoming window.

Usage:
    python retrive_eeg_processed.py
"""

import asyncio
import websockets
import json
import numpy as np          # ← was missing in the original file


async def listen():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri} …")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to filtered EEG stream. Waiting for data …\n")
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                arr = np.array(data["data"])   # (256, 14)
                print(
                    f"t={data['timestamp']:.3f} | "
                    f"shape={arr.shape} | "
                    f"channels={data.get('channels', '?')} | "
                    f"mean={arr.mean():.4f}  std={arr.std():.4f}"
                )
    except ConnectionRefusedError:
        print("ERROR: Could not connect. Is eeg_server.py running?")
    except KeyboardInterrupt:
        print("\nListener stopped.")


if __name__ == "__main__":
    asyncio.run(listen())