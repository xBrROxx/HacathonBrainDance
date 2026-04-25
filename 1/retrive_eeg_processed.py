import asyncio
import websockets
import json

async def listen():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to filtered EEG stream")
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            print(f"t={data['timestamp']:.2f}, data shape={np.array(data['data']).shape}")

asyncio.run(listen())