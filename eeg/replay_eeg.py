# replay_standalone.py
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/gav/hackayon/project/eeg")
from emotion_inference import EmotionInferenceEngine

RECORDING_PATH = "/home/gav/hackayon/project/recordings/.csv"
FS = 128

engine = EmotionInferenceEngine()
df = pd.read_csv(RECORDING_PATH, skiprows=1)
channels = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
data = df[channels].values

print(f"[REPLAY] {len(data)} samples loaded")

for i, sample in enumerate(data):
    engine.push_sample(sample.tolist())
    if i % FS == 0:  # predict once per second
        result = engine.predict()
        if result:
            print(f"[{i//FS:4d}s] {result}")
    time.sleep(1 / FS)  # real-time pace