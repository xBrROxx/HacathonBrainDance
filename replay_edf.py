"""
replay_edf.py — Feed an EmotivPRO EDF recording through the emotion pipeline
=============================================================================

Use this when LSL fails and you want to run the full inference pipeline
on a previously recorded EDF file from EmotivPRO.

Usage
-----
    python replay_edf.py --file /path/to/recording.edf
    python replay_edf.py --file /path/to/recording.edf --realtime   # pace at 128 Hz
    python replay_edf.py --file /path/to/recording.edf --out results.jsonl

Dependencies
------------
    pip install pyedflib pandas

Run order
---------
    1. Record in EmotivPRO → Export as EDF
    2. python replay_edf.py --file my_session.edf
    No headset, no eeg_server.py, no LSL needed.
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / "eeg" / ".env")

try:
    import pyedflib
except ImportError:
    print("[ERROR] pyedflib not installed. Run: pip install pyedflib")
    sys.exit(1)

# ── path to emotion_inference.py ──────────────────────────────────────────
# Adjust this if your folder layout differs
SCRIPT_DIR   = Path(__file__).resolve().parent
EEG_DIR      = SCRIPT_DIR / "eeg"          # project/eeg/emotion_inference.py
sys.path.insert(0, str(EEG_DIR))
sys.path.insert(0, str(SCRIPT_DIR))        # fallback if same folder

try:
    from emotion_inference import EmotionInferenceEngine
except ImportError as e:
    print(f"[ERROR] Cannot import emotion_inference: {e}")
    print("        Make sure emotion_inference.py is in ./eeg/ relative to this script")
    sys.exit(1)

# ── the 14 model channels in order ───────────────────────────────────────
MODEL_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2",  "P8", "T8", "FC6", "F4", "F8", "AF4",
]

FS = 128  # Emotiv EPOC X sample rate


# ══════════════════════════════════════════════════════════════════════════
#  EDF READER
# ══════════════════════════════════════════════════════════════════════════

def load_edf(path: str):
    """
    Read an EmotivPRO EDF file.

    Returns
    -------
    data       : np.ndarray of shape (n_samples, 14)  — model channels only
    fs         : int — sample rate reported by the file
    duration_s : float — recording duration in seconds
    """
    path = str(path)
    print(f"[EDF] Opening {path}")

    f = pyedflib.EdfReader(path)

    try:
        n_signals   = f.signals_in_file
        labels      = f.getSignalLabels()          # list of channel name strings
        sample_rate = int(f.getSampleFrequency(0)) # assume uniform across channels
        n_samples   = f.getNSamples()[0]

        print(f"[EDF] Channels ({n_signals}): {labels}")
        print(f"[EDF] Sample rate: {sample_rate} Hz  |  Samples: {n_samples}  "
              f"|  Duration: {n_samples/sample_rate:.1f}s")

        # ── map EDF labels → model channel indices ────────────────────────
        # EmotivPRO sometimes exports as "EEG.AF3" or "AF3" — strip prefix
        clean_labels = [l.replace("EEG.", "").strip() for l in labels]
        missing = [ch for ch in MODEL_CHANNELS if ch not in clean_labels]

        if missing:
            print(f"[WARN] Channels missing from EDF: {missing}")
            print(f"       Available: {clean_labels}")
            print("       Will use available channels in EDF order as fallback.")
            # fallback: use first 14 channels
            indices = list(range(min(14, n_signals)))
        else:
            indices = [clean_labels.index(ch) for ch in MODEL_CHANNELS]

        # ── read selected channels ────────────────────────────────────────
        print(f"[EDF] Reading {len(indices)} channels …")
        data = np.zeros((n_samples, len(indices)), dtype=np.float64)
        for out_i, edf_i in enumerate(indices):
            data[:, out_i] = f.readSignal(edf_i)

    finally:
        f.close()

    print(f"[EDF] Loaded  shape={data.shape}  "
          f"range=[{data.min():.1f}, {data.max():.1f}] µV")
    return data, sample_rate, n_samples / sample_rate


# ══════════════════════════════════════════════════════════════════════════
#  REPLAY LOOP
# ══════════════════════════════════════════════════════════════════════════

def replay(edf_path: str,
           realtime: bool = False,
           out_path: str | None = None):
    """
    Push EDF data through EmotionInferenceEngine sample-by-sample,
    print results, and optionally save to JSONL.
    """
    data, fs, duration_s = load_edf(edf_path)
    n_samples = len(data)

    # ── init engine ───────────────────────────────────────────────────────
    try:
        engine = EmotionInferenceEngine()
        print("[MODEL] EmotionInferenceEngine ready\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # ── output file ───────────────────────────────────────────────────────
    out_file = None
    if out_path:
        out_file = open(out_path, "w", buffering=1)
        print(f"[RECORD] Saving results → {out_path}\n")

    # ── counters ──────────────────────────────────────────────────────────
    predict_every = fs          # call predict() once per second of data
    n_predictions = 0
    n_emotions    = 0
    t_start       = time.time()

    print("=" * 60)
    print(f"  Replaying {duration_s:.1f}s of EEG  ({n_samples} samples)")
    print(f"  Real-time pacing: {'ON' if realtime else 'OFF (fast)'}")
    print("=" * 60)
    print()

    try:
        for i, sample in enumerate(data):
            engine.push_sample(sample.tolist())

            # predict once per second of data
            if i % predict_every == 0:
                ts = i / fs          # seconds into recording
                result = engine.predict(timestamp=ts)
                n_predictions += 1

                if result is None:
                    pass  # buffer still filling

                elif result.get("status") == "calibrating":
                    pct = result.get("progress", 0) * 100
                    print(f"  [{ts:6.1f}s] CALIBRATING … {pct:.0f}%")

                elif result.get("type") == "emotion":
                    emotion   = result["emotion"].upper()
                    conf      = result["confidence"]
                    bar_len   = int(conf * 20)
                    bar       = "█" * bar_len + "░" * (20 - bar_len)
                    feats     = result.get("features", {})
                    feat_str  = "  ".join(
                        f"{k}={v:.2f}" for k, v in feats.items()
                    )
                    print(f"  [{ts:6.1f}s] {emotion:<8s}  [{bar}]  "
                          f"conf={conf:.3f}   {feat_str}")
                    n_emotions += 1

                    if out_file:
                        out_file.write(json.dumps(result) + "\n")

            if realtime:
                time.sleep(1 / fs)

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")

    finally:
        if out_file:
            out_file.close()

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  Finished in {elapsed:.1f}s")
    print(f"  Predictions made : {n_predictions}")
    print(f"  Emotions emitted : {n_emotions}")
    if out_path:
        print(f"  Results saved to : {out_path}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Replay an EmotivPRO EDF recording through the emotion pipeline"
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the .edf file exported from EmotivPRO"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace playback at real 128 Hz (slow). Default: as fast as possible."
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Optional path to save emotion predictions as JSONL"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)

    replay(
        edf_path=args.file,
        realtime=args.realtime,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()