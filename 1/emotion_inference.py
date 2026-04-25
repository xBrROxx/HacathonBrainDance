import collections
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import signal


class MissingArtifactsError(RuntimeError):
    pass


class RealTimeValenceRecognizer:
    """Realtime valence recognizer copied from the notebook's Step 11 logic."""

    FS = 128
    WINDOW_SECONDS = 4
    CHANNELS = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    ]
    BANDS = {
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45),
    }
    LEFT_IDX = [0, 1, 2, 3, 4, 5, 6]
    RIGHT_IDX = [13, 12, 11, 10, 9, 8, 7]

    def __init__(
        self,
        valence_xgb_path=None,
        scaler_path=None,
        calibration_seconds=30,
    ):
        valence_xgb_path, scaler_path = resolve_artifact_paths(
            valence_xgb_path=valence_xgb_path,
            scaler_path=scaler_path,
        )

        self.valence_model = joblib.load(valence_xgb_path)
        self.scaler = joblib.load(scaler_path)

        self.fs = self.FS
        self.channels = list(self.CHANNELS)
        self.window_seconds = self.WINDOW_SECONDS
        self.window_size = self.window_seconds * self.fs
        self.n_channels = len(self.channels)

        self._buffers = [
            collections.deque(maxlen=self.window_size)
            for _ in range(self.n_channels)
        ]
        # Match the notebook's calibration idea: normalize this wearer against
        # a short unlabeled baseline instead of retraining the model.
        self._calib_target_windows = max(8, int(calibration_seconds / self.window_seconds))
        self._calib_features = []
        self._calib_mean = None
        self._calib_std = None
        self._band_filters = {
            name: signal.butter(4, [low, high], btype="band", fs=self.fs)
            for name, (low, high) in self.BANDS.items()
        }

    def is_calibrated(self):
        return self._calib_mean is not None

    def calibration_progress(self):
        if self.is_calibrated():
            return 1.0
        return len(self._calib_features) / self._calib_target_windows

    def push_sample(self, sample):
        if len(sample) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")
        for index, value in enumerate(sample):
            self._buffers[index].append(float(value))

    def _ready(self):
        return all(len(buffer) == self.window_size for buffer in self._buffers)

    def _get_window(self):
        return np.array([list(buffer) for buffer in self._buffers], dtype=np.float64)

    def _extract_features(self, window):
        n_channels = self.n_channels
        n_pairs = len(self.LEFT_IDX)
        de = np.empty(n_channels * len(self.BANDS), dtype=np.float64)
        dasm = np.empty(n_pairs * len(self.BANDS), dtype=np.float64)
        band_power = {}

        for band_index, (name, (b, a)) in enumerate(self._band_filters.items()):
            filtered = signal.filtfilt(b, a, window, axis=1)
            variance = np.var(filtered, axis=1) + 1e-12
            band_power[name] = float(np.mean(variance))
            de_band = 0.5 * np.log(2 * np.pi * np.e * variance)
            start = band_index * n_channels
            de[start:start + n_channels] = de_band
            pair_start = band_index * n_pairs
            dasm[pair_start:pair_start + n_pairs] = (
                de_band[self.LEFT_IDX] - de_band[self.RIGHT_IDX]
            )

        # Relative band power is carried forward for UI/debug use and for the
        # lightweight 5-way mood mapping layered on top of binary valence.
        total_power = sum(band_power.values()) or 1.0
        relative_power = {
            name: float(value / total_power)
            for name, value in band_power.items()
        }
        return np.concatenate([de, dasm]), relative_power

    def _normalize(self, features):
        calibrated = (features - self._calib_mean) / self._calib_std
        return self.scaler.transform(calibrated.reshape(1, -1))

    def predict(self):
        if not self._ready():
            return None

        features, band_features = self._extract_features(self._get_window())

        if not self.is_calibrated():
            self._calib_features.append(features.copy())
            if len(self._calib_features) >= self._calib_target_windows:
                feature_array = np.array(self._calib_features)
                self._calib_mean = feature_array.mean(axis=0)
                self._calib_std = feature_array.std(axis=0) + 1e-8
            return {
                "type": "status",
                "status": "calibrating",
                "progress": round(self.calibration_progress(), 3),
            }

        normalized = self._normalize(features)
        valence_prob = float(self.valence_model.predict_proba(normalized)[0, 1])
        return {
            "type": "valence",
            "status": "predicting",
            "valence": "positive" if valence_prob > 0.5 else "negative",
            "valence_prob": valence_prob,
            "features": band_features,
        }


class EmotionSmoother:
    APP_EMOTIONS = ("calm", "happy", "angry", "sad", "focused")

    def __init__(
        self,
        min_confidence=0.62,
        history_size=5,
        min_stable_votes=3,
        cooldown_seconds=10.0,
        duplicate_silence_seconds=3.0,
    ):
        self.min_confidence = min_confidence
        self.history = collections.deque(maxlen=history_size)
        self.min_stable_votes = min_stable_votes
        self.cooldown_seconds = cooldown_seconds
        self.duplicate_silence_seconds = duplicate_silence_seconds
        self.last_emitted_emotion = None
        self.last_emitted_at = 0.0
        self.last_status = None

    def process(self, prediction, timestamp):
        if not prediction:
            return None

        if prediction.get("type") == "status":
            status = prediction.get("status")
            if status == "calibrating":
                progress = prediction.get("progress", 0.0)
                rounded = round(progress, 2)
                if rounded != self.last_status:
                    self.last_status = rounded
                    return {
                        "type": "status",
                        "status": "calibrating",
                        "progress": rounded,
                        "timestamp": timestamp,
                    }
            return None

        candidate = self._map_to_emotion(prediction)
        candidate["timestamp"] = timestamp
        self.history.append(candidate)

        # Hackathon-friendly stabilization: require repeated agreement before
        # switching the live mood or restarting music.
        recent = [item for item in self.history if item["confidence"] >= self.min_confidence]
        if not recent:
            return None

        counts = collections.Counter(item["emotion"] for item in recent)
        winner, votes = counts.most_common(1)[0]
        if votes < self.min_stable_votes:
            return None

        now = timestamp
        if winner == self.last_emitted_emotion:
            if now - self.last_emitted_at < self.duplicate_silence_seconds:
                return None
        elif now - self.last_emitted_at < self.cooldown_seconds:
            return None

        winner_candidates = [item for item in recent if item["emotion"] == winner]
        avg_confidence = float(np.mean([item["confidence"] for item in winner_candidates]))
        latest = winner_candidates[-1]
        self.last_emitted_emotion = winner
        self.last_emitted_at = now

        return {
            "type": "emotion",
            "emotion": winner,
            "confidence": round(avg_confidence, 3),
            "timestamp": timestamp,
            "features": {
                name: round(float(value), 3)
                for name, value in latest["features"].items()
            },
            "source": "eeg_model",
        }

    def _map_to_emotion(self, prediction):
        features = prediction.get("features", {})
        alpha = float(features.get("alpha", 0.0))
        beta = float(features.get("beta", 0.0))
        theta = float(features.get("theta", 0.0))
        gamma = float(features.get("gamma", 0.0))

        # The notebook exports binary valence only, so the app-level 5-way
        # labels are derived from valence plus simple arousal/focus heuristics.
        valence_prob = float(prediction["valence_prob"])
        model_confidence = 0.5 + abs(valence_prob - 0.5)
        low_band = alpha + theta + 1e-6
        high_band = beta + gamma + 1e-6
        arousal_ratio = high_band / low_band
        focus_ratio = beta / (alpha + theta + 1e-6)
        calm_ratio = alpha / high_band

        if valence_prob >= 0.5:
            if focus_ratio >= 0.34 and beta >= theta * 0.9:
                emotion = "focused"
                rule_conf = min(0.99, 0.58 + min(focus_ratio, 1.2) * 0.3)
            elif arousal_ratio < 0.9 or calm_ratio >= 1.0:
                emotion = "calm"
                rule_conf = min(0.99, 0.58 + min(calm_ratio, 1.2) * 0.25)
            else:
                emotion = "happy"
                rule_conf = min(0.99, 0.58 + min(arousal_ratio, 1.4) * 0.22)
        else:
            if arousal_ratio >= 0.95 or gamma >= alpha * 0.35:
                emotion = "angry"
                rule_conf = min(0.99, 0.58 + min(arousal_ratio, 1.4) * 0.22)
            else:
                emotion = "sad"
                rule_conf = min(0.99, 0.58 + min(theta + alpha, 1.0) * 0.18)

        confidence = min(0.99, 0.7 * model_confidence + 0.3 * rule_conf)
        return {
            "emotion": emotion,
            "confidence": confidence,
            "features": {
                "alpha": alpha,
                "beta": beta,
                "theta": theta,
                "gamma": gamma,
            },
        }


class EmotionInferenceEngine:
    CHANNELS = RealTimeValenceRecognizer.CHANNELS

    def __init__(self, **kwargs):
        self.recognizer = RealTimeValenceRecognizer(**kwargs)
        self.smoother = EmotionSmoother()

    def push_sample(self, sample):
        self.recognizer.push_sample(sample)

    def predict(self, timestamp=None):
        timestamp = float(timestamp if timestamp is not None else __import__("time").time())
        prediction = self.recognizer.predict()
        return self.smoother.process(prediction, timestamp)


def resolve_artifact_paths(valence_xgb_path=None, scaler_path=None):
    env_model = os.getenv("EEG_VALENCE_MODEL_PATH")
    env_scaler = os.getenv("EEG_SCALER_PATH")
    artifact_dir = os.getenv("EEG_EMOTION_ARTIFACT_DIR")

    candidate_dirs = []
    if artifact_dir:
        candidate_dirs.append(Path(artifact_dir))

    base_dir = Path(__file__).resolve().parent
    candidate_dirs.extend([
        base_dir / "emotion_artifacts",
        base_dir.parent / "emotion_artifacts",
        Path.home() / "eeg_emotion",
    ])

    if not valence_xgb_path:
        if env_model:
            valence_xgb_path = Path(env_model)
        else:
            for directory in candidate_dirs:
                candidate = directory / "models" / "valence_xgb.joblib"
                if candidate.exists():
                    valence_xgb_path = candidate
                    break

    if not scaler_path:
        if env_scaler:
            scaler_path = Path(env_scaler)
        else:
            for directory in candidate_dirs:
                candidate = directory / "artifacts" / "scaler.joblib"
                if candidate.exists():
                    scaler_path = candidate
                    break

    missing = []
    if not valence_xgb_path or not Path(valence_xgb_path).exists():
        missing.append("valence_xgb.joblib")
    if not scaler_path or not Path(scaler_path).exists():
        missing.append("scaler.joblib")
    if missing:
        raise MissingArtifactsError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Set EEG_EMOTION_ARTIFACT_DIR or EEG_VALENCE_MODEL_PATH / EEG_SCALER_PATH."
        )

    return str(valence_xgb_path), str(scaler_path)


def export_runtime_artifacts(valence_model, scaler, artifact_dir=None):
    artifact_root = Path(
        artifact_dir
        or os.getenv("EEG_EMOTION_ARTIFACT_DIR")
        or (Path(__file__).resolve().parent / "emotion_artifacts")
    )
    models_dir = artifact_root / "models"
    artifacts_dir = artifact_root / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "valence_xgb.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    metadata_path = artifact_root / "metadata.json"

    joblib.dump(valence_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "sample_rate": RealTimeValenceRecognizer.FS,
        "window_seconds": RealTimeValenceRecognizer.WINDOW_SECONDS,
        "window_size": RealTimeValenceRecognizer.FS * RealTimeValenceRecognizer.WINDOW_SECONDS,
        "channels": RealTimeValenceRecognizer.CHANNELS,
        "feature_bands": RealTimeValenceRecognizer.BANDS,
        "model_output": {
            "type": "binary_valence",
            "labels": ["negative", "positive"],
        },
        "app_emotion_mapping": list(EmotionSmoother.APP_EMOTIONS),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metadata_path": str(metadata_path),
    }
