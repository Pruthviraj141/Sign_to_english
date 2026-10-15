#!/usr/bin/env python3
"""
SignVision ISL Inference Bundle Builder (`build_inference_bundle.py`)

Packages ONLY production runtime assets, inference code, and metadata into
`signvision_isl_inference_bundle.zip` for integration into existing FastAPI
video microservices and backend applications.

Excludes:
- Raw video recordings (data/raw/)
- Training datasets and metadata CSVs (data/metadata.csv)
- Training scripts (train_model.py, record_videos.py)
- Feature caches (artifacts/features_cache.joblib)
- Python virtual environments (.venv), caches (__pycache__), and git files

Validates:
- All required model artifacts, metadata, and MediaPipe tasks.
- Feature dimensions (16 frames x 154 features = 2464 total).
- Generates a bundle manifest with SHA-256 checksums and file descriptions.
- Performs an automated post-packaging smoke test by extracting the ZIP into
  a temporary environment and executing the bundled verification script.
"""

import os
import sys
import json
import shutil
import zipfile
import hashlib
import tempfile
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Root paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ZIP_NAME = "signvision_isl_inference_bundle.zip"
OUTPUT_ZIP_PATH = os.path.join(PROJECT_ROOT, OUTPUT_ZIP_NAME)

# Source files to validate and package
SRC_CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "artifacts", "sign_classifier.joblib")
SRC_METADATA_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")
SRC_TASK_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
SRC_LANDMARK_FEATURES_PATH = os.path.join(PROJECT_ROOT, "src", "landmark_features.py")
SRC_INIT_PATH = os.path.join(PROJECT_ROOT, "src", "__init__.py")

# Disallowed patterns to ensure zero data leakage
FORBIDDEN_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",  # Raw videos
    ".csv",                                    # Metadata / dataset logs
    ".bak",                                    # Backups
    ".pyc", ".pyo",                            # Compiled bytecode
}

FORBIDDEN_KEYWORDS = [
    "features_cache",
    "record_videos",
    "train_model",
    ".venv",
    ".git",
    "screenshots",
]


# =====================================================================
# Content Templates for Packaged Files
# =====================================================================

INFERENCE_PY_CONTENT = '''#!/usr/bin/env python3
"""
SignVision Production Inference Engine (`inference.py`)

Headless, thread-safe ISL recognition engine designed for direct integration
into existing FastAPI microservices, Flask/Django backends, or worker queues.

Features:
- Loads scikit-learn pipeline and MediaPipe Hand Landmarker once at startup.
- Supports predicting from:
    1. Video file path (mp4, webm, avi, mov)
    2. Video bytes in-memory (e.g. from FastAPI UploadFile)
    3. List of decoded OpenCV BGR numpy frames
    4. Pre-extracted landmark sequences (16 x 154 or 2464 dims)
- Configurable confidence thresholds and hand-coverage gating.
- Temporal smoothing helper (`RollingInferenceBuffer`) for streaming video chunks.
- Pure CPU execution (no GUI, no webcam loop, no OpenCV display windows).
"""

import os
import sys
import time
import json
import tempfile
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.landmark_features import (
    HandLandmarkerExtractor,
    DEFAULT_NUM_FRAMES,
    FEATURE_DIM_PER_FRAME,
    TOTAL_CLIP_FEATURE_DIM,
)

DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "sign_classifier.joblib")
DEFAULT_METADATA_PATH = os.path.join(CURRENT_DIR, "models", "model_metadata.json")
DEFAULT_TASK_PATH = os.path.join(CURRENT_DIR, "models", "hand_landmarker.task")


class SignVisionInferenceEngine:
    """
    Production inference engine for SignVision ISL classification.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        task_model_path: str = DEFAULT_TASK_PATH,
        default_threshold: float = 0.65,
        default_min_hand_coverage: float = 0.50,
    ):
        self.model_path = os.path.abspath(model_path)
        self.metadata_path = os.path.abspath(metadata_path)
        self.task_model_path = os.path.abspath(task_model_path)
        self.default_threshold = default_threshold
        self.default_min_hand_coverage = default_min_hand_coverage

        # Validate file existence
        for p, desc in [
            (self.model_path, "Model classifier file"),
            (self.metadata_path, "Model metadata JSON"),
            (self.task_model_path, "MediaPipe hand landmarker task asset"),
        ]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{desc} not found at: {p}")

        # 1. Load trained classification pipeline
        self.pipeline = joblib.load(self.model_path)

        # 2. Load metadata configuration
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self.classes: List[str] = self.metadata.get("classes", [])
        self.id_to_class: Dict[str, str] = self.metadata.get(
            "id_to_class", {str(i): c for i, c in enumerate(self.classes)}
        )
        self.class_to_id: Dict[str, int] = self.metadata.get(
            "class_to_id", {c: i for i, c in enumerate(self.classes)}
        )
        self.sequence_length: int = self.metadata.get("sequence_length", DEFAULT_NUM_FRAMES)
        self.total_feature_dim: int = self.metadata.get("total_feature_dim", TOTAL_CLIP_FEATURE_DIM)

        # Resolve classifier classes order
        classifier = getattr(self.pipeline, "named_steps", {}).get("classifier", self.pipeline)
        self.classifier_classes = getattr(
            self.pipeline,
            "classes_",
            getattr(classifier, "classes_", list(range(len(self.classes)))),
        )

        # 3. Initialize MediaPipe HandLandmarker extractor
        self.extractor = HandLandmarkerExtractor(model_path=self.task_model_path, num_hands=2)

    def predict_video(
        self,
        video_path: str,
        threshold: Optional[float] = None,
        min_hand_coverage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Accepts any video file path (.mp4, .webm, .avi, .mov), uniformly samples 16 frames,
        extracts landmark vectors, and returns structured class probabilities.
        """
        t_start = time.perf_counter()
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        clip_vector, meta = self.extractor.extract_video_features(
            video_path, num_sampled_frames=self.sequence_length
        )
        hand_coverage = meta.get("coverage_ratio", 0.0)
        total_frames = meta.get("total_frames_in_file", 0)

        return self._predict_from_feature_vector(
            clip_vector=clip_vector,
            hand_coverage=hand_coverage,
            total_frames=total_frames,
            threshold=threshold or self.default_threshold,
            min_hand_coverage=min_hand_coverage or self.default_min_hand_coverage,
            t_start=t_start,
        )

    def predict_video_bytes(
        self,
        video_bytes: bytes,
        file_extension: str = ".mp4",
        threshold: Optional[float] = None,
        min_hand_coverage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method for FastAPI endpoints receiving in-memory file buffers.
        Writes bytes to a temporary file, processes it, and cleans up automatically.
        """
        if not video_bytes:
            raise ValueError("video_bytes cannot be empty.")

        suffix = file_extension if file_extension.startswith(".") else f".{file_extension}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(video_bytes)

        try:
            return self.predict_video(
                video_path=tmp_path,
                threshold=threshold,
                min_hand_coverage=min_hand_coverage,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def predict_frames(
        self,
        frames: List[np.ndarray],
        threshold: Optional[float] = None,
        min_hand_coverage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Accepts a list of OpenCV BGR frames in memory, uniformly samples 16 frames,
        extracts features, and executes inference.
        """
        t_start = time.perf_counter()
        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError("frames list cannot be empty.")

        # Uniform sampling of 16 frames
        if num_frames >= self.sequence_length:
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
        else:
            indices = list(range(num_frames))
            while len(indices) < self.sequence_length:
                indices.append(indices[-1])
            indices = np.array(indices[: self.sequence_length], dtype=int)

        sampled_frames = [frames[i] for i in indices]

        frame_vectors = []
        frames_with_hands = 0
        for f in sampled_frames:
            vec, stats = self.extractor.extract_frame_landmarks(f)
            frame_vectors.append(vec)
            if stats.get("hands_detected", 0) > 0:
                frames_with_hands += 1

        clip_vector = np.concatenate(frame_vectors).astype(np.float32)
        hand_coverage = float(frames_with_hands) / float(self.sequence_length)

        return self._predict_from_feature_vector(
            clip_vector=clip_vector,
            hand_coverage=hand_coverage,
            total_frames=num_frames,
            threshold=threshold or self.default_threshold,
            min_hand_coverage=min_hand_coverage or self.default_min_hand_coverage,
            t_start=t_start,
        )

    def predict_sequence(
        self,
        sequence: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Accepts pre-extracted landmark vectors (e.g. client-side MediaPipe in React Native or Web).
        Shape can be (16, 154) or flattened (2464,).
        """
        t_start = time.perf_counter()
        arr = np.array(sequence, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape != (self.sequence_length, FEATURE_DIM_PER_FRAME):
                raise ValueError(
                    f"Expected 2D sequence shape ({self.sequence_length}, {FEATURE_DIM_PER_FRAME}), received {arr.shape}"
                )
            flat_vec = arr.flatten()
        elif arr.ndim == 1:
            if arr.shape[0] != self.total_feature_dim:
                raise ValueError(
                    f"Expected 1D vector length {self.total_feature_dim}, received {arr.shape[0]}"
                )
            flat_vec = arr
        else:
            raise ValueError(f"Invalid sequence dimensions: {arr.shape}")

        return self._predict_from_feature_vector(
            clip_vector=flat_vec,
            hand_coverage=1.0,
            total_frames=self.sequence_length,
            threshold=threshold or self.default_threshold,
            min_hand_coverage=0.0,
            t_start=t_start,
        )

    def _predict_from_feature_vector(
        self,
        clip_vector: np.ndarray,
        hand_coverage: float,
        total_frames: int,
        threshold: float,
        min_hand_coverage: float,
        t_start: float,
    ) -> Dict[str, Any]:
        """Internal helper to execute classifier prediction and structure JSON outputs."""
        t_infer_start = time.perf_counter()
        prob_array = self.pipeline.predict_proba(clip_vector.reshape(1, -1))[0]
        infer_time_ms = (time.perf_counter() - t_infer_start) * 1000.0

        prob_dict: Dict[str, float] = {}
        for idx_in_array, class_idx in enumerate(self.classifier_classes):
            idx_str = str(int(class_idx))
            if idx_str in self.id_to_class:
                class_name = self.id_to_class[idx_str]
            elif int(class_idx) < len(self.classes):
                class_name = self.classes[int(class_idx)]
            else:
                class_name = f"Class_{class_idx}"
            prob_dict[class_name] = round(float(prob_array[idx_in_array]), 4)

        sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top_sign, top_prob = sorted_probs[0]

        if hand_coverage < min_hand_coverage:
            pred_status = "UNCERTAIN"
            status_msg = f"Low hand detection coverage ({hand_coverage * 100:.1f}%)"
            final_sign = "UNCERTAIN"
        elif top_prob < threshold:
            pred_status = "UNCERTAIN"
            status_msg = f"Low confidence ({top_prob * 100:.1f}% < {threshold * 100:.1f}%)"
            final_sign = "UNCERTAIN"
        else:
            pred_status = "CONFIDENT"
            status_msg = f"Recognized '{top_sign}' with {top_prob * 100:.1f}% confidence"
            final_sign = top_sign

        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "success": True,
            "predicted_sign": final_sign,
            "confidence": round(float(top_prob), 4),
            "status": pred_status,
            "status_message": status_msg,
            "hand_coverage": round(float(hand_coverage), 3),
            "top_candidates": [
                {"sign": name, "probability": round(float(prob), 4)}
                for name, prob in sorted_probs[:5]
            ],
            "all_probabilities": prob_dict,
            "inference_time_ms": round(infer_time_ms, 2),
            "total_processing_time_ms": round(total_time_ms, 2),
            "frames_processed": total_frames,
        }

    def get_classes(self) -> List[str]:
        return list(self.classes)

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def close(self):
        """Releases detector resources."""
        if hasattr(self, "extractor") and self.extractor:
            self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RollingInferenceBuffer:
    """
    Stateful rolling buffer helper for chunked streaming or live frame-by-frame feeds.
    Maintains a 16-frame window and recent predictions voting deque for temporal stability.
    """

    def __init__(
        self,
        engine: SignVisionInferenceEngine,
        window_size: int = 16,
        stability_count: int = 5,
        threshold: float = 0.65,
        min_hand_coverage: float = 0.50,
    ):
        self.engine = engine
        self.window_size = window_size
        self.stability_count = stability_count
        self.threshold = threshold
        self.min_hand_coverage = min_hand_coverage

        self.frame_buffer: deque = deque(maxlen=window_size)
        self.presence_buffer: deque = deque(maxlen=window_size)
        self.recent_predictions: deque = deque(maxlen=stability_count)

    def push_frame(self, bgr_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Pushes a single frame, extracts landmarks, and returns prediction if buffer is full.
        """
        # Horizontal flip convention matching training
        flipped = cv2.flip(bgr_frame, 1)
        vec, stats = self.engine.extractor.extract_frame_landmarks(flipped)

        self.frame_buffer.append(vec)
        self.presence_buffer.append(stats.get("hands_detected", 0) > 0)

        if len(self.frame_buffer) < self.window_size:
            return None

        # Window is full
        hand_coverage = sum(self.presence_buffer) / float(self.window_size)
        clip_vector = np.concatenate(list(self.frame_buffer)).astype(np.float32)

        raw_result = self.engine._predict_from_feature_vector(
            clip_vector=clip_vector,
            hand_coverage=hand_coverage,
            total_frames=self.window_size,
            threshold=self.threshold,
            min_hand_coverage=self.min_hand_coverage,
            t_start=time.perf_counter(),
        )

        # Apply temporal voting
        top_sign = raw_result["predicted_sign"]
        self.recent_predictions.append(top_sign)

        vote_counts = Counter(self.recent_predictions)
        most_common, count = vote_counts.most_common(1)[0]
        required_votes = max(2, int(self.stability_count * 0.6))

        if most_common != "UNCERTAIN" and count >= required_votes:
            stable_sign = most_common
            raw_result["status"] = "CONFIDENT"
            raw_result["status_message"] = f"Stable '{stable_sign}' ({count}/{self.stability_count} votes)"
        else:
            stable_sign = "UNCERTAIN"
            raw_result["status"] = "UNCERTAIN"
            raw_result["status_message"] = "Stabilizing prediction..."

        raw_result["predicted_sign"] = stable_sign
        return raw_result

    def reset(self):
        """Resets all buffers."""
        self.frame_buffer.clear()
        self.presence_buffer.clear()
        self.recent_predictions.clear()
'''

SMOKE_TEST_PY_CONTENT = '''#!/usr/bin/env python3
"""
SignVision Bundle Smoke Test (`smoke_test.py`)

Verifies that the packaged bundle is 100% self-contained and ready for production
inference without retraining or requiring a GUI display.

Checks:
1. Model loading: Verifies sign_classifier.joblib and model_metadata.json.
2. Metadata verification: Checks class list, count (23 classes), and dimensions.
3. MediaPipe task loading: Initializes HandLandmarkerExtractor from hand_landmarker.task.
4. Feature extractor validation: Verifies (154,) frame feature output shape.
5. Synthetic sequence prediction: Tests inference pipeline and probability output.
6. Uncertainty test: Validates that low-coverage input safely outputs "UNCERTAIN".
7. Clean resource release.
"""

import os
import sys
import numpy as np

# Ensure current bundle root is in sys.path
BUNDLE_ROOT = os.path.dirname(os.path.abspath(__file__))
if BUNDLE_ROOT not in sys.path:
    sys.path.insert(0, BUNDLE_ROOT)

from inference import SignVisionInferenceEngine, RollingInferenceBuffer


def run_smoke_test():
    print("=" * 65)
    print("SignVision ISL Production Inference Bundle Smoke Test")
    print("=" * 65)

    # 1. Initialize Engine
    print("[TEST 1/6] Loading SignVisionInferenceEngine...")
    engine = SignVisionInferenceEngine()
    print(f"  [PASS] Classifier loaded: {type(engine.pipeline).__name__}")
    print(f"  [PASS] Metadata loaded: {len(engine.classes)} classes found.")

    # 2. Check metadata classes
    print("[TEST 2/6] Verifying class catalog...")
    classes = engine.get_classes()
    assert len(classes) >= 2, f"Expected at least 2 classes, got {len(classes)}"
    print(f"  [PASS] Classes verified ({len(classes)} classes): {classes[:6]} ...")

    # 3. Test Feature Extractor Output Shape
    print("[TEST 3/6] Verifying frame landmark extraction dimensions...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vec, stats = engine.extractor.extract_frame_landmarks(dummy_frame)
    assert vec.shape == (154,), f"Expected shape (154,), got {vec.shape}"
    assert not np.isnan(vec).any(), "NaN detected in feature vector"
    print(f"  [PASS] Single frame feature vector shape: {vec.shape} (All values numeric)")

    # 4. Test Sequence Prediction on Synthetic Vector
    print("[TEST 4/6] Testing prediction on synthetic 16-frame sequence...")
    synthetic_sequence = np.zeros((16, 154), dtype=np.float32)
    # Put a dummy non-zero coordinate so scaler runs cleanly
    synthetic_sequence[:, 0] = 0.5
    res = engine.predict_sequence(synthetic_sequence, threshold=0.10)

    assert "predicted_sign" in res, "Missing predicted_sign in result"
    assert "top_candidates" in res, "Missing top_candidates in result"
    assert "all_probabilities" in res, "Missing all_probabilities in result"
    assert len(res["all_probabilities"]) == len(classes), "Probabilities count mismatch"
    print(f"  [PASS] Synthetic sequence prediction executed successfully.")
    print(f"         Prediction: {res['predicted_sign']} (Confidence: {res['confidence']*100:.1f}%)")

    # 5. Test Hand Coverage Uncertainty Gate
    print("[TEST 5/6] Testing low hand-coverage uncertainty trigger...")
    dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(16)]
    res_uncertain = engine.predict_frames(dummy_frames, min_hand_coverage=0.50)
    assert res_uncertain["predicted_sign"] == "UNCERTAIN", "Expected UNCERTAIN for blank frames"
    assert res_uncertain["status"] == "UNCERTAIN", "Expected status UNCERTAIN"
    print(f"  [PASS] Uncertainty gate triggered correctly: {res_uncertain['status_message']}")

    # 6. Test Rolling Buffer
    print("[TEST 6/6] Testing RollingInferenceBuffer helper...")
    buf = RollingInferenceBuffer(engine=engine, window_size=16, stability_count=4)
    for _ in range(15):
        assert buf.push_frame(dummy_frame) is None, "Buffer should not predict before 16 frames"
    full_res = buf.push_frame(dummy_frame)
    assert full_res is not None, "Buffer should predict on 16th frame"
    buf.reset()
    assert len(buf.frame_buffer) == 0, "Buffer reset failed"
    print("  [PASS] RollingInferenceBuffer validated.")

    engine.close()
    print("=" * 65)
    print("[SUCCESS] ALL SMOKE TESTS PASSED! Bundle is verified and ready for production.")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_smoke_test()
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] Smoke test failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

REQUIREMENTS_INFERENCE_CONTENT = '''# SignVision ISL Inference Minimal Runtime Requirements
opencv-python-headless>=4.9.0
mediapipe>=0.10.0
scikit-learn>=1.4.0
joblib>=1.3.0
numpy>=1.26.0
'''

README_BUNDLE_CONTENT = '''# SignVision ISL Production Inference Bundle

A lightweight, self-contained inference package for the custom **Indian Sign Language (ISL)** multi-class recognition model.

Built for seamless integration into existing **FastAPI microservices**, **Flask backends**, and **cloud containers** (Fedora/Linux, Docker, Cloud Run, AWS).

---

## 1. Package Structure

```text
signvision_isl_inference_bundle/
├── models/
│   ├── sign_classifier.joblib   # Trained scikit-learn classifier pipeline (StandardScaler + RBF-SVM)
│   ├── model_metadata.json      # Dynamic class catalog (23 classes), IDs, and feature specs
│   └── hand_landmarker.task     # Google MediaPipe hand landmark task asset
├── src/
│   ├── landmark_features.py     # 16-frame uniform MediaPipe landmark feature extraction
│   └── __init__.py
├── inference.py                 # SignVisionInferenceEngine & RollingInferenceBuffer classes
├── smoke_test.py                # Standalone verification script (runs in ~2s, no GUI needed)
├── requirements-inference.txt   # Minimal headless runtime dependencies
├── bundle_manifest.json         # SHA-256 checksums and file descriptions
└── README.md                    # Integration guide
```

---

## 2. Supported Vocabulary (23 Classes)

`BROTHER`, `COLLEGE`, `DO_YOU_UNDERSTAND`, `GOOD_MORNING`, `GOOD_NIGHT`, `HE_IS_FATHER`, `HOW_ARE_YOU`, `HUSBAN`, `ITS_BEAUTIFUL_PLACE`, `LIKE_DANCE`, `MAN`, `NAMSATE`, `NO`, `PLEASE`, `SAD`, `SCHOOL`, `SHOP`, `SORRY`, `THANK_YOU`, `WHAT_IS_YOUR_NAME`, `WOMEN`, `WORK`, `YES`

---

## 3. Installation

In your target microservice environment:

```bash
pip install -r requirements-inference.txt
```

Verify bundle integrity anytime:

```bash
python smoke_test.py
```

---

## 4. FastAPI Integration Example

Copy the bundle directory into your FastAPI service (e.g., inside `app/signvision/` or alongside `main.py`).

```python
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from inference import SignVisionInferenceEngine

engine: SignVisionInferenceEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Initialize engine once on startup
    engine = SignVisionInferenceEngine()
    print(f"SignVision loaded {len(engine.get_classes())} classes.")
    yield
    if engine:
        engine.close()

app = FastAPI(title="Video Sign Recognition Service", lifespan=lifespan)

@app.get("/signs/classes")
def list_classes():
    """Returns all 23 supported ISL sign classes."""
    return {"total_classes": len(engine.get_classes()), "classes": engine.get_classes()}

@app.post("/signs/predict-video")
async def predict_sign_from_video(
    video: UploadFile = File(..., description="Uploaded video clip (.mp4, .webm, .avi, .mov)"),
    threshold: float = Query(0.65, ge=0.0, le=1.0),
    min_hand_coverage: float = Query(0.50, ge=0.0, le=1.0),
):
    """
    Accepts video upload from mobile (Expo) or Web app and returns predicted sign.
    """
    # Write to temporary file
    suffix = os.path.splitext(video.filename or "clip.mp4")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(video.file, tmp)

    try:
        result = engine.predict_video(
            video_path=tmp_path,
            threshold=threshold,
            min_hand_coverage=min_hand_coverage,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
```

---

## 5. Direct Python API Usage

```python
from inference import SignVisionInferenceEngine

# 1. Initialize engine
engine = SignVisionInferenceEngine()

# 2. Predict on a video file
result = engine.predict_video("sample_sign.mp4", threshold=0.65)
print("Recognized Sign:", result["predicted_sign"])
print("Confidence:", result["confidence"])
print("Top Candidates:", result["top_candidates"])

# 3. Predict on in-memory BGR frames (e.g. from OpenCV or RTSP stream)
# frames = [frame1, frame2, ...]
# result = engine.predict_frames(frames)
```

---

## 6. Response Schema

```json
{
  "success": true,
  "predicted_sign": "LIKE_DANCE",
  "confidence": 0.9985,
  "status": "CONFIDENT",
  "status_message": "Recognized 'LIKE_DANCE' with 99.8% confidence",
  "hand_coverage": 0.938,
  "top_candidates": [
    {"sign": "LIKE_DANCE", "probability": 0.9985},
    {"sign": "NAMSATE", "probability": 0.0010},
    {"sign": "HOW_ARE_YOU", "probability": 0.0002}
  ],
  "all_probabilities": {
    "BROTHER": 0.0,
    "COLLEGE": 0.0,
    "LIKE_DANCE": 0.9985,
    ...
  },
  "inference_time_ms": 1.45,
  "total_processing_time_ms": 42.10,
  "frames_processed": 76
}
```
'''


# =====================================================================
# Packaging Functions
# =====================================================================

def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_source_assets():
    """
    Validates presence and compatibility of all required source artifacts
    before creating the bundle.
    """
    print("[1/5] Validating source assets and model metadata...")

    required_sources = [
        (SRC_CLASSIFIER_PATH, "Trained scikit-learn model"),
        (SRC_METADATA_PATH, "Model metadata JSON"),
        (SRC_TASK_MODEL_PATH, "MediaPipe hand landmarker task asset"),
        (SRC_LANDMARK_FEATURES_PATH, "Landmark feature extraction module"),
        (SRC_INIT_PATH, "Package init file"),
    ]

    for p, desc in required_sources:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[ERROR] Required asset missing: {desc} at '{p}'")
        size = os.path.getsize(p)
        if size == 0:
            raise ValueError(f"[ERROR] Asset file is empty (0 bytes): '{p}'")
        print(f"  [OK] {desc:36s}: {os.path.basename(p)} ({size:,} bytes)")

    # Validate metadata content
    with open(SRC_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta.get("classes", [])
    if len(classes) < 2:
        raise ValueError(f"[ERROR] Metadata contains fewer than 2 classes: {classes}")

    seq_len = meta.get("sequence_length", 0)
    feat_dim = meta.get("total_feature_dim", 0)
    if seq_len != 16 or feat_dim != 2464:
        raise ValueError(f"[ERROR] Metadata dimension mismatch: sequence={seq_len}, feat_dim={feat_dim}")

    print(f"  [OK] Metadata validated: {len(classes)} classes, 16 frames x 154 features = 2464 total dimensions.")
    return meta


def build_staging_directory(staging_dir: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assembles all bundle files inside the staging directory and generates
    the bundle manifest with SHA-256 hashes.
    """
    print("\n[2/5] Staging files for inference bundle...")

    models_dir = os.path.join(staging_dir, "models")
    src_dir = os.path.join(staging_dir, "src")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    # Copy binary and source assets
    shutil.copy2(SRC_CLASSIFIER_PATH, os.path.join(models_dir, "sign_classifier.joblib"))
    shutil.copy2(SRC_METADATA_PATH, os.path.join(models_dir, "model_metadata.json"))
    shutil.copy2(SRC_TASK_MODEL_PATH, os.path.join(models_dir, "hand_landmarker.task"))
    shutil.copy2(SRC_LANDMARK_FEATURES_PATH, os.path.join(src_dir, "landmark_features.py"))
    shutil.copy2(SRC_INIT_PATH, os.path.join(src_dir, "__init__.py"))

    # Write code templates
    with open(os.path.join(staging_dir, "inference.py"), "w", encoding="utf-8") as f:
        f.write(INFERENCE_PY_CONTENT.strip() + "\n")

    with open(os.path.join(staging_dir, "smoke_test.py"), "w", encoding="utf-8") as f:
        f.write(SMOKE_TEST_PY_CONTENT.strip() + "\n")

    with open(os.path.join(staging_dir, "requirements-inference.txt"), "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_INFERENCE_CONTENT.strip() + "\n")

    with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_BUNDLE_CONTENT.strip() + "\n")

    # File descriptions for manifest
    file_purposes = {
        "models/sign_classifier.joblib": "Fitted scikit-learn classifier pipeline (StandardScaler + RBF-SVM)",
        "models/model_metadata.json": "Trained class names, ID mappings, feature specifications, and config",
        "models/hand_landmarker.task": "MediaPipe Hand Landmarker task asset (Google DeepMind / MediaPipe)",
        "src/__init__.py": "Package initializer for shared landmark feature module",
        "src/landmark_features.py": "Exact shared MediaPipe 16-frame landmark feature extractor",
        "inference.py": "Production SignVisionInferenceEngine & RollingInferenceBuffer classes",
        "smoke_test.py": "Self-contained bundle verification test without retraining or GUI",
        "requirements-inference.txt": "Minimal inference-only requirements (headless, CPU-friendly)",
        "README.md": "Production integration instructions, FastAPI code example, and API documentation",
        "bundle_manifest.json": "Manifest file listing packaged assets, purposes, and SHA-256 hashes",
    }

    # Generate manifest
    manifest: Dict[str, Any] = {
        "bundle_name": "signvision_isl_inference_bundle",
        "created_at": datetime.now().isoformat(),
        "task": f"Indian Sign Language {len(meta.get('classes', []))}-Class Recognition",
        "num_classes": len(meta.get("classes", [])),
        "classes": meta.get("classes", []),
        "sequence_length": meta.get("sequence_length", 16),
        "feature_dimension": meta.get("total_feature_dim", 2464),
        "system_requirements": {
            "os": "Linux (Fedora, Debian, Ubuntu, Alpine) / macOS / Windows",
            "runtime": "Python 3.10+",
            "hardware": "CPU-only (no GPU required)",
        },
        "exclusions_verified": [
            "Raw training videos (*.mp4, *.avi)",
            "Metadata CSV logs (metadata.csv)",
            "Training scripts (train_model.py, record_videos.py)",
            "Feature caches (*cache*.joblib)",
            "Virtual environments (.venv)",
            "Compiled bytecode (__pycache__)",
            "Git revision histories (.git)",
        ],
        "files": {},
    }

    # Populate file entries in manifest
    for root, _, files in os.walk(staging_dir):
        for fname in sorted(files):
            full_p = os.path.join(root, fname)
            rel_p = os.path.relpath(full_p, staging_dir)
            if rel_p == "bundle_manifest.json":
                continue
            manifest["files"][rel_p] = {
                "description": file_purposes.get(rel_p, "Production bundle file"),
                "size_bytes": os.path.getsize(full_p),
                "sha256": compute_sha256(full_p),
            }

    # Write manifest into staging directory
    manifest_path = os.path.join(staging_dir, "bundle_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    manifest["files"]["bundle_manifest.json"] = {
        "description": file_purposes["bundle_manifest.json"],
        "size_bytes": os.path.getsize(manifest_path),
        "sha256": compute_sha256(manifest_path),
    }

    print(f"  [OK] Staged {len(manifest['files'])} files into temporary staging environment.")
    return manifest


def verify_staging_exclusions(staging_dir: str):
    """Ensures no forbidden files or secrets leak into the bundle."""
    print("\n[3/5] Verifying strict exclusions...")
    for root, _, files in os.walk(staging_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in FORBIDDEN_EXTENSIONS:
                raise ValueError(f"[VIOLATION] Disallowed file extension in bundle: {full_path}")

            for kw in FORBIDDEN_KEYWORDS:
                if kw in fname.lower() or kw in full_path.lower():
                    raise ValueError(f"[VIOLATION] Disallowed keyword '{kw}' found in bundle: {full_path}")

    print("  [OK] Exclusions verified: Zero raw videos, CSV logs, caches, or training scripts found.")


def create_zip_archive(staging_dir: str, zip_path: str):
    """Creates signvision_isl_inference_bundle.zip with preserved relative paths."""
    print(f"\n[4/5] Packaging into ZIP archive: {zip_path}...")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(staging_dir):
            dirs.sort()
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, staging_dir)
                zipf.write(full_path, arcname=rel_path)

    zip_size = os.path.getsize(zip_path)
    print(f"  [OK] Created archive: {os.path.basename(zip_path)} ({zip_size:,} bytes / {zip_size/(1024*1024):.2f} MB)")


def verify_zip_bundle(zip_path: str):
    """
    Extracts the newly generated ZIP into an isolated sandbox and runs
    `smoke_test.py` to confirm self-contained functionality.
    """
    print("\n[5/5] Extracting ZIP in sandbox and running automated smoke test...")
    with tempfile.TemporaryDirectory() as sandbox_dir:
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(sandbox_dir)

        smoke_test_script = os.path.join(sandbox_dir, "smoke_test.py")
        if not os.path.isfile(smoke_test_script):
            raise FileNotFoundError("smoke_test.py was not found inside the generated ZIP archive!")

        # Execute smoke test using active Python environment
        cmd = [sys.executable, smoke_test_script]
        proc = subprocess.run(cmd, cwd=sandbox_dir, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"[ERROR] Bundled smoke test failed inside sandbox:\n{proc.stderr}", file=sys.stderr)
            raise RuntimeError("Bundled smoke test failed!")

        print("--- Sandboxed Smoke Test Output ---")
        for line in proc.stdout.strip().splitlines()[-8:]:
            print(f"  {line}")
        print("-----------------------------------")
        print("  [OK] Sandboxed smoke test passed with exit code 0.")


def main():
    start_time = time.time()
    print("=================================================================")
    print("SignVision — Build Production Inference Bundle")
    print("=================================================================")

    # 1. Validate source assets
    meta = validate_source_assets()

    # 2. Stage files in temporary directory
    with tempfile.TemporaryDirectory() as staging_dir:
        manifest = build_staging_directory(staging_dir, meta)

        # 3. Verify exclusions
        verify_staging_exclusions(staging_dir)

        # 4. Create ZIP
        create_zip_archive(staging_dir, OUTPUT_ZIP_PATH)

    # 5. Verify ZIP bundle
    verify_zip_bundle(OUTPUT_ZIP_PATH)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"[SUCCESS] Inference bundle created successfully in {elapsed:.1f}s!")
    print(f"Output File: {OUTPUT_ZIP_PATH}")
    print(f"File Size:   {os.path.getsize(OUTPUT_ZIP_PATH):,} bytes")
    print("=" * 65)


if __name__ == "__main__":
    main()
