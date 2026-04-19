"""MediaPipe FaceLandmarker wrapper: landmarks + blendshapes + transform."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp
import os
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = settings.model_dir / "face_landmarker.task"

# Force CPU-only mode to avoid OpenGL dependency on cloud platforms
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    log.info("Downloading MediaPipe face_landmarker model ...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info(f"Saved to {MODEL_PATH}")
    return MODEL_PATH


class FacePerception:
    """Wraps MediaPipe tasks API for video-mode inference with CPU-only mode."""

    def __init__(self, num_faces: int = 1):
        ensure_model()
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        Options = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        # Use CPU delegate to avoid OpenGL dependency
        base_opts = BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=BaseOptions.Delegate.CPU  # Force CPU-only mode
        )

        opts = Options(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        try:
            self.landmarker = FaceLandmarker.create_from_options(opts)
            log.info("FacePerception ready (CPU-only mode).")
        except Exception as e:
            log.error(f"Failed to create FaceLandmarker: {e}")
            # Fallback: try without delegate (may use GPU if available)
            base_opts_fallback = BaseOptions(model_asset_path=str(MODEL_PATH))
            opts_fallback = Options(
                base_options=base_opts_fallback,
                running_mode=RunningMode.VIDEO,
                num_faces=num_faces,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = FaceLandmarker.create_from_options(opts_fallback)
            log.info("FacePerception ready (fallback mode).")

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_img, timestamp_ms)

    def close(self):
        self.landmarker.close()
