"""MediaPipe FaceLandmarker wrapper: landmarks + blendshapes + transform."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = settings.model_dir / "face_landmarker.task"


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    log.info("Downloading MediaPipe face_landmarker model ...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info(f"Saved to {MODEL_PATH}")
    return MODEL_PATH


class FacePerception:
    """Wraps MediaPipe tasks API for video-mode inference."""

    def __init__(self, num_faces: int = 1):
        ensure_model()
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        Options = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        opts = Options(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(opts)
        log.info("FacePerception ready.")

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_img, timestamp_ms)

    def close(self):
        self.landmarker.close()
