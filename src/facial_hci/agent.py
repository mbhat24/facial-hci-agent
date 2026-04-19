"""FacialHCIAgent — orchestrates perception, features, inference, reasoning."""
from __future__ import annotations
import time
import json
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional
import numpy as np

from .perception.face_mesh import FacePerception
from .perception.head_pose import head_pose_from_matrix
from .perception.iris_gaze import estimate_gaze
from .features.action_units import extract_action_units, AU_THRESHOLD
from .features.blink_analysis import BlinkTracker
from .features.micro_expressions import MicroExpressionDetector, MicroEvent
from .features.audio_prosody import ProsodyFeatures
from .inference.emotion_rules import classify_emotion, emotion_probabilities
from .inference.cognitive_state import classify_cognitive_state
from .inference.multimodal_fusion import fuse
from .inference.llm_reasoner import LLMReasoner
from .logging_utils import get_logger
from .config import settings

log = get_logger(__name__)


@dataclass
class AnalysisResult:
    ts: float
    face_detected: bool = False
    action_units: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, float] = field(default_factory=dict)
    gaze: Dict[str, float] = field(default_factory=dict)
    blink: bool = False
    blink_rate_per_min: float = 0.0
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    emotion_probs: Dict[str, float] = field(default_factory=dict)
    cognitive_state: str = "neutral"
    cognitive_confidence: float = 0.0
    micro_events: List[str] = field(default_factory=list)
    thought_inference: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class FacialHCIAgent:
    def __init__(self,
                 session_id: str = "default",
                 personalization=None,
                 log_dir: Optional[Path] = None):
        self.session_id = session_id
        try:
            self.perception = FacePerception()
            self.blink_tracker = BlinkTracker()
            self.micro_detector = MicroExpressionDetector()
            self.reasoner = LLMReasoner()
        except Exception as e:
            log.error(f"Failed to initialize agent components: {e}")
            raise
        
        self.history: Deque[str] = deque(maxlen=20)
        self.micro_history: Deque[str] = deque(maxlen=20)
        self.personalization = personalization   # optional PersonalProfile

        self._t0 = time.time()
        self.log_dir = log
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._error_count = 0
        self._max_errors = 10  # Max consecutive errors before disabling

    def analyze(self, frame_bgr: np.ndarray) -> AnalysisResult:
        """Run full pipeline on one BGR frame."""
        try:
            self._frame_count += 1
            self._error_count = 0  # Reset on success
            timestamp_ms = int((time.time() - self._t0) * 1000)
            result = AnalysisResult(ts=time.time())

            # 1. Perception
            mp_result = self.perception.process(frame_bgr, timestamp_ms)
            if not mp_result.face_landmarks:
                return result
            result.face_detected = True

            landmarks = mp_result.face_landmarks[0]
            blendshapes = mp_result.face_blendshapes[0] if mp_result.face_blendshapes else []

            # 2. Features
            result.action_units = extract_action_units(blendshapes)
            if mp_result.facial_transformation_matrixes:
                result.head_pose = head_pose_from_matrix(
                    mp_result.facial_transformation_matrixes[0])
            result.gaze = estimate_gaze(landmarks)

            # Apply personalization if available
            if self.personalization:
                result.action_units = self.personalization.adjust_aus(result.action_units)

            # 3. Temporal features
            au45 = result.action_units.get("AU45", 0.0)
            result.blink = self.blink_tracker.update(au45)
            result.blink_rate_per_min = self.blink_tracker.rate_per_minute()

            micro_events = self.micro_detector.add_frame(result.action_units)
            result.micro_events = [f"{e.au} ({e.duration_ms:.0f}ms)" for e in micro_events]
            for me in result.micro_events:
                self.micro_history.append(me)

            # 4. Inference
            result.emotion, result.emotion_confidence = classify_emotion(result.action_units)
            result.emotion_probs = emotion_probabilities(result.action_units)

            result.cognitive_state, result.cognitive_confidence = classify_cognitive_state(
                result.action_units,
                result.head_pose,
                result.gaze,
                result.blink_rate_per_min,
            )

            # 5. LLM reasoning (throttled)
            if self.reasoner.available():
                try:
                    thought = self.reasoner.reason(
                        active_aus=result.action_units,
                        head_pose=result.head_pose,
                        gaze=result.gaze,
                        emotion=result.emotion,
                        cognitive=result.cognitive_state,
                        history=list(self.history),
                        micro_events=list(self.micro_history),
                    )
                    if thought:
                        result.thought_inference = thought
                        self.history.append(thought)
                except Exception as e:
                    log.warning(f"LLM reasoning failed for session {self.session_id}: {e}")

            # 6. Logging (optional)
            if self.log_dir and self._frame_count % 30 == 0:
                log_path = self.log_dir / f"{self.session_id}.jsonl"
                with open(log_path, "a") as f:
                    f.write(result.to_json() + "\n")

            return result
            
        except Exception as e:
            self._error_count += 1
            log.error(f"Analysis error for session {self.session_id} (frame {self._frame_count}): {e}")
            if self._error_count >= self._max_errors:
                log.error(f"Too many consecutive errors ({self._error_count}), disabling analysis")
            # Return neutral result on error
            return AnalysisResult(ts=time.time(), emotion="neutral")

    def reset_baseline(self):
        self.micro_detector.reset_baseline()
        log.info("Micro-expression baseline reset.")

    def close(self):
        self.perception.close()
        log.info("Agent closed.")
