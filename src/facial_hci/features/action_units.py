"""FACS Action Units from MediaPipe blendshapes.

Based on Sălăgean, Leba, Ionica (2025) — mapping + weak-signal amplification.
"""
from typing import Dict, List, Deque
from collections import deque
import numpy as np

BLENDSHAPE_TO_AU: Dict[str, List[str]] = {
    "AU1":  ["browInnerUp"],
    "AU2":  ["browOuterUpLeft", "browOuterUpRight"],
    "AU4":  ["browDownLeft", "browDownRight"],
    "AU5":  ["eyeWideLeft", "eyeWideRight"],
    "AU6":  ["cheekSquintLeft", "cheekSquintRight"],
    "AU7":  ["eyeSquintLeft", "eyeSquintRight"],
    "AU9":  ["noseSneerLeft", "noseSneerRight"],
    "AU10": ["mouthUpperUpLeft", "mouthUpperUpRight"],
    "AU12": ["mouthSmileLeft", "mouthSmileRight"],
    "AU14": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17": ["mouthShrugLower"],
    "AU20": ["mouthStretchLeft", "mouthStretchRight"],
    "AU23": ["mouthPressLeft", "mouthPressRight"],
    "AU24": ["mouthPucker"],
    "AU25": ["mouthClose"],
    "AU26": ["jawOpen"],
    "AU28": ["mouthRollLower", "mouthRollUpper"],
    "AU43": ["eyeLookDownLeft", "eyeLookDownRight"],
    "AU45": ["eyeBlinkLeft", "eyeBlinkRight"],
}

AU_AMPLIFICATION = {
    "AU9": 3.0, "AU10": 2.5, "AU12": 3.0, "AU6": 2.0,
    "AU4": 2.0, "AU7": 3.0, "AU20": 3.0, "AU5": 2.0,
    "AU15": 3.0, "AU1": 2.0, "AU17": 2.0, "AU26": 2.5, "AU2": 1.8,
    "AU14": 2.2, "AU23": 2.0, "AU24": 2.0, "AU45": 1.0, "AU43": 1.5,
}

AU_THRESHOLD = 0.15


class TemporalSmoother:
    """Exponential moving average smoother to reduce AU jitter."""
    
    def __init__(self, alpha: float = 0.3, window_size: int = 5):
        """
        Args:
            alpha: Smoothing factor (0-1), lower = more smoothing
            window_size: Number of frames to keep in history for moving average
        """
        self.alpha = alpha
        self.window_size = window_size
        self.history: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self.current_smoothed: Dict[str, float] = {}
    
    def update(self, new_aus: Dict[str, float]) -> Dict[str, float]:
        """
        Update with new AU values and return smoothed values.
        
        Args:
            new_aus: New AU values from current frame
            
        Returns:
            Smoothed AU values
        """
        self.history.append(new_aus.copy())
        
        if not self.current_smoothed:
            # First frame - initialize with current values
            self.current_smoothed = new_aus.copy()
            return self.current_smoothed
        
        # Exponential moving average: smoothed = alpha * new + (1-alpha) * old
        for au, value in new_aus.items():
            if au in self.current_smoothed:
                self.current_smoothed[au] = self.alpha * value + (1 - self.alpha) * self.current_smoothed[au]
            else:
                self.current_smoothed[au] = value
        
        # Ensure all AUs are present
        for au in self.current_smoothed.keys():
            if au not in new_aus:
                self.current_smoothed[au] = self.current_smoothed[au] * (1 - self.alpha)
        
        return self.current_smoothed
    
    def reset(self):
        """Reset smoothing history."""
        self.history.clear()
        self.current_smoothed.clear()


def extract_action_units(blendshapes) -> Dict[str, float]:
    """blendshapes: list of Category objects from MediaPipe."""
    bs = {b.category_name: b.score for b in blendshapes}
    out: Dict[str, float] = {}
    for au, names in BLENDSHAPE_TO_AU.items():
        vals = [bs.get(n, 0.0) for n in names]
        score = float(np.mean(vals)) if vals else 0.0
        score *= AU_AMPLIFICATION.get(au, 1.0)
        out[au] = float(min(1.0, score))
    return out


def active_aus(aus: Dict[str, float], thresh: float = AU_THRESHOLD) -> set:
    return {k for k, v in aus.items() if v > thresh}
