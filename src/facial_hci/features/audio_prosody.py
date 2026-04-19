"""Audio prosody features (stub).

For browser-hosted deployments, the browser can capture audio via
getUserMedia and send PCM chunks over WebSocket. This stub defines the
interface; a full implementation would use librosa/praat-parselmouth
to extract pitch, jitter, shimmer, energy.

Kept minimal to stay inside Render free-tier memory.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ProsodyFeatures:
    pitch_hz: float = 0.0
    energy: float = 0.0
    jitter: float = 0.0
    available: bool = False


def extract_prosody(pcm_chunk: Optional[np.ndarray], sr: int = 16000) -> ProsodyFeatures:
    if pcm_chunk is None or len(pcm_chunk) < 400:
        return ProsodyFeatures()
    energy = float(np.sqrt(np.mean(pcm_chunk.astype(np.float32) ** 2)))
    # naive zero-crossing-based pitch estimate
    zc = np.where(np.diff(np.signbit(pcm_chunk)))[0]
    if len(zc) > 1:
        period = np.mean(np.diff(zc)) * 2
        pitch = sr / period if period > 0 else 0.0
    else:
        pitch = 0.0
    return ProsodyFeatures(
        pitch_hz=float(pitch), energy=energy, jitter=0.0, available=True)
