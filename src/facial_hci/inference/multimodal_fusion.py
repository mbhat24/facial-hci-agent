"""Weighted fusion of visual + audio (+ context) signals.

Fusion weights are configurable. Default face-dominant (0.75 face / 0.25 voice).
"""
from typing import Dict, Optional
from ..features.audio_prosody import ProsodyFeatures


def fuse(
    face_probs: Dict[str, float],
    prosody: Optional[ProsodyFeatures] = None,
    face_weight: float = 0.75,
) -> Dict[str, float]:
    if prosody is None or not prosody.available:
        return face_probs

    # Very simple prosody -> arousal/valence boost (can be replaced with a
    # trained model). High pitch + high energy -> fear/surprise/anger boost.
    voice_probs = {k: 0.0 for k in face_probs}
    if prosody.energy > 0.05 and prosody.pitch_hz > 220:
        for k in ("fear", "surprise", "anger"):
            if k in voice_probs:
                voice_probs[k] += 0.3
    elif prosody.energy < 0.01:
        voice_probs["sadness"] = voice_probs.get("sadness", 0) + 0.2
    # normalize voice side
    s = sum(voice_probs.values()) + 1e-6
    voice_probs = {k: v / s for k, v in voice_probs.items()}

    fused = {k: face_weight * face_probs.get(k, 0)
              + (1 - face_weight) * voice_probs.get(k, 0)
             for k in face_probs}
    tot = sum(fused.values()) + 1e-6
    return {k: v / tot for k, v in fused.items()}
