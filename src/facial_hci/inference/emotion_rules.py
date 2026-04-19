"""FACS-based hierarchical emotion classifier.

Based on Ekman & Friesen (1978) prototypes and the hybrid rule system from
Sălăgean et al. (2025) which reports 93.3% on CASME II.
"""
from typing import Dict, Tuple, List
from ..features.action_units import AU_THRESHOLD, active_aus

EMOTION_RULES: Dict[str, Dict[str, List[str]]] = {
    "happiness": {"must": ["AU12"],            "should": ["AU6"]},
    "sadness":   {"must": ["AU15"],            "should": ["AU1", "AU17"]},
    "fear":      {"must": ["AU20"],            "should": ["AU1", "AU2", "AU4", "AU5"]},
    "surprise":  {"must": ["AU26"],            "should": ["AU1", "AU2", "AU5"]},
    "disgust":   {"must": ["AU9"],             "should": ["AU10", "AU15"]},
    "anger":     {"must": ["AU4"],             "should": ["AU5", "AU7", "AU23"]},
    "contempt":  {"must": ["AU14"],            "should": ["AU24"]},
}


def classify_emotion(aus: Dict[str, float],
                     threshold: float = AU_THRESHOLD) -> Tuple[str, float]:
    """Return (emotion, confidence in [0,1])."""
    active = active_aus(aus, threshold)
    best_label, best_score = "neutral", 0.0
    for label, rule in EMOTION_RULES.items():
        must = set(rule["must"])
        should = set(rule["should"])
        if not must.issubset(active):
            continue
        m_sum = sum(aus[a] for a in must)
        s_sum = sum(aus[a] for a in should if a in active)
        denom = len(must) + 0.5 * len(should)
        score = (m_sum + 0.5 * s_sum) / max(denom, 1e-6)
        if score > best_score:
            best_label, best_score = label, score
    return best_label, float(min(1.0, best_score))


def emotion_probabilities(aus: Dict[str, float]) -> Dict[str, float]:
    """Softer distribution — useful for charts."""
    active = active_aus(aus)
    probs = {"neutral": 0.2}
    for label, rule in EMOTION_RULES.items():
        must = set(rule["must"])
        should = set(rule["should"])
        m_hit = sum(aus.get(a, 0) for a in must if a in active)
        s_hit = sum(aus.get(a, 0) for a in should if a in active)
        probs[label] = float(m_hit + 0.4 * s_hit)
    # normalize
    total = sum(probs.values()) + 1e-6
    return {k: v / total for k, v in probs.items()}
