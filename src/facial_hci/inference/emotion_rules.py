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
    """Return probability distribution over emotions."""
    scores = {}
    for emotion, rules in EMOTION_RULES.items():
        score = 0.0
        for au, weight in rules.items():
            score += aus.get(au, 0.0) * weight
        scores[emotion] = score
    total = sum(scores.values()) or 1.0
    return {e: s / total for e, s in scores.items()}


def emotion_confidence_interval(aus: Dict[str, float], samples: int = 10) -> Dict[str, tuple]:
    """
    Estimate confidence intervals for emotion probabilities using bootstrap.
    
    Args:
        aus: Action unit values
        samples: Number of bootstrap samples
        
    Returns:
        Dict mapping emotions to (lower_bound, upper_bound) tuples
    """
    probs_list = []
    for _ in range(samples):
        # Add small noise to simulate measurement uncertainty
        noisy_aus = {k: max(0, min(1, v + np.random.normal(0, 0.05))) 
                     for k, v in aus.items()}
        probs = emotion_probabilities(noisy_aus)
        probs_list.append(probs)
    
    # Calculate confidence intervals
    intervals = {}
    for emotion in EMOTION_RULES.keys():
        emotion_probs = [p[emotion] for p in probs_list]
        lower = np.percentile(emotion_probs, 5)  # 5th percentile
        upper = np.percentile(emotion_probs, 95)  # 95th percentile
        intervals[emotion] = (float(lower), float(upper))
    
    return intervals
