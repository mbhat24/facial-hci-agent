"""Cognitive/affective state inference from AU + head + gaze + blink.

Heuristics grounded in HCI/affective-computing literature:
  - Engagement      : stable head + low gaze aversion + some AU1/AU2
  - Confusion       : AU4 + AU7 + head tilt (|roll| > 10)
  - Cognitive load  : AU4 + reduced blink rate (< 10/min)
  - Stress          : elevated blinks (> 25/min) + AU4
  - Boredom         : minimal AU activity + slow blinks + gaze aversion
  - Surprise/interest: AU1/AU2 + slight AU5 + forward head pose
"""
from typing import Dict, Tuple
from ..features.action_units import AU_THRESHOLD
from ..perception.head_pose import head_stable
from ..perception.iris_gaze import gaze_aversion


def classify_cognitive_state(
    aus: Dict[str, float],
    head_pose: Dict[str, float],
    gaze: Dict[str, float],
    blink_rate_per_min: float,
) -> Tuple[str, float]:
    """Return (state_label, confidence)."""
    def on(a): return aus.get(a, 0.0) > AU_THRESHOLD
    roll = head_pose.get("roll", 0.0)
    total_au = sum(aus.values())
    stable = head_stable(head_pose)
    averted = gaze_aversion(gaze)

    # Ordered: most specific first
    if on("AU4") and on("AU7") and abs(roll) > 10:
        return "confused", 0.75
    if on("AU4") and blink_rate_per_min < 10 and blink_rate_per_min > 0:
        return "high_cognitive_load", 0.7
    if blink_rate_per_min > 25 and on("AU4"):
        return "stressed", 0.7
    if total_au < 1.0 and blink_rate_per_min < 8 and averted:
        return "bored_or_disengaged", 0.65
    if stable and (on("AU1") or on("AU2")) and not averted:
        return "engaged_interested", 0.7
    if stable and not averted:
        return "attentive", 0.6
    if averted:
        return "distracted", 0.55
    return "neutral", 0.4
