from src.facial_hci.inference.cognitive_state import classify_cognitive_state


POSE_STABLE = {"pitch": 2, "yaw": 3, "roll": 1}
POSE_TILTED = {"pitch": 2, "yaw": 3, "roll": 15}
GAZE_CENTER = {"gaze_x": 0.0, "gaze_y": 0.0, "available": True}
GAZE_AVERTED = {"gaze_x": 0.7, "gaze_y": 0.1, "available": True}


def test_confused_needs_au4_au7_and_tilt():
    aus = {"AU4": 0.4, "AU7": 0.4}
    label, _ = classify_cognitive_state(aus, POSE_TILTED, GAZE_CENTER, 15)
    assert label == "confused"


def test_high_cognitive_load():
    aus = {"AU4": 0.4}
    label, _ = classify_cognitive_state(aus, POSE_STABLE, GAZE_CENTER, 5)
    assert label == "high_cognitive_load"


def test_stressed_high_blink_plus_au4():
    aus = {"AU4": 0.4}
    label, _ = classify_cognitive_state(aus, POSE_STABLE, GAZE_CENTER, 30)
    assert label == "stressed"


def test_bored_low_activity():
    aus = {"AU1": 0.02}
    label, _ = classify_cognitive_state(aus, POSE_STABLE, GAZE_AVERTED, 4)
    assert label == "bored_or_disengaged"


def test_engaged_interested():
    aus = {"AU1": 0.3, "AU2": 0.3}
    label, _ = classify_cognitive_state(aus, POSE_STABLE, GAZE_CENTER, 15)
    assert label in ("engaged_interested", "attentive")
