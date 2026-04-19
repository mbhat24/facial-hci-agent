from src.facial_hci.inference.emotion_rules import (
    classify_emotion, emotion_probabilities,
)


def test_neutral_when_no_aus():
    label, conf = classify_emotion({})
    assert label == "neutral"
    assert conf == 0.0


def test_happiness_from_au6_au12():
    aus = {"AU6": 0.5, "AU12": 0.7}
    label, conf = classify_emotion(aus)
    assert label == "happiness"
    assert conf > 0


def test_sadness_from_au15_au1_au17():
    aus = {"AU15": 0.6, "AU1": 0.4, "AU17": 0.4}
    label, _ = classify_emotion(aus)
    assert label == "sadness"


def test_surprise_requires_au26():
    aus = {"AU1": 0.5, "AU2": 0.5, "AU5": 0.5}   # no AU26
    label, _ = classify_emotion(aus)
    assert label != "surprise"
    aus["AU26"] = 0.5
    label, _ = classify_emotion(aus)
    assert label == "surprise"


def test_probs_sum_to_one_approx():
    probs = emotion_probabilities({"AU12": 0.6, "AU6": 0.4})
    assert abs(sum(probs.values()) - 1.0) < 1e-3
