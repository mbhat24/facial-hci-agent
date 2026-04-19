from src.facial_hci.training.personalize import PersonalProfile


def test_baseline_updates():
    p = PersonalProfile(session_id="t")
    for _ in range(40):
        p.update_baseline({"AU1": 0.2, "AU4": 0.1})
    assert abs(p.baseline_mean["AU1"] - 0.2) < 1e-6
    assert p.frame_count == 40


def test_adjust_reduces_au_after_warmup():
    p = PersonalProfile(session_id="t")
    for _ in range(50):
        p.update_baseline({"AU4": 0.3})
    adj = p.adjust_aus({"AU4": 0.3})
    assert adj["AU4"] < 0.3
