from types import SimpleNamespace
from src.facial_hci.features.action_units import (
    extract_action_units, active_aus, AU_THRESHOLD, BLENDSHAPE_TO_AU,
)


def _bs(d):
    return [SimpleNamespace(category_name=k, score=v) for k, v in d.items()]


def test_empty_blendshapes_gives_low_aus():
    aus = extract_action_units(_bs({}))
    assert all(v == 0.0 for v in aus.values())
    assert set(aus.keys()) == set(BLENDSHAPE_TO_AU.keys())


def test_smile_activates_au12():
    aus = extract_action_units(_bs({"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}))
    assert aus["AU12"] > AU_THRESHOLD


def test_brow_down_activates_au4():
    aus = extract_action_units(_bs({"browDownLeft": 0.4, "browDownRight": 0.4}))
    assert aus["AU4"] > AU_THRESHOLD


def test_active_aus_filter():
    aus = {"AU1": 0.5, "AU2": 0.05, "AU3": 0.2}
    assert active_aus(aus) == {"AU1", "AU3"}


def test_au_scores_are_clipped():
    aus = extract_action_units(_bs({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.95}))
    assert 0.0 <= aus["AU12"] <= 1.0
