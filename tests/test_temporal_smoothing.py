"""Tests for temporal smoothing."""
import pytest
from src.facial_hci.features.action_units import TemporalSmoother


def test_temporal_smoother_init():
    """Test TemporalSmoother initialization."""
    smoother = TemporalSmoother(alpha=0.3, window_size=5)
    assert smoother.alpha == 0.3
    assert smoother.window_size == 5
    assert len(smoother.history) == 0
    assert len(smoother.current_smoothed) == 0


def test_temporal_smoother_first_frame():
    """Test that first frame is returned as-is."""
    smoother = TemporalSmoother(alpha=0.3, window_size=5)
    aus = {"AU1": 0.5, "AU2": 0.3}
    
    result = smoother.update(aus)
    
    assert result == aus
    assert len(smoother.history) == 1


def test_temporal_smoother_exponential_average():
    """Test exponential moving average calculation."""
    smoother = TemporalSmoother(alpha=0.5, window_size=5)
    
    # First frame
    aus1 = {"AU1": 1.0}
    result1 = smoother.update(aus1)
    assert result1["AU1"] == 1.0
    
    # Second frame with lower value
    aus2 = {"AU1": 0.0}
    result2 = smoother.update(aus2)
    # With alpha=0.5: smoothed = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    assert abs(result2["AU1"] - 0.5) < 0.01
    
    # Third frame
    aus3 = {"AU1": 0.0}
    result3 = smoother.update(aus3)
    # smoothed = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
    assert abs(result3["AU1"] - 0.25) < 0.01


def test_temporal_smoother_history_limit():
    """Test that history is limited to window_size."""
    smoother = TemporalSmoother(alpha=0.3, window_size=3)
    
    for i in range(5):
        smoother.update({"AU1": i * 0.1})
    
    assert len(smoother.history) == 3


def test_temporal_smoother_reset():
    """Test reset functionality."""
    smoother = TemporalSmoother(alpha=0.3, window_size=5)
    smoother.update({"AU1": 0.5})
    
    smoother.reset()
    
    assert len(smoother.history) == 0
    assert len(smoother.current_smoothed) == 0


def test_temporal_smoother_multiple_aus():
    """Test smoothing with multiple AUs."""
    smoother = TemporalSmoother(alpha=0.3, window_size=5)
    
    aus1 = {"AU1": 0.8, "AU2": 0.2, "AU12": 0.9}
    result1 = smoother.update(aus1)
    
    aus2 = {"AU1": 0.2, "AU2": 0.8, "AU12": 0.1}
    result2 = smoother.update(aus2)
    
    # All AUs should be present
    assert "AU1" in result2
    assert "AU2" in result2
    assert "AU12" in result2
    
    # Values should be smoothed (not equal to raw input)
    assert result2["AU1"] != 0.2
    assert result2["AU2"] != 0.8
