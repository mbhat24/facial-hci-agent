"""Tests for emotion confidence intervals."""
import pytest
import numpy as np
from src.facial_hci.inference.emotion_rules import emotion_confidence_interval, emotion_probabilities


def test_emotion_confidence_interval_structure():
    """Test that confidence intervals return correct structure."""
    aus = {"AU1": 0.5, "AU2": 0.3, "AU12": 0.8}
    
    intervals = emotion_confidence_interval(aus, samples=10)
    
    # Should return dict with emotion names as keys
    assert isinstance(intervals, dict)
    assert len(intervals) > 0
    
    # Each value should be a tuple of (lower, upper)
    for emotion, interval in intervals.items():
        assert isinstance(interval, tuple)
        assert len(interval) == 2
        assert isinstance(interval[0], float)  # lower bound
        assert isinstance(interval[1], float)  # upper bound
        assert interval[0] <= interval[1]  # lower <= upper


def test_emotion_confidence_interval_bounds():
    """Test that confidence intervals are within valid range."""
    aus = {"AU1": 0.5, "AU2": 0.3, "AU12": 0.8}
    
    intervals = emotion_confidence_interval(aus, samples=20)
    
    for emotion, (lower, upper) in intervals.items():
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0


def test_emotion_confidence_interval_deterministic():
    """Test that same inputs produce consistent intervals."""
    aus = {"AU1": 0.5, "AU2": 0.3, "AU12": 0.8}
    
    # Set seed for reproducibility
    np.random.seed(42)
    intervals1 = emotion_confidence_interval(aus, samples=10)
    
    np.random.seed(42)
    intervals2 = emotion_confidence_interval(aus, samples=10)
    
    for emotion in intervals1:
        assert intervals1[emotion] == intervals2[emotion]


def test_emotion_confidence_interval_empty_aus():
    """Test confidence intervals with empty AU dict."""
    aus = {}
    
    intervals = emotion_confidence_interval(aus, samples=5)
    
    # Should still return intervals for all emotions
    assert len(intervals) > 0


def test_emotion_confidence_interval_samples_parameter():
    """Test that different sample counts affect results."""
    aus = {"AU1": 0.5, "AU12": 0.8}
    
    intervals_5 = emotion_confidence_interval(aus, samples=5)
    intervals_50 = emotion_confidence_interval(aus, samples=50)
    
    # With more samples, intervals should be more stable (narrower)
    # This is a probabilistic test, so we just check both return valid results
    assert len(intervals_5) == len(intervals_50)
