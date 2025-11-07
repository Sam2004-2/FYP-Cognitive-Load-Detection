"""
Unit tests for TEPR (Task-Evoked Pupillary Response) feature computation.
"""

import numpy as np
import pytest

from src.cle.extract.features import compute_tepr_features


def test_tepr_baseline_computation():
    """Test TEPR baseline computation with known data."""
    fps = 30.0
    baseline_s = 10.0
    n_frames = int(fps * 20)  # 20 seconds of data

    # Create pupil series: constant baseline for first 10s, then increase
    baseline_value = 0.5
    pupil_series = np.ones(n_frames) * baseline_value

    # Increase pupil size after baseline period (simulate cognitive load)
    baseline_frames = int(baseline_s * fps)
    pupil_series[baseline_frames:] = 0.6  # 10% increase

    # Compute TEPR features
    features = compute_tepr_features(pupil_series, fps, baseline_s)

    # Check baseline
    assert abs(features["tepr_baseline"] - baseline_value) < 0.01

    # Check delta mean (should be positive, averaging the baseline and increased portions)
    assert features["tepr_delta_mean"] > 0, "TEPR delta should be positive for increased pupil"

    # Check delta peak (should be ~0.1)
    assert abs(features["tepr_delta_peak"] - 0.1) < 0.02


def test_tepr_negative_change():
    """Test TEPR with decreasing pupil size."""
    fps = 30.0
    baseline_s = 5.0
    n_frames = int(fps * 10)

    # Start high, then decrease
    pupil_series = np.ones(n_frames) * 0.6
    baseline_frames = int(baseline_s * fps)
    pupil_series[baseline_frames:] = 0.4  # Decrease

    features = compute_tepr_features(pupil_series, fps, baseline_s)

    # Delta should be negative
    assert features["tepr_delta_mean"] < 0, "TEPR delta should be negative for decreased pupil"
    assert features["tepr_delta_peak"] < 0, "TEPR peak should be negative"


def test_tepr_with_noise():
    """Test TEPR robustness to noise."""
    fps = 30.0
    baseline_s = 10.0
    n_frames = int(fps * 20)

    # Create pupil series with noise
    baseline_value = 0.5
    pupil_series = np.ones(n_frames) * baseline_value
    pupil_series += np.random.randn(n_frames) * 0.02  # Add noise

    # Clear increase after baseline
    baseline_frames = int(baseline_s * fps)
    pupil_series[baseline_frames:] += 0.15

    features = compute_tepr_features(pupil_series, fps, baseline_s)

    # Should still detect positive change despite noise
    assert features["tepr_delta_mean"] > 0.05, "Should detect increase despite noise"


def test_tepr_insufficient_baseline():
    """Test TEPR with insufficient baseline data."""
    fps = 30.0
    baseline_s = 10.0
    n_frames = int(fps * 5)  # Only 5 seconds, less than baseline requirement

    pupil_series = np.ones(n_frames) * 0.5

    # Should use median of all data as baseline
    features = compute_tepr_features(pupil_series, fps, baseline_s, min_baseline_samples=150)

    assert features["tepr_baseline"] > 0
    assert features["tepr_delta_mean"] is not None


def test_tepr_empty_series():
    """Test TEPR with empty pupil series."""
    fps = 30.0
    pupil_series = np.array([])

    features = compute_tepr_features(pupil_series, fps, baseline_s=10.0)

    assert features["tepr_delta_mean"] == 0.0
    assert features["tepr_delta_peak"] == 0.0
    assert features["tepr_auc"] == 0.0
    assert features["tepr_baseline"] == 0.0


def test_tepr_all_zeros():
    """Test TEPR with all-zero pupil series (invalid data)."""
    fps = 30.0
    n_frames = 300
    pupil_series = np.zeros(n_frames)

    features = compute_tepr_features(pupil_series, fps, baseline_s=5.0)

    # Should return zero features for invalid data
    assert features["tepr_delta_mean"] == 0.0
    assert features["tepr_baseline"] == 0.0


def test_tepr_gradual_increase():
    """Test TEPR with gradual pupil increase (simulating real cognitive load)."""
    fps = 30.0
    baseline_s = 10.0
    duration_s = 30.0
    n_frames = int(fps * duration_s)

    # Gradual linear increase from 0.5 to 0.7
    pupil_series = np.linspace(0.5, 0.7, n_frames)

    features = compute_tepr_features(pupil_series, fps, baseline_s)

    # Baseline should be close to starting value
    assert 0.48 < features["tepr_baseline"] < 0.52

    # Mean delta should be positive
    assert features["tepr_delta_mean"] > 0

    # Peak delta should be at the end (~0.2 increase)
    assert features["tepr_delta_peak"] > 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

