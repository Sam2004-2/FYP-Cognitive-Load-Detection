"""
Unit tests for blink detection.
"""

import numpy as np
import pytest

from src.cle.extract.features import compute_blink_features, detect_blinks


def test_detect_blinks_simple():
    """Test basic blink detection with synthetic data."""
    # Create synthetic EAR time series with known blinks
    fps = 30.0
    duration_s = 2.0
    n_frames = int(fps * duration_s)

    # Start with eyes open (EAR ~ 0.3)
    ear_series = np.ones(n_frames) * 0.3

    # Add two blinks (eyes closed for ~4 frames = 133ms at 30fps)
    # Blink 1: frames 10-13
    ear_series[10:14] = 0.15  # Below threshold

    # Blink 2: frames 40-43
    ear_series[40:44] = 0.18  # Below threshold

    # Detect blinks
    blinks = detect_blinks(
        ear_series,
        fps=fps,
        ear_threshold=0.21,
        min_blink_ms=120,
        max_blink_ms=400,
    )

    # Should detect exactly 2 blinks
    assert len(blinks) == 2, f"Expected 2 blinks, got {len(blinks)}"

    # Check blink positions
    assert blinks[0][0] == 10
    assert blinks[1][0] == 40


def test_detect_blinks_too_short():
    """Test that very short blinks are rejected."""
    fps = 30.0
    n_frames = 100

    # Eyes open
    ear_series = np.ones(n_frames) * 0.3

    # Add very short blink (only 2 frames = 67ms at 30fps)
    ear_series[50:52] = 0.15

    blinks = detect_blinks(
        ear_series,
        fps=fps,
        ear_threshold=0.21,
        min_blink_ms=120,  # Minimum 120ms
    )

    # Should not detect any blinks (too short)
    assert len(blinks) == 0


def test_detect_blinks_too_long():
    """Test that very long blinks are rejected."""
    fps = 30.0
    n_frames = 100

    # Eyes open
    ear_series = np.ones(n_frames) * 0.3

    # Add very long blink (20 frames = 667ms at 30fps)
    ear_series[40:60] = 0.15

    blinks = detect_blinks(
        ear_series,
        fps=fps,
        ear_threshold=0.21,
        max_blink_ms=400,  # Maximum 400ms
    )

    # Should not detect any blinks (too long)
    assert len(blinks) == 0


def test_compute_blink_features():
    """Test blink feature computation."""
    fps = 30.0
    duration_s = 1.0  # 1 minute for easy blink rate calculation
    n_frames = int(fps * duration_s * 60)  # 1800 frames

    # Create EAR series with known number of blinks
    ear_series = np.ones(n_frames) * 0.3

    # Add 15 blinks (15 blinks/min)
    blink_positions = np.linspace(100, n_frames - 100, 15, dtype=int)
    for pos in blink_positions:
        ear_series[pos:pos+4] = 0.15

    # Compute features
    config = {
        "blink.ear_thresh": 0.21,
        "blink.min_blink_ms": 120,
        "blink.max_blink_ms": 400,
    }
    features = compute_blink_features(ear_series, fps, config)

    # Check blink rate (should be ~15 blinks/min)
    assert 14 <= features["blink_rate"] <= 16, f"Expected ~15 blinks/min, got {features['blink_rate']}"

    # Check blink count
    assert features["blink_count"] == 15

    # Check mean blink duration (~133ms for 4 frames at 30fps)
    assert 120 <= features["mean_blink_duration"] <= 150


def test_empty_ear_series():
    """Test handling of empty EAR series."""
    fps = 30.0
    ear_series = np.array([])

    blinks = detect_blinks(ear_series, fps)
    assert len(blinks) == 0

    config = {"blink.ear_thresh": 0.21, "blink.min_blink_ms": 120, "blink.max_blink_ms": 400}
    features = compute_blink_features(ear_series, fps, config)

    assert features["blink_rate"] == 0.0
    assert features["blink_count"] == 0.0
    assert features["mean_blink_duration"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

