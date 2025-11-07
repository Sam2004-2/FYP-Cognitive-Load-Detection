"""
Unit tests for windowing logic.
"""

import numpy as np
import pytest

from src.cle.extract.windowing import (
    WindowBuffer,
    interpolate_gaps,
    sliding_window_indices,
    validate_window_quality,
)


def test_sliding_window_indices_basic():
    """Test basic sliding window generation."""
    n_frames = 600  # 20 seconds at 30fps
    fps = 30.0
    length_s = 10.0  # 10 second windows
    step_s = 5.0     # 5 second step

    windows = sliding_window_indices(n_frames, fps, length_s, step_s)

    # Should generate multiple windows
    assert len(windows) > 0

    # First window should start at 0
    assert windows[0][0] == 0
    assert windows[0][1] == int(length_s * fps)  # 300 frames

    # Windows should overlap by (length_s - step_s)
    if len(windows) > 1:
        assert windows[1][0] == int(step_s * fps)  # Start at 150


def test_sliding_window_indices_no_overlap():
    """Test sliding windows with no overlap (step = length)."""
    n_frames = 600
    fps = 30.0
    length_s = 10.0
    step_s = 10.0  # No overlap

    windows = sliding_window_indices(n_frames, fps, length_s, step_s)

    # Windows should not overlap
    for i in range(len(windows) - 1):
        assert windows[i][1] == windows[i + 1][0]


def test_sliding_window_indices_exact_fit():
    """Test sliding windows with exact fit."""
    fps = 30.0
    length_s = 10.0
    step_s = 10.0
    n_frames = int(fps * length_s * 3)  # Exactly 3 windows

    windows = sliding_window_indices(n_frames, fps, length_s, step_s)

    assert len(windows) == 3


def test_validate_window_quality_all_valid():
    """Test window quality validation with all valid frames."""
    frame_features = [{"valid": True} for _ in range(100)]

    is_valid, bad_ratio = validate_window_quality(frame_features, max_bad_ratio=0.2)

    assert is_valid is True
    assert bad_ratio == 0.0


def test_validate_window_quality_some_bad():
    """Test window quality validation with some bad frames."""
    frame_features = [{"valid": True} for _ in range(80)]
    frame_features.extend([{"valid": False} for _ in range(20)])

    # 20% bad frames, should be at the threshold
    is_valid, bad_ratio = validate_window_quality(frame_features, max_bad_ratio=0.2)

    assert bad_ratio == 0.2
    assert is_valid is True  # Exactly at threshold

    # Exceed threshold
    is_valid, bad_ratio = validate_window_quality(frame_features, max_bad_ratio=0.15)

    assert bad_ratio == 0.2
    assert is_valid is False


def test_validate_window_quality_empty():
    """Test window quality validation with empty window."""
    frame_features = []

    is_valid, bad_ratio = validate_window_quality(frame_features)

    assert is_valid is False
    assert bad_ratio == 1.0


def test_interpolate_gaps_small():
    """Test interpolation of small gaps."""
    frame_features = [
        {"valid": True, "ear_mean": 0.3, "pupil_mean": 0.5},
        {"valid": False, "ear_mean": 0.0, "pupil_mean": 0.0},  # Gap
        {"valid": True, "ear_mean": 0.4, "pupil_mean": 0.6},
    ]

    interpolated = interpolate_gaps(frame_features, max_gap=3)

    # Gap should be interpolated
    assert interpolated[1]["valid"] is True
    assert 0.3 < interpolated[1]["ear_mean"] < 0.4
    assert 0.5 < interpolated[1]["pupil_mean"] < 0.6


def test_interpolate_gaps_too_large():
    """Test that large gaps are not interpolated."""
    frame_features = [
        {"valid": True, "ear_mean": 0.3},
        {"valid": False, "ear_mean": 0.0},
        {"valid": False, "ear_mean": 0.0},
        {"valid": False, "ear_mean": 0.0},
        {"valid": False, "ear_mean": 0.0},  # Gap of 4 frames
        {"valid": True, "ear_mean": 0.4},
    ]

    interpolated = interpolate_gaps(frame_features, max_gap=3)

    # Gap too large, should remain invalid
    assert interpolated[1]["valid"] is False
    assert interpolated[2]["valid"] is False


def test_window_buffer_basic():
    """Test WindowBuffer basic operations."""
    window_length_s = 5.0
    fps = 30.0

    buffer = WindowBuffer(window_length_s, fps)

    assert len(buffer) == 0
    assert not buffer.is_ready()

    # Add frames until buffer is ready
    max_frames = int(window_length_s * fps)
    for i in range(max_frames):
        buffer.add_frame({"frame_idx": i, "valid": True})

    assert buffer.is_ready()
    assert len(buffer) == max_frames


def test_window_buffer_overflow():
    """Test WindowBuffer overflow behavior (ring buffer)."""
    window_length_s = 2.0  # Small window
    fps = 10.0
    max_frames = int(window_length_s * fps)  # 20 frames

    buffer = WindowBuffer(window_length_s, fps)

    # Add more frames than buffer capacity
    for i in range(max_frames + 10):
        buffer.add_frame({"frame_idx": i, "valid": True})

    # Buffer should maintain max size
    assert len(buffer) == max_frames

    # Should contain most recent frames
    window = buffer.get_window()
    assert window[-1]["frame_idx"] == max_frames + 9  # Last frame added


def test_window_buffer_times():
    """Test WindowBuffer time calculation."""
    window_length_s = 10.0
    fps = 30.0

    buffer = WindowBuffer(window_length_s, fps)

    # Add frames
    for i in range(int(fps * window_length_s)):
        buffer.add_frame({"frame_idx": i})

    start_time, end_time = buffer.get_window_times()

    assert start_time == 0.0
    assert abs(end_time - window_length_s) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

