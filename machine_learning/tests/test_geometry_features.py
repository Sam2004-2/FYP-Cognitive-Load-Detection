"""
Unit tests for geometry feature helpers in per_frame.py.
"""

import numpy as np
import pytest

from src.cle.extract.per_frame import eye_outer_center, head_roll, mouth_aspect_ratio


def test_mouth_aspect_ratio_simple_case():
    # Create minimal landmark array covering max index used (291)
    lm = np.zeros((292, 3), dtype=float)

    # Mouth corners (width = 1.0)
    lm[61, :2] = [0.0, 0.0]
    lm[291, :2] = [1.0, 0.0]

    # Upper/lower lip (height = 0.2)
    lm[13, :2] = [0.5, 0.2]
    lm[14, :2] = [0.5, 0.0]

    assert mouth_aspect_ratio(lm) == pytest.approx(0.2, abs=1e-6)


def test_head_roll_and_eye_center():
    lm = np.zeros((292, 3), dtype=float)

    # Outer eye corners (slope 1 => 45 degrees)
    lm[33, :2] = [0.0, 0.0]
    lm[263, :2] = [1.0, 1.0]

    cx, cy = eye_outer_center(lm)
    assert cx == pytest.approx(0.5, abs=1e-6)
    assert cy == pytest.approx(0.5, abs=1e-6)

    assert head_roll(lm) == pytest.approx(np.pi / 4, abs=1e-6)


def test_geometry_helpers_handle_short_landmarks():
    lm = np.zeros((10, 3), dtype=float)  # too short for required indices

    assert mouth_aspect_ratio(lm) == 0.0
    assert head_roll(lm) == 0.0
    assert eye_outer_center(lm) == (0.0, 0.0)

