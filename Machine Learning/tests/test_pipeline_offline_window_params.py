"""
Tests that pipeline_offline window extraction respects Config defaults.
"""

import pytest

from src.cle.config import Config
from src.cle.extract.pipeline_offline import extract_windows


def _make_frame(valid: bool = True) -> dict:
    return {
        "ear_left": 0.3,
        "ear_right": 0.3,
        "ear_mean": 0.3,
        "brightness": 100.0,
        "quality": 1.0,
        "valid": valid,
    }


def test_extract_windows_respects_default_config():
    config = Config.default()
    fps = 30.0

    # 30 seconds of frames
    frame_features = [_make_frame(True) for _ in range(int(30 * fps))]

    video_metadata = {
        "user_id": "user01",
        "task": "Relax",
        "video_file": "user01_Relax.mp4",
        "label": "low",
        "role": "train",
    }

    windows = extract_windows(
        frame_features=frame_features,
        fps=fps,
        config=config,
        video_metadata=video_metadata,
    )

    # With 10s window and 2.5s step over 30s: starts at 0..20 in 2.5 increments => 9 windows
    assert len(windows) == 9

    starts = [w["t_start_s"] for w in windows]
    ends = [w["t_end_s"] for w in windows]

    assert starts[0] == pytest.approx(0.0)
    assert ends[0] == pytest.approx(10.0)
    assert starts[1] == pytest.approx(2.5)
    assert ends[1] == pytest.approx(12.5)
    assert starts[-1] == pytest.approx(20.0)
    assert ends[-1] == pytest.approx(30.0)

