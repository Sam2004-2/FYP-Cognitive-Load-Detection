"""
Integration tests for full pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cle.config import Config
from src.cle.extract.features import get_feature_names


def test_config_loading():
    """Test configuration loading and validation."""
    # Test default config
    config = Config.default()

    assert config.get("seed") == 42
    assert config.get("windows.length_s") == 10.0
    assert config.get("blink.ear_thresh") == 0.21

    # Test validation
    with pytest.raises(ValueError):
        Config({"seed": 42})  # Missing required keys


def test_feature_names_consistency():
    """Test that feature names are consistent across configs."""
    config_dict = {
        "seed": 42,
        "windows": {"length_s": 20.0, "step_s": 5.0},
        "quality": {"min_face_conf": 0.5, "max_bad_frame_ratio": 0.2},
        "blink": {"ear_thresh": 0.21, "min_blink_ms": 120, "max_blink_ms": 400},
        "tepr": {"baseline_s": 10.0, "min_baseline_samples": 150},
        "features_enabled": {
            "tepr": False,  # TEPR disabled
            "blinks": True,
            "perclos": True,
            "brightness": True,
            "fix_sac": False,
            "gaze_entropy": False,
        },
        "model": {"type": "logreg", "calibration": "platt"},
    }

    config = Config(config_dict)
    feature_names = get_feature_names(config.to_dict())

    # Check expected features are present
    # NOTE: brightness and quality metrics are excluded from model features
    expected_features = [
        "blink_rate",
        "blink_count",
        "mean_blink_duration",
        "ear_std",
        "perclos",
    ]

    assert len(feature_names) == len(expected_features)
    for expected in expected_features:
        assert expected in feature_names, f"Missing feature: {expected}"


def test_feature_names_with_geometry_enabled():
    """Test that geometry features are appended when enabled."""
    config_dict = {
        "seed": 42,
        "windows": {"length_s": 20.0, "step_s": 5.0},
        "quality": {"min_face_conf": 0.5, "max_bad_frame_ratio": 0.2},
        "blink": {"ear_thresh": 0.21, "min_blink_ms": 120, "max_blink_ms": 400},
        "tepr": {"baseline_s": 10.0, "min_baseline_samples": 150},
        "features_enabled": {
            "tepr": False,  # TEPR disabled
            "blinks": True,
            "perclos": True,
            "brightness": True,
            "geometry": True,
            "fix_sac": False,
            "gaze_entropy": False,
        },
        "model": {"type": "logreg", "calibration": "platt"},
    }

    feature_names = get_feature_names(config_dict)

    expected = [
        "blink_rate",
        "blink_count",
        "mean_blink_duration",
        "ear_std",
        "perclos",
        "mouth_open_mean",
        "mouth_open_std",
        "roll_std",
        "pitch_std",
        "yaw_std",
        "motion_mean",
        "motion_std",
    ]

    assert feature_names == expected


def test_feature_order_consistency():
    """Test that feature order is consistent."""
    config_dict = {
        "seed": 42,
        "windows": {"length_s": 20.0, "step_s": 5.0},
        "quality": {"min_face_conf": 0.5, "max_bad_frame_ratio": 0.2},
        "blink": {"ear_thresh": 0.21, "min_blink_ms": 120, "max_blink_ms": 400},
        "tepr": {"baseline_s": 10.0, "min_baseline_samples": 150},
        "features_enabled": {
            "tepr": False,  # TEPR disabled
            "blinks": True,
            "perclos": True,
            "brightness": True,
            "fix_sac": False,
            "gaze_entropy": False,
        },
        "model": {"type": "logreg", "calibration": "platt"},
    }

    # Get feature names multiple times
    names1 = get_feature_names(config_dict)
    names2 = get_feature_names(config_dict)

    # Order should be exactly the same
    assert names1 == names2


def test_manifest_loading():
    """Test manifest CSV loading."""
    from src.cle.utils.io import load_manifest

    # Create temporary manifest
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("video_file,label,role,user_id,notes\n")
        f.write("test1.mp4,low,train,user01,test\n")
        f.write("test2.mp4,high,train,user01,test\n")
        temp_path = f.name

    try:
        df = load_manifest(temp_path)

        assert len(df) == 2
        assert "video_file" in df.columns
        assert "label" in df.columns
        assert df.iloc[0]["label"] == "low"
        assert df.iloc[1]["label"] == "high"
    finally:
        Path(temp_path).unlink()


def test_model_artifact_save_load():
    """Test saving and loading model artifacts."""
    from src.cle.utils.io import load_model_artifact, save_model_artifact

    # Create dummy model (simple dict)
    model = {"type": "test", "params": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "model.bin"

        # Save
        save_model_artifact(model, artifact_path)
        assert artifact_path.exists()

        # Load
        loaded_model = load_model_artifact(artifact_path)
        assert loaded_model == model


def test_json_save_load():
    """Test JSON save/load."""
    from src.cle.utils.io import load_json, save_json

    data = {
        "features": ["feat1", "feat2", "feat3"],
        "n_features": 3,
        "meta": {"date": "2024-01-01"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test.json"

        # Save
        save_json(data, json_path)
        assert json_path.exists()

        # Load
        loaded_data = load_json(json_path)
        assert loaded_data == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
