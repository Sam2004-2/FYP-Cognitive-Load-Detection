"""
Unit tests for configuration management.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.cle.config import Config, load_config


def _minimal_config():
    """Return a minimal valid config dict."""
    return {
        "seed": 42,
        "windows": {"length_s": 10.0, "step_s": 2.5},
        "quality": {"min_face_conf": 0.5, "max_bad_frame_ratio": 0.05},
        "blink": {"ear_thresh": 0.21, "min_blink_ms": 120, "max_blink_ms": 400},
        "features_enabled": {
            "tepr": False,
            "blinks": True,
            "perclos": True,
            "brightness": True,
            "geometry": True,
            "fix_sac": False,
            "gaze_entropy": False,
        },
    }


def test_config_from_dict():
    """Test creating Config from a dictionary."""
    cfg = Config(_minimal_config())
    assert cfg.get("seed") == 42
    assert cfg.get("windows.length_s") == 10.0


def test_config_missing_required_key():
    """Test that missing required keys raise ValueError."""
    bad = {"seed": 1}
    with pytest.raises(ValueError, match="Missing required config key"):
        Config(bad)


def test_config_invalid_window_length():
    """Test that non-positive window length is rejected."""
    d = _minimal_config()
    d["windows"]["length_s"] = 0
    with pytest.raises(ValueError, match="windows.length_s must be positive"):
        Config(d)


def test_config_step_exceeds_length():
    """Test that step_s > length_s is rejected."""
    d = _minimal_config()
    d["windows"]["step_s"] = 20.0
    with pytest.raises(ValueError, match="windows.step_s cannot exceed"):
        Config(d)


def test_config_invalid_bad_frame_ratio():
    """Test that bad_frame_ratio outside [0,1] is rejected."""
    d = _minimal_config()
    d["quality"]["max_bad_frame_ratio"] = 1.5
    with pytest.raises(ValueError, match="max_bad_frame_ratio must be in"):
        Config(d)


def test_config_invalid_ear_thresh():
    """Test that non-positive ear_thresh is rejected."""
    d = _minimal_config()
    d["blink"]["ear_thresh"] = -0.1
    with pytest.raises(ValueError, match="blink.ear_thresh must be positive"):
        Config(d)


def test_config_invalid_min_blink_ms():
    """Test that non-positive min_blink_ms is rejected."""
    d = _minimal_config()
    d["blink"]["min_blink_ms"] = 0
    with pytest.raises(ValueError, match="blink.min_blink_ms must be positive"):
        Config(d)


def test_config_get_dot_notation():
    """Test dot-notation key access."""
    cfg = Config(_minimal_config())
    assert cfg.get("blink.ear_thresh") == 0.21
    assert cfg.get("quality.max_bad_frame_ratio") == 0.05


def test_config_get_default():
    """Test that get returns default for missing key."""
    cfg = Config(_minimal_config())
    assert cfg.get("nonexistent", "fallback") == "fallback"
    assert cfg.get("windows.unknown_key", 99) == 99


def test_config_get_deep_missing():
    """Test dot-notation where intermediate key is missing."""
    cfg = Config(_minimal_config())
    assert cfg.get("a.b.c.d", None) is None


def test_config_getitem():
    """Test dictionary-style access."""
    cfg = Config(_minimal_config())
    assert cfg["seed"] == 42
    with pytest.raises(KeyError):
        _ = cfg["missing_key"]


def test_config_contains():
    """Test __contains__."""
    cfg = Config(_minimal_config())
    assert "seed" in cfg
    assert "nonexistent" not in cfg


def test_config_to_dict():
    """Test that to_dict returns a copy."""
    d = _minimal_config()
    cfg = Config(d)
    result = cfg.to_dict()
    assert result == d
    # Mutating the copy should not affect original
    result["seed"] = 999
    assert cfg.get("seed") == 42


def test_config_hash_deterministic():
    """Test that hash is deterministic for same config."""
    cfg1 = Config(_minimal_config())
    cfg2 = Config(_minimal_config())
    assert cfg1.hash() == cfg2.hash()


def test_config_hash_changes_with_content():
    """Test that hash changes when config values differ."""
    d1 = _minimal_config()
    d2 = _minimal_config()
    d2["seed"] = 99
    assert Config(d1).hash() != Config(d2).hash()


def test_config_from_yaml():
    """Test loading config from a YAML file."""
    d = _minimal_config()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(d, f)
        temp_path = f.name

    try:
        cfg = Config.from_yaml(temp_path)
        assert cfg.get("seed") == 42
        assert cfg.get("windows.step_s") == 2.5
    finally:
        Path(temp_path).unlink()


def test_config_from_yaml_missing_file():
    """Test that missing YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Config.from_yaml("/tmp/does_not_exist_12345.yaml")


def test_load_config_none_uses_default():
    """Test that load_config(None) falls back to default.yaml."""
    cfg = load_config(None)
    # Default config should have seed=42
    assert cfg.get("seed") == 42


def test_load_config_with_path():
    """Test that load_config with explicit path uses that file."""
    d = _minimal_config()
    d["seed"] = 7
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(d, f)
        temp_path = f.name

    try:
        cfg = load_config(temp_path)
        assert cfg.get("seed") == 7
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
