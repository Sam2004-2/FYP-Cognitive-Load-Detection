"""
Unit tests for I/O utilities.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cle.utils.io import (
    load_features_csv,
    load_json,
    load_manifest,
    load_model_artifact,
    save_features_csv,
    save_json,
    save_model_artifact,
)


def test_save_and_load_json():
    """Test round-trip JSON save/load."""
    data = {"key": "value", "nested": {"a": 1, "b": [2, 3]}}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_json(data, str(path))
        loaded = load_json(str(path))
        assert loaded == data


def test_load_json_missing_file():
    """Test that loading a missing JSON file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="JSON file not found"):
        load_json("/tmp/nonexistent_file_12345.json")


def test_save_json_creates_directories():
    """Test that save_json creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "dir" / "test.json"
        save_json({"x": 1}, str(path))
        assert path.exists()
        assert load_json(str(path)) == {"x": 1}


def test_save_and_load_model_artifact():
    """Test round-trip model artifact save/load."""
    artifact = {"weights": [1.0, 2.0, 3.0], "intercept": 0.5}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.bin"
        save_model_artifact(artifact, str(path))
        loaded = load_model_artifact(str(path))
        assert loaded == artifact


def test_load_model_artifact_missing():
    """Test missing artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_artifact("/tmp/nonexistent_model_12345.bin")


def test_save_and_load_features_csv():
    """Test round-trip features CSV save/load."""
    df = pd.DataFrame({"feat_a": [0.1, 0.2], "feat_b": [0.3, 0.4]})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "features.csv"
        save_features_csv(df, str(path))
        loaded = load_features_csv(str(path))
        assert len(loaded) == 2
        assert "feat_a" in loaded.columns


def test_load_features_csv_missing():
    """Test missing features CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Features file not found"):
        load_features_csv("/tmp/nonexistent_features_12345.csv")


def test_save_features_csv_creates_directories():
    """Test that save_features_csv creates parent directories."""
    df = pd.DataFrame({"x": [1, 2]})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "deep" / "nested" / "features.csv"
        save_features_csv(df, str(path))
        assert path.exists()


def test_load_manifest_valid():
    """Test loading a valid manifest CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("video_file,label,role,user_id,notes\n")
        f.write("v1.mp4,low,train,u1,ok\n")
        f.write("v2.mp4,high,test,u2,ok\n")
        temp_path = f.name

    try:
        df = load_manifest(temp_path)
        assert len(df) == 2
        assert list(df.columns) == ["video_file", "label", "role", "user_id", "notes"]
    finally:
        Path(temp_path).unlink()


def test_load_manifest_missing_columns():
    """Test that a manifest with missing required columns raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("video_file,label\n")
        f.write("v1.mp4,low\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Missing required columns"):
            load_manifest(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_manifest_missing_file():
    """Test that a missing manifest raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        load_manifest("/tmp/nonexistent_manifest_12345.csv")


def test_save_model_artifact_creates_directories():
    """Test that save_model_artifact creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "models" / "sub" / "model.bin"
        save_model_artifact({"dummy": True}, str(path))
        assert path.exists()


def test_save_and_load_numpy_artifact():
    """Test saving and loading numpy arrays as artifacts."""
    arr = np.array([1.0, 2.0, 3.0])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "array.bin"
        save_model_artifact(arr, str(path))
        loaded = load_model_artifact(str(path))
        np.testing.assert_array_equal(loaded, arr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
