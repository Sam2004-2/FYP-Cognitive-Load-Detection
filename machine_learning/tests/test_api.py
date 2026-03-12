"""
Unit tests for the public prediction API.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.cle.api import predict_window


def _make_artifacts(n_features=3, task_mode="classification"):
    """Create mock model artifacts for testing."""
    feature_names = [f"feat_{i}" for i in range(n_features)]

    model = MagicMock()
    if task_mode == "classification":
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
    else:
        model.predict.return_value = np.array([0.65])

    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x  # identity

    return {
        "model": model,
        "scaler": scaler,
        "imputer": None,
        "feature_spec": {"features": feature_names},
        "calibration": {},
        "task_mode": task_mode,
    }


def test_predict_window_from_dict():
    """Test prediction from feature dict."""
    artifacts = _make_artifacts(3, "classification")
    features = {"feat_0": 0.1, "feat_1": 0.2, "feat_2": 0.3}

    cli = predict_window(features, artifacts)
    assert isinstance(cli, float)
    assert 0.0 <= cli <= 1.0
    assert cli == pytest.approx(0.7, abs=1e-6)


def test_predict_window_from_list():
    """Test prediction from feature list."""
    artifacts = _make_artifacts(3, "classification")
    features = [0.1, 0.2, 0.3]

    cli = predict_window(features, artifacts)
    assert isinstance(cli, float)
    assert cli == pytest.approx(0.7, abs=1e-6)


def test_predict_window_from_array():
    """Test prediction from numpy array."""
    artifacts = _make_artifacts(3, "classification")
    features = np.array([0.1, 0.2, 0.3])

    cli = predict_window(features, artifacts)
    assert isinstance(cli, float)


def test_predict_window_regression():
    """Test prediction in regression mode."""
    artifacts = _make_artifacts(3, "regression")

    cli = predict_window([0.1, 0.2, 0.3], artifacts)
    assert isinstance(cli, float)
    assert 0.0 <= cli <= 1.0
    assert cli == pytest.approx(0.65, abs=1e-6)


def test_predict_window_regression_clipping():
    """Test that regression output outside [0,1] is clipped."""
    artifacts = _make_artifacts(3, "regression")
    artifacts["model"].predict.return_value = np.array([1.5])

    cli = predict_window([0.1, 0.2, 0.3], artifacts)
    assert cli == pytest.approx(1.0)


def test_predict_window_dimension_mismatch():
    """Test that feature dimension mismatch raises ValueError."""
    artifacts = _make_artifacts(3, "classification")
    features = [0.1, 0.2]  # only 2, expect 3

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        predict_window(features, artifacts)


def test_predict_window_unsupported_type():
    """Test that unsupported feature type raises ValueError."""
    artifacts = _make_artifacts(3, "classification")

    with pytest.raises(ValueError, match="Unsupported feature type"):
        predict_window("not_a_valid_type", artifacts)


def test_predict_window_nan_handling():
    """Test that NaN values in features are replaced with zeros."""
    artifacts = _make_artifacts(3, "classification")
    features = [0.1, float("nan"), 0.3]

    # Should not raise
    cli = predict_window(features, artifacts)
    assert isinstance(cli, float)

    # Verify scaler received array with no NaN
    call_args = artifacts["scaler"].transform.call_args[0][0]
    assert not np.any(np.isnan(call_args))


def test_predict_window_with_imputer():
    """Test that imputer is called when present in artifacts."""
    artifacts = _make_artifacts(3, "classification")
    imputer = MagicMock()
    imputer.transform.side_effect = lambda x: x
    artifacts["imputer"] = imputer

    predict_window([0.1, 0.2, 0.3], artifacts)

    imputer.transform.assert_called_once()


def test_predict_window_dict_missing_keys_default_zero():
    """Test that missing keys in feature dict default to 0.0."""
    artifacts = _make_artifacts(3, "classification")
    features = {"feat_0": 0.5}  # missing feat_1, feat_2

    cli = predict_window(features, artifacts)
    assert isinstance(cli, float)

    # Verify the array sent to scaler has correct shape
    call_args = artifacts["scaler"].transform.call_args[0][0]
    assert call_args.shape == (1, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
