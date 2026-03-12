"""
Unit tests for feature engineering (baseline centering and deltas).
"""

import numpy as np
import pandas as pd
import pytest

from src.cle.train.feature_engineering import (
    EngineeredFeatureSpec,
    add_centered_and_delta,
    build_feature_spec,
    compute_user_baseline,
)


def test_build_feature_spec_basic():
    """Test feature spec generation from base features."""
    spec = build_feature_spec(["ear_std", "perclos"])
    assert spec.base_features == ["ear_std", "perclos"]
    assert spec.centered_features == ["ear_std_centered", "perclos_centered"]
    assert spec.delta_features == ["ear_std_delta", "perclos_delta"]


def test_build_feature_spec_all_features():
    """Test all_features property concatenation."""
    spec = build_feature_spec(["a", "b"])
    assert spec.all_features == ["a", "b", "a_centered", "b_centered", "a_delta", "b_delta"]


def test_build_feature_spec_empty():
    """Test feature spec with empty base features."""
    spec = build_feature_spec([])
    assert spec.base_features == []
    assert spec.all_features == []


def test_compute_user_baseline_basic():
    """Test per-user baseline computation with baseline task."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1"],
            "task": ["Relax", "Relax", "Relax", "Relax", "Task1"],
            "t_start_s": [0, 2.5, 5.0, 7.5, 10.0],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 10.0],
        }
    )

    baseline = compute_user_baseline(df, feature_cols=["feat_a"], n_windows=4)

    assert len(baseline) == 1
    assert baseline.iloc[0]["user_id"] == "u1"
    # Median of [1, 2, 3, 4] = 2.5
    assert baseline.iloc[0]["baseline_feat_a"] == pytest.approx(2.5)


def test_compute_user_baseline_fallback():
    """Test fallback when user has no baseline tasks."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "task": ["HardTask", "HardTask"],
            "t_start_s": [0, 5],
            "feat_a": [8.0, 12.0],
        }
    )

    baseline = compute_user_baseline(df, feature_cols=["feat_a"])

    # Fallback: median of all windows = median([8, 12]) = 10
    assert baseline.iloc[0]["baseline_feat_a"] == pytest.approx(10.0)


def test_compute_user_baseline_multiple_users():
    """Test baseline computation for multiple users."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "task": ["Relax", "Relax", "Relax", "Relax"],
            "t_start_s": [0, 5, 0, 5],
            "feat_a": [2.0, 4.0, 10.0, 20.0],
        }
    )

    baseline = compute_user_baseline(df, feature_cols=["feat_a"])

    assert len(baseline) == 2
    u1_row = baseline[baseline["user_id"] == "u1"].iloc[0]
    u2_row = baseline[baseline["user_id"] == "u2"].iloc[0]
    assert u1_row["baseline_feat_a"] == pytest.approx(3.0)
    assert u2_row["baseline_feat_a"] == pytest.approx(15.0)


def test_compute_user_baseline_missing_columns():
    """Test that missing columns raise ValueError."""
    df = pd.DataFrame({"user_id": ["u1"], "task": ["Relax"]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_user_baseline(df, feature_cols=["feat_a"])


def test_add_centered_and_delta_basic():
    """Test that centered and delta features are computed correctly."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "task": ["Task1", "Task1", "Task1"],
            "t_start_s": [0.0, 2.5, 5.0],
            "feat_a": [10.0, 12.0, 15.0],
        }
    )

    baseline_df = pd.DataFrame(
        {"user_id": ["u1"], "baseline_feat_a": [10.0]}
    )

    result, spec = add_centered_and_delta(
        df, feature_cols=["feat_a"], baseline_df=baseline_df
    )

    assert "feat_a_centered" in result.columns
    assert "feat_a_delta" in result.columns

    centered = result["feat_a_centered"].tolist()
    assert centered[0] == pytest.approx(0.0)   # 10 - 10
    assert centered[1] == pytest.approx(2.0)   # 12 - 10
    assert centered[2] == pytest.approx(5.0)   # 15 - 10

    deltas = result["feat_a_delta"].tolist()
    assert deltas[0] == pytest.approx(0.0)     # first delta is 0
    assert deltas[1] == pytest.approx(2.0)     # 2 - 0
    assert deltas[2] == pytest.approx(3.0)     # 5 - 2


def test_add_centered_and_delta_spec():
    """Test that returned spec matches feature columns."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "task": ["T", "T"],
            "t_start_s": [0, 5],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
        }
    )

    baseline_df = pd.DataFrame(
        {"user_id": ["u1"], "baseline_x": [1.0], "baseline_y": [3.0]}
    )

    _, spec = add_centered_and_delta(
        df, feature_cols=["x", "y"], baseline_df=baseline_df
    )

    assert spec.base_features == ["x", "y"]
    assert spec.centered_features == ["x_centered", "y_centered"]
    assert spec.delta_features == ["x_delta", "y_delta"]


def test_add_centered_and_delta_missing_columns():
    """Test that missing columns raise ValueError."""
    df = pd.DataFrame({"user_id": ["u1"], "task": ["T"], "t_start_s": [0]})
    baseline_df = pd.DataFrame({"user_id": ["u1"], "baseline_a": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        add_centered_and_delta(df, feature_cols=["a"], baseline_df=baseline_df)


def test_add_centered_and_delta_missing_baseline_columns():
    """Test that missing baseline columns raise ValueError."""
    df = pd.DataFrame(
        {"user_id": ["u1"], "task": ["T"], "t_start_s": [0], "a": [1.0]}
    )
    baseline_df = pd.DataFrame({"user_id": ["u1"]})  # missing baseline_a
    with pytest.raises(ValueError, match="baseline_df missing columns"):
        add_centered_and_delta(df, feature_cols=["a"], baseline_df=baseline_df)


def test_engineered_feature_spec_frozen():
    """Test that EngineeredFeatureSpec is immutable."""
    spec = EngineeredFeatureSpec(
        base_features=["a"], centered_features=["a_c"], delta_features=["a_d"]
    )
    with pytest.raises(AttributeError):
        spec.base_features = ["b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
