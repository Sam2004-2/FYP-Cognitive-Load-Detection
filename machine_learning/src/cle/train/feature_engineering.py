"""
Feature engineering for physio-aligned real-time cognitive load estimation.

Adds:
  - Per-user baseline normalization (centered features)
  - Per-session temporal deltas (delta of centered features)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import pandas as pd


DEFAULT_BASELINE_TASKS: Set[str] = {"Relax", "Breathing", "Video1"}


@dataclass(frozen=True)
class EngineeredFeatureSpec:
    base_features: List[str]
    centered_features: List[str]
    delta_features: List[str]

    @property
    def all_features(self) -> List[str]:
        return [*self.base_features, *self.centered_features, *self.delta_features]


def build_feature_spec(base_features: Sequence[str]) -> EngineeredFeatureSpec:
    base = list(base_features)
    centered = [f"{f}_centered" for f in base]
    delta = [f"{f}_delta" for f in base]
    return EngineeredFeatureSpec(base_features=base, centered_features=centered, delta_features=delta)


def compute_user_baseline(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    baseline_tasks: Set[str] = DEFAULT_BASELINE_TASKS,
    n_windows: int = 4,
    time_col: str = "t_start_s",
    user_col: str = "user_id",
    task_col: str = "task",
) -> pd.DataFrame:
    """
    Compute per-user baseline as the median of the earliest N baseline windows.

    Fallback: if a user has no baseline task windows, use the median of all windows
    for that user.
    """
    required = {user_col, task_col, time_col, *feature_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_user_baseline: missing required columns: {sorted(missing)}")

    rows = []
    for user_id, user_df in df.groupby(user_col, sort=False):
        baseline_df = user_df[user_df[task_col].isin(baseline_tasks)].copy()
        if len(baseline_df) > 0:
            baseline_df = baseline_df.sort_values(time_col, ascending=True).head(n_windows)
        else:
            # Fallback: user median across all windows (not time-limited).
            baseline_df = user_df.copy()
        medians = baseline_df[list(feature_cols)].median(axis=0, numeric_only=True)
        row = {user_col: user_id, **{f"baseline_{c}": float(medians.get(c)) for c in feature_cols}}
        rows.append(row)

    return pd.DataFrame(rows)


def add_centered_and_delta(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    baseline_df: pd.DataFrame,
    session_cols: Sequence[str] = ("user_id", "task"),
    time_col: str = "t_start_s",
    user_col: str = "user_id",
) -> Tuple[pd.DataFrame, EngineeredFeatureSpec]:
    """
    Add centered and delta features.

    - Centered features are per-user: x - baseline_x
    - Delta features are per-session: centered_x[t] - centered_x[t-1]
      (first window delta is 0.0)
    """
    spec = build_feature_spec(feature_cols)

    required = {user_col, time_col, *feature_cols, *session_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_centered_and_delta: missing required columns: {sorted(missing)}")

    baseline_required = {user_col, *[f"baseline_{c}" for c in feature_cols]}
    baseline_missing = baseline_required - set(baseline_df.columns)
    if baseline_missing:
        raise ValueError(
            f"add_centered_and_delta: baseline_df missing columns: {sorted(baseline_missing)}"
        )

    out = df.copy()
    out = out.merge(baseline_df[[user_col, *[f"baseline_{c}" for c in feature_cols]]], on=user_col, how="left")

    for base_col in feature_cols:
        out[f"{base_col}_centered"] = out[base_col] - out[f"baseline_{base_col}"]

    out = out.sort_values([*session_cols, time_col], ascending=True).copy()
    for base_col in feature_cols:
        centered_col = f"{base_col}_centered"
        out[f"{base_col}_delta"] = (
            out.groupby(list(session_cols), sort=False)[centered_col].diff().fillna(0.0)
        )

    return out, spec
