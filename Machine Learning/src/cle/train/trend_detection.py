"""
Trend detection for real-time cognitive load monitoring.

Detects whether cognitive load is increasing, decreasing, or stable
by computing a rolling linear regression slope over recent predictions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class LoadTrend(str, Enum):
    """Cognitive load trend direction."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class TrendResult:
    """Result of a trend detection computation."""

    trend: LoadTrend
    slope: float  # change in CLI per window step
    slope_per_minute: float  # change in CLI per minute
    confidence: float  # R² of the linear fit
    n_windows: int


class TrendDetector:
    """
    Detect cognitive load trends from a stream of predictions.

    Uses a rolling window of recent predictions and fits a linear
    regression to determine the slope (direction and magnitude).

    Args:
        window_size: Number of recent predictions to consider.
        window_step_s: Seconds between consecutive predictions.
        slope_threshold: Minimum absolute slope (per window) to declare
            a trend.  Values below this are classified as STABLE.
        min_confidence: Minimum R² of the linear fit to trust the trend.
            Low R² means the recent predictions are noisy / non-linear,
            so we fall back to STABLE.
    """

    def __init__(
        self,
        window_size: int = 5,
        window_step_s: float = 2.5,
        slope_threshold: float = 0.02,
        min_confidence: float = 0.1,
    ) -> None:
        self.window_size = window_size
        self.window_step_s = window_step_s
        self.slope_threshold = slope_threshold
        self.min_confidence = min_confidence
        self._history: deque[float] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prediction: float) -> TrendResult:
        """Append a new prediction and return the current trend."""
        self._history.append(float(prediction))
        return self.get_trend()

    def get_trend(self) -> TrendResult:
        """Compute the trend from the current prediction history."""
        n = len(self._history)

        if n < 3:
            return TrendResult(
                trend=LoadTrend.INSUFFICIENT_DATA,
                slope=0.0,
                slope_per_minute=0.0,
                confidence=0.0,
                n_windows=n,
            )

        y = np.array(self._history, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)

        # Ordinary least squares: y = slope * x + intercept
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = float(((x - x_mean) ** 2).sum())
        ss_xy = float(((x - x_mean) * (y - y_mean)).sum())

        slope = ss_xy / ss_xx if ss_xx > 1e-10 else 0.0

        # R² (coefficient of determination)
        y_pred = slope * x + (y_mean - slope * x_mean)
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y_mean) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        # Convert slope from "per window step" to "per minute"
        windows_per_minute = 60.0 / self.window_step_s
        slope_per_minute = slope * windows_per_minute

        # Classify trend
        if r_squared < self.min_confidence:
            trend = LoadTrend.STABLE
        elif slope > self.slope_threshold:
            trend = LoadTrend.INCREASING
        elif slope < -self.slope_threshold:
            trend = LoadTrend.DECREASING
        else:
            trend = LoadTrend.STABLE

        return TrendResult(
            trend=trend,
            slope=float(slope),
            slope_per_minute=float(slope_per_minute),
            confidence=float(r_squared),
            n_windows=n,
        )

    def reset(self) -> None:
        """Clear prediction history (e.g. on session change)."""
        self._history.clear()

    @property
    def history(self) -> List[float]:
        """Return a copy of the current prediction history."""
        return list(self._history)
