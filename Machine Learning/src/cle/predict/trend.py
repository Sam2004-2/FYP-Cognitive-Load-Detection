"""
Trend detection for cognitive load.

Detects whether cognitive load is INCREASING, DECREASING, or STABLE
by comparing recent predictions to earlier predictions using moving averages.

No ML required - simple statistical comparison.
"""

from collections import deque
from enum import Enum
from typing import List, Optional

import numpy as np


class Trend(str, Enum):
    """Cognitive load trend states."""
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def detect_trend(
    predictions: List[float],
    window: int = 5,
    threshold: float = 0.1,
) -> Trend:
    """
    Detect cognitive load trend from recent predictions.

    Compares the mean of the most recent `window` predictions to
    the mean of the previous `window` predictions.

    Args:
        predictions: List of continuous load predictions [0, 1]
        window: Number of predictions to average (default 5)
        threshold: Minimum change to detect trend (default 0.1 = 10%)

    Returns:
        Trend enum value: INCREASING, DECREASING, STABLE, or INSUFFICIENT_DATA

    Example:
        >>> preds = [0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        >>> detect_trend(preds, window=5, threshold=0.1)
        <Trend.INCREASING: 'INCREASING'>
    """
    if len(predictions) < window * 2:
        return Trend.INSUFFICIENT_DATA

    # Get recent and earlier windows
    recent = predictions[-window:]
    earlier = predictions[-window * 2:-window]

    # Compute means
    recent_avg = np.mean(recent)
    earlier_avg = np.mean(earlier)

    # Compare with threshold
    change = recent_avg - earlier_avg

    if change > threshold:
        return Trend.INCREASING
    elif change < -threshold:
        return Trend.DECREASING
    else:
        return Trend.STABLE


class TrendDetector:
    """
    Stateful trend detector for real-time use.

    Maintains a buffer of recent predictions and provides
    continuous trend detection.

    Example:
        detector = TrendDetector(window=5, threshold=0.1)
        for pred in stream_of_predictions:
            detector.add(pred)
            print(f"Current trend: {detector.get_trend()}")
    """

    def __init__(
        self,
        window: int = 5,
        threshold: float = 0.1,
        max_history: int = 100,
    ):
        """
        Initialize trend detector.

        Args:
            window: Window size for averaging
            threshold: Minimum change threshold for trend detection
            max_history: Maximum predictions to keep in buffer
        """
        self.window = window
        self.threshold = threshold
        self.max_history = max_history
        self._buffer: deque = deque(maxlen=max_history)

    def add(self, prediction: float) -> None:
        """Add a new prediction to the buffer."""
        self._buffer.append(float(prediction))

    def get_trend(self) -> Trend:
        """Get current trend based on buffered predictions."""
        return detect_trend(
            list(self._buffer),
            window=self.window,
            threshold=self.threshold,
        )

    def get_stats(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary with:
                - trend: Current trend
                - recent_avg: Mean of recent window
                - earlier_avg: Mean of earlier window
                - change: Difference (recent - earlier)
                - n_predictions: Number of predictions in buffer
        """
        predictions = list(self._buffer)
        trend = self.get_trend()

        if len(predictions) < self.window * 2:
            return {
                "trend": trend.value,
                "recent_avg": None,
                "earlier_avg": None,
                "change": None,
                "n_predictions": len(predictions),
            }

        recent = predictions[-self.window:]
        earlier = predictions[-self.window * 2:-self.window]

        recent_avg = float(np.mean(recent))
        earlier_avg = float(np.mean(earlier))

        return {
            "trend": trend.value,
            "recent_avg": round(recent_avg, 4),
            "earlier_avg": round(earlier_avg, 4),
            "change": round(recent_avg - earlier_avg, 4),
            "n_predictions": len(predictions),
        }

    def reset(self) -> None:
        """Clear the prediction buffer."""
        self._buffer.clear()

    @property
    def ready(self) -> bool:
        """Check if enough data for trend detection."""
        return len(self._buffer) >= self.window * 2
