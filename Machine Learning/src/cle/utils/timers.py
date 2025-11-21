"""
Timing utilities for CLE.

Context managers and decorators for performance tracking.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator, Optional

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(name: str, log: bool = True) -> Generator[dict, None, None]:
    """
    Context manager for timing code blocks.

    Args:
        name: Name of the timed operation
        log: Whether to log the elapsed time

    Yields:
        Dictionary with 'elapsed' key (updated after block completes)

    Example:
        with timer("data_loading") as t:
            data = load_data()
        print(f"Took {t['elapsed']:.2f}s")
    """
    result = {"elapsed": 0.0}
    start = time.perf_counter()

    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed

        if log:
            logger.debug(f"{name} took {elapsed:.3f}s")


def timed(func: Callable) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


class FPSCounter:
    """
    FPS counter for real-time processing.

    Tracks frames per second using exponential moving average.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize FPS counter.

        Args:
            alpha: Smoothing factor for exponential moving average (0-1)
        """
        self.alpha = alpha
        self.fps = 0.0
        self.last_time = None
        self.frame_count = 0

    def update(self) -> float:
        """
        Update FPS counter with new frame.

        Returns:
            Current FPS estimate
        """
        current_time = time.perf_counter()

        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                if self.frame_count == 0:
                    self.fps = instant_fps
                else:
                    self.fps = self.alpha * instant_fps + (1 - self.alpha) * self.fps

        self.last_time = current_time
        self.frame_count += 1
        return self.fps

    def get_fps(self) -> float:
        """
        Get current FPS estimate.

        Returns:
            Current FPS
        """
        return self.fps

    def reset(self) -> None:
        """Reset FPS counter."""
        self.fps = 0.0
        self.last_time = None
        self.frame_count = 0


class LatencyTracker:
    """
    Track latency statistics for processing pipeline.

    Maintains running statistics of processing latency.
    """

    def __init__(self):
        """Initialize latency tracker."""
        self.latencies = []
        self.total_frames = 0

    def add(self, latency: float) -> None:
        """
        Add latency measurement.

        Args:
            latency: Processing latency in seconds
        """
        self.latencies.append(latency)
        self.total_frames += 1

    def get_stats(self) -> dict:
        """
        Get latency statistics.

        Returns:
            Dictionary with mean, median, p95, p99, max latency
        """
        if not self.latencies:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "count": 0,
            }

        import numpy as np

        latencies_arr = np.array(self.latencies)
        return {
            "mean": float(np.mean(latencies_arr)),
            "median": float(np.median(latencies_arr)),
            "p95": float(np.percentile(latencies_arr, 95)),
            "p99": float(np.percentile(latencies_arr, 99)),
            "max": float(np.max(latencies_arr)),
            "count": len(self.latencies),
        }

    def reset(self) -> None:
        """Reset latency tracker."""
        self.latencies = []
        self.total_frames = 0

