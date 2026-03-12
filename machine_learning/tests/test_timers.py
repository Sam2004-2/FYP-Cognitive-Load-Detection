"""
Unit tests for timing utilities.
"""

import time

import pytest

from src.cle.utils.timers import FPSCounter, LatencyTracker, timed, timer


def test_timer_basic():
    """Test that timer context manager measures elapsed time."""
    with timer("test_op", log=False) as t:
        time.sleep(0.05)

    assert t["elapsed"] > 0.04
    assert t["elapsed"] < 0.5


def test_timer_result_key_exists_before_block():
    """Test that the result dict has elapsed=0.0 before block completes."""
    with timer("pre_check", log=False) as t:
        assert "elapsed" in t
        assert t["elapsed"] == 0.0


def test_timer_exception_still_records():
    """Test that timer records elapsed even when exception is raised."""
    result = {}
    with pytest.raises(ValueError):
        with timer("fail_op", log=False) as t:
            result = t
            time.sleep(0.02)
            raise ValueError("intentional")

    assert result["elapsed"] > 0.01


def test_timed_decorator():
    """Test timed decorator returns correct result."""

    @timed
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_timed_decorator_preserves_name():
    """Test timed decorator preserves function name."""

    @timed
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


def test_fps_counter_initial_state():
    """Test FPSCounter initial state."""
    counter = FPSCounter(alpha=0.1)
    assert counter.get_fps() == 0.0
    assert counter.frame_count == 0


def test_fps_counter_after_updates():
    """Test FPSCounter tracks FPS after updates."""
    counter = FPSCounter(alpha=0.5)

    # First update just records the time
    counter.update()
    time.sleep(0.01)
    fps = counter.update()

    assert fps > 0
    assert counter.frame_count == 2


def test_fps_counter_reset():
    """Test FPSCounter reset clears state."""
    counter = FPSCounter()
    counter.update()
    counter.update()
    assert counter.frame_count == 2

    counter.reset()
    assert counter.get_fps() == 0.0
    assert counter.frame_count == 0
    assert counter.last_time is None


def test_latency_tracker_empty():
    """Test LatencyTracker stats for empty tracker."""
    tracker = LatencyTracker()
    stats = tracker.get_stats()
    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["median"] == 0.0


def test_latency_tracker_single_value():
    """Test LatencyTracker with a single measurement."""
    tracker = LatencyTracker()
    tracker.add(0.05)
    stats = tracker.get_stats()
    assert stats["count"] == 1
    assert stats["mean"] == pytest.approx(0.05, abs=1e-9)


def test_latency_tracker_multiple_values():
    """Test LatencyTracker with multiple measurements."""
    tracker = LatencyTracker()
    for val in [0.01, 0.02, 0.03, 0.04, 0.05]:
        tracker.add(val)

    stats = tracker.get_stats()
    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(0.03, abs=1e-9)
    assert stats["median"] == pytest.approx(0.03, abs=1e-9)
    assert stats["max"] == pytest.approx(0.05, abs=1e-9)
    assert stats["p95"] > stats["median"]


def test_latency_tracker_reset():
    """Test LatencyTracker reset clears state."""
    tracker = LatencyTracker()
    tracker.add(0.1)
    tracker.add(0.2)
    assert tracker.total_frames == 2

    tracker.reset()
    assert tracker.total_frames == 0
    assert len(tracker.latencies) == 0
    stats = tracker.get_stats()
    assert stats["count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
