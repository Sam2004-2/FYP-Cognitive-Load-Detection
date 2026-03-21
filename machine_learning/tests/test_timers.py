"""
Unit tests for timing utilities.
"""

import time

import pytest

from src.cle.utils.timers import timer


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
