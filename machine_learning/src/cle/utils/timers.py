"""
Timing utilities for CLE.

Context managers for performance tracking.
"""

import time
from contextlib import contextmanager
from typing import Generator

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
