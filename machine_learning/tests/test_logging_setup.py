"""
Unit tests for logging setup.
"""

import logging
import tempfile
from pathlib import Path

import pytest

from src.cle.logging_setup import get_logger, setup_logging


def test_setup_logging_returns_logger():
    """Test that setup_logging returns a logger instance."""
    logger = setup_logging(level="DEBUG")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "cle"


def test_setup_logging_level():
    """Test that the logger level is set correctly."""
    logger = setup_logging(level="WARNING")
    assert logger.level == logging.WARNING


def test_setup_logging_console_handler():
    """Test that console handler is attached."""
    logger = setup_logging(level="INFO")
    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "StreamHandler" in handler_types


def test_setup_logging_file_handler():
    """Test that file handler is created when log_dir is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(level="INFO", log_dir=tmpdir, log_file="test.log")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "FileHandler" in handler_types
        assert (Path(tmpdir) / "test.log").exists()


def test_setup_logging_no_file_handler_by_default():
    """Test that no file handler is created when log_dir is None."""
    logger = setup_logging(level="INFO", log_dir=None)
    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "FileHandler" not in handler_types


def test_setup_logging_clears_existing_handlers():
    """Test that calling setup_logging twice doesn't duplicate handlers."""
    setup_logging(level="INFO")
    logger = setup_logging(level="INFO")
    # Should only have one handler (console)
    assert len(logger.handlers) == 1


def test_setup_logging_default_log_file_name():
    """Test that default log file name is cle.log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_logging(level="INFO", log_dir=tmpdir)
        assert (Path(tmpdir) / "cle.log").exists()


def test_setup_logging_custom_format():
    """Test that custom format string is applied."""
    custom_fmt = "%(name)s - %(message)s"
    logger = setup_logging(level="INFO", format_string=custom_fmt)
    formatter = logger.handlers[0].formatter
    assert formatter._fmt == custom_fmt


def test_get_logger_returns_child():
    """Test that get_logger returns a cle.* child logger."""
    logger = get_logger("mymodule")
    assert logger.name == "cle.mymodule"
    assert isinstance(logger, logging.Logger)


def test_get_logger_different_names():
    """Test that different names produce different loggers."""
    logger_a = get_logger("module_a")
    logger_b = get_logger("module_b")
    assert logger_a.name != logger_b.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
