"""
Configuration management for CLE.

Loads and validates YAML configuration files.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for CLE."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self._config = config_dict
        self._validate()

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    @classmethod
    def default(cls) -> "Config":
        """
        Load default configuration.

        Returns:
            Config instance with default settings
        """
        configs_dir = Path(__file__).parent.parent.parent / "configs"
        # Try default.yaml first, fall back to config.yaml
        default_path = configs_dir / "default.yaml"
        if not default_path.exists():
            default_path = configs_dir / "config.yaml"
        return cls.from_yaml(str(default_path))

    def _validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level keys
        required_keys = ["seed", "windows", "quality", "blink", "features_enabled"]
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate windows
        if self._config["windows"]["length_s"] <= 0:
            raise ValueError("windows.length_s must be positive")
        if self._config["windows"]["step_s"] <= 0:
            raise ValueError("windows.step_s must be positive")
        if self._config["windows"]["step_s"] > self._config["windows"]["length_s"]:
            raise ValueError("windows.step_s cannot exceed windows.length_s")

        # Validate quality thresholds
        if not 0 <= self._config["quality"]["max_bad_frame_ratio"] <= 1:
            raise ValueError("quality.max_bad_frame_ratio must be in [0, 1]")

        # Validate blink parameters
        if self._config["blink"]["ear_thresh"] <= 0:
            raise ValueError("blink.ear_thresh must be positive")
        if self._config["blink"]["min_blink_ms"] <= 0:
            raise ValueError("blink.min_blink_ms must be positive")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).

        Args:
            key: Configuration key (e.g., 'windows.length_s')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def hash(self) -> str:
        """
        Generate hash of configuration for versioning.

        Returns:
            MD5 hash of config as hex string
        """
        config_str = yaml.dump(self._config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use default.

    Args:
        config_path: Path to config file, or None for default

    Returns:
        Config instance
    """
    if config_path is None:
        return Config.default()
    return Config.from_yaml(config_path)

