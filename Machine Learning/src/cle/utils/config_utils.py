"""
Configuration utility functions.

Shared helpers for accessing configuration values across modules.
"""

from typing import Any, Dict, Union


def get_config_value(config: Union[Dict, Any], key: str, default: Any = None) -> Any:
    """
    Get configuration value supporting both Config objects and plain dicts.

    Supports dot notation for nested keys (e.g., 'blink.ear_thresh').

    Args:
        config: Configuration (Config object or dict)
        key: Configuration key (supports dot notation)
        default: Default value if key not found

    Returns:
        Configuration value or default

    Examples:
        >>> config = {"blink": {"ear_thresh": 0.21}}
        >>> get_config_value(config, "blink.ear_thresh", 0.2)
        0.21
        >>> get_config_value(config, "missing.key", "default")
        'default'
    """
    # Try as plain dict first (handles both dict and Config.to_dict())
    if isinstance(config, dict):
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    # If config has a get method that supports dot notation (Config object)
    if hasattr(config, "get") and callable(config.get):
        try:
            return config.get(key, default)
        except (AttributeError, TypeError):
            pass

    return default
