"""Load and merge YAML configuration files."""

import os
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str = None) -> dict:
    """Load configuration from robot_config.yaml, with settings.yaml overrides."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply local settings overrides if present
    settings_path = os.path.join(os.path.dirname(config_path), 'settings.yaml')
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            overrides = yaml.safe_load(f)
        if overrides:
            _deep_merge(config, overrides)

    return config
