"""
Utility to load config.yaml from anywhere in the project.
"""

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load project configuration from YAML file."""
    if config_path is None:
        # Walk up to find configs/config.yaml
        current = Path(__file__).resolve().parent
        while current != current.parent:
            candidate = current / "configs" / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    if config_path is None:
        raise FileNotFoundError("Could not find configs/config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
