"""Configuration loading utilities.

Loads config/config.yaml (or any YAML config file) into a plain dict
and resolves relative paths against the project root, matching the
convention documented at the top of config.yaml ("All paths are
relative to the project root and are resolved with pathlib at
runtime").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# src/utils/config_loader.py -> src/utils -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Loads a YAML config file into a dict.

    Args:
        config_path: Path to the YAML config file. Defaults to
            config/config.yaml at the project root.

    Returns:
        Parsed configuration as a dictionary. An empty file yields {}.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger.info("Loaded config from %s", config_path)
    return config


def resolve_path(relative_path: str | Path) -> Path:
    """Resolves a path from config.yaml against the project root.

    Args:
        relative_path: Path string as found in config.yaml (e.g.
            "models/yolov8n.pt"), or an already-absolute path.

    Returns:
        An absolute Path. Absolute inputs are returned unchanged.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
