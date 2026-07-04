"""Configuration loader utility.

Reads config.yaml from the project root and exposes a singleton ``CFG``
dict so every module imports the same parsed object without re-reading disk.

Usage::

    from src.utils.config_loader import CFG

    model_name = CFG["models"]["mcq_generator"]["name"]
    chunk_size = CFG["retriever"]["chunk_size"]
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Project root = two levels up from this file (src/utils/config_loader.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH  = _PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """Load and return the YAML configuration as a plain ``dict``.

    Results are cached after the first call; subsequent calls return the
    same object.  Pass an explicit *path* to load an alternate config file
    (useful in tests).

    Args:
        path: Optional absolute path to a YAML config file.
            Defaults to ``config.yaml`` in the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    config_path = Path(path) if path else _CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure config.yaml exists in the project root."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logger.info("Configuration loaded from: %s", config_path)
    return config


# Singleton – import this directly for convenience
CFG: dict = load_config()
