"""Helpers for loading configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        path: Location of the YAML file.

    Returns:
        Parsed configuration dictionary.
    """

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
