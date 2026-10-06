"""Configuration loading. Business rules live in config/demo.yaml, never in code."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config" / "demo.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    """Return a fresh copy of the configuration, with dotted-key overrides applied.

    >>> cfg = load_config(**{"inventory.target_service_level": 0.97})
    """
    path = Path(path or os.getenv("QSTATS_CONFIG", DEFAULT_CONFIG))
    cfg = copy.deepcopy(_load(str(path)))
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
