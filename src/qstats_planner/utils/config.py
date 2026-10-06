"""Configuration loading. Business rules live in config/demo.yaml, never in code."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# The repository root: config/, data/ and the SQLite file live under it. QSTATS_ROOT sets it for a
# non-editable install, where this file sits inside site-packages.
ROOT = Path(os.getenv("QSTATS_ROOT") or Path(__file__).resolve().parents[3])
DEFAULT_CONFIG = ROOT / "config" / "demo.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    """Return a fresh copy of the configuration, with dotted-key overrides applied.

    >>> cfg = load_config(**{"inventory.excess_weeks_of_cover": 20})
    """
    # a relative QSTATS_CONFIG is read against the repository root, like every other path here
    path = resolve(path or os.getenv("QSTATS_CONFIG") or DEFAULT_CONFIG)
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


def database_url(cfg: dict) -> str:
    """The configured URL; a relative SQLite path is resolved against the repository root.
    QSTATS_DATABASE_URL overrides it (e.g. a PostgreSQL or Azure SQL URL)."""
    url = os.getenv("QSTATS_DATABASE_URL", cfg["paths"]["database_url"])
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        url = "sqlite:///" + str(resolve(url.removeprefix("sqlite:///")))
    return url
