"""Deterministic randomness.

Common random numbers: a draw is a function of (seed, stream name, keys), not of the order in which
the simulation asks for it. Both simulated worlds therefore see the same supplier luck for the same
SKU and order week, whichever policy is running.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def stable_uniform(values: pd.Series, salt: str) -> np.ndarray:
    """Uniform(0, 1) per value, stable across runs and machines (used to assign customers)."""
    hashed = pd.util.hash_pandas_object(values.astype(str) + "|" + salt, index=False).to_numpy()
    return (hashed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def stream(seed: int, *keys: object) -> np.random.Generator:
    """A generator for one named stream, independent of every other stream."""
    digest = hashlib.sha256("|".join(map(str, (seed, *keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))
