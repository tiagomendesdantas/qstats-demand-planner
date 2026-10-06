"""Worker count for the policy comparison (each worker builds its own world, about 1 GB)."""

from __future__ import annotations

import os


def default_workers(per_worker_gb: float = 1.2, cap: int = 8) -> int:
    """QSTATS_WORKERS if set; otherwise min(cap, CPUs, 75% of physical memory / per-worker need)."""
    env = os.getenv("QSTATS_WORKERS")
    if env:
        return max(1, int(env))
    n = min(cap, os.cpu_count() or 1)
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        n = min(n, int(0.75 * mem_gb / per_worker_gb))
    except (AttributeError, OSError, ValueError):
        pass
    return max(1, n)
