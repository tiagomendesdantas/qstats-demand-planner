"""Constrained-demand recovery benchmark (evaluation layer: the only place baseline is read).

Daily error is not the primary score. Demand here is lumpy (most days sell nothing, some days a
case or a wholesale order), so no estimate of a day's expected demand can match a day's
realisation. The planner consumes weekly totals, so the primary score is the absolute error of
the total imputed per censored episode; bias, weekly error and the share of lost demand recovered
are reported next to it, and daily MAE is kept for completeness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.demand.reconstruction import _episodes


def episode_table(
    sales: np.ndarray, adjusted: np.ndarray, baseline: np.ndarray, censored: np.ndarray, active: np.ndarray
) -> pd.DataFrame:
    """One row per censored episode (SKU x channel): observed, reconstructed and true totals."""
    start, end = _episodes(censored, active)
    T, n, C = censored.shape
    rows = []
    ids = np.where(censored, start, -1)
    for i in range(n):
        for c in range(C):
            col = ids[:, i, c]
            for s in np.unique(col[col >= 0]):
                m = col == s
                rows.append(
                    (
                        i,
                        c,
                        int(s),
                        int(end[s, i, c]),
                        int(m.sum()),
                        float(sales[m, i, c].sum()),
                        float(adjusted[m, i, c].sum()),
                        float(baseline[m, i, c].sum()),
                    )
                )
    return pd.DataFrame(
        rows, columns=["sku_idx", "channel", "start_day", "end_day", "days", "observed", "reconstructed", "baseline"]
    )


def weekly_error(adjusted: np.ndarray, baseline: np.ndarray, censored: np.ndarray) -> float:
    """MAE of weekly totals over SKU-channel-weeks that contain at least one censored day."""
    T = (adjusted.shape[0] // 7) * 7
    r = lambda a: a[:T].reshape(T // 7, 7, *a.shape[1:]).sum(axis=1)  # noqa: E731
    wk_c = r(censored.astype(float)) > 0
    return float(np.abs(r(adjusted) - r(baseline))[wk_c].mean()) if wk_c.any() else np.nan


def score(sales: np.ndarray, adjusted: np.ndarray, baseline: np.ndarray, censored: np.ndarray, active: np.ndarray) -> dict:
    m = censored
    err = adjusted[m] - baseline[m]
    lost = float((baseline[m] - sales[m]).sum())
    recovered = float((adjusted[m] - sales[m]).sum())
    ep = episode_table(sales, adjusted, baseline, censored, active)
    ep_err = ep["reconstructed"] - ep["baseline"]
    return {
        "censored_days": int(m.sum()),
        "episodes": len(ep),
        "episode_mae": float(ep_err.abs().mean()) if len(ep) else np.nan,
        "episode_bias": float(ep_err.mean()) if len(ep) else np.nan,
        "weekly_mae": weekly_error(adjusted, baseline, censored),
        "daily_mae": float(np.abs(err).mean()) if m.any() else np.nan,
        "daily_bias": float(err.mean()) if m.any() else np.nan,
        "lost_units": lost,
        "recovered_units": recovered,
        "recovery_pct": recovered / lost if lost > 0 else np.nan,
    }
