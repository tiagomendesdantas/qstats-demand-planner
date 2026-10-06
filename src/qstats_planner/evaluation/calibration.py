"""Are the QStats intervals honest? Realised coverage of the lead-time-demand quantiles.

At every weekly plan, QStats records the quantiles it believed for demand over (lead time +
review period). The realised value uses the lead time that SKU's order would actually have had
that week (a function of SKU and week in the simulation) and the hidden baseline demand. A P90
that holds 90% of outcomes is calibrated; the report shows where it does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.5, 0.8, 0.9, 0.95)


def calibration_rows(env, history: list[dict], start: int, review_days: int) -> pd.DataFrame:
    rows = []
    n = env.n_sku
    net = env.baseline.sum(axis=2)
    cs = np.concatenate([np.zeros((1, n)), np.cumsum(net, axis=0)])
    for h in history:
        t = h["t"]
        if t < start:
            continue
        lt = np.array([env.lead_time(i, t)[0] for i in range(n)])
        end = t + 1 + lt + review_days
        ok = end <= env.n_days
        realised = np.where(ok, cs[np.minimum(end, env.n_days), np.arange(n)] - cs[t + 1], np.nan)
        month = env.days[min(t + 1, env.n_days - 1)].month
        for i in np.where(ok & (env.launch_day <= t))[0]:
            row = {"t": t, "sku_idx": i, "segment": h["segments"][i], "peak": month in (9, 10, 11, 12),
                   "realised": realised[i], "mean": h["ltd_mean"][i]}
            for q in QUANTILES:
                row[f"q{int(q * 100)}"] = h["ltd_q"][q][i]
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(rows: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        out = {"n": len(g)}
        for q in QUANTILES:
            col = f"q{int(q * 100)}"
            out[f"coverage_{col}"] = float((g["realised"] <= g[col]).mean())
            u = g["realised"] - g[col]
            out[f"pinball_{col}"] = float(np.mean(np.maximum(q * u, (q - 1) * u)))
        return pd.Series(out)

    if rows.empty:
        return pd.DataFrame()
    if not by:
        return agg(rows).to_frame().T
    return rows.groupby(by).apply(agg, include_groups=False).reset_index()
