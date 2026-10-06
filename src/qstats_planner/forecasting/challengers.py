"""As-of challenger: statsmodels ETS with parameters fitted by maximum likelihood.

The closed-loop candidates use fixed parameter grids (fast, and every origin comes out of one
pass). This challenger asks whether fitting the parameters per SKU and per origin would do
better. ETS(A, Ad, N) is fitted to the seasonally adjusted rate at every fourth origin of the last
52 weeks and scored on exactly the windows and scaled error used to choose the champion. It runs
once, in the live planning cycle; it is reported, not used to plan.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from qstats_planner.forecasting.pipeline import ForecastState, History


def ets_challenger(h: History, fs: ForecastState, step: int = 4, lookback: int = 52, min_weeks: int = 26) -> pd.DataFrame:
    bt = fs.backtest
    if bt is None:
        raise ValueError("needs the backtest (fit with keep_backtest=True)")
    W, n = h.Y.shape
    hsel = fs.extra["selection_horizon"]
    champ = fs.selection.champion
    with np.errstate(invalid="ignore", divide="ignore"):
        y = h.Y / (h.exposure[:, None] * h.season)
    es = h.exposure[:, None] * h.season
    cs = np.cumsum(np.where(h.valid, h.Y, 0.0), axis=0)
    cn = np.cumsum(h.valid, axis=0)
    mean_level = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
    rows = []
    for i in range(n):
        hh = int(hsel[i])
        e_ets, e_ch = [], []
        for w in range(max(W - lookback, 0), W - hh, step):
            if not bt.scored[hh][w, i] or not np.isfinite(mean_level[w, i]) or mean_level[w, i] <= 0:
                continue
            hist = y[: w + 1, i][h.valid[: w + 1, i]]
            if len(hist) < min_weeks:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ExponentialSmoothing(hist, trend="add", damped_trend=True, initialization_method="estimated").fit()
                    path = np.maximum(fit.forecast(hh), 0)
            except Exception:
                continue
            f = float((path * es[w + 1 : w + 1 + hh, i]).sum())
            scale = mean_level[w, i] * hh
            e_ets.append(abs(f - bt.actual[hh][w, i]) / scale)
            e_ch.append(abs(bt.forecast[hh][champ[i], w, i] - bt.actual[hh][w, i]) / scale)
        rows.append(
            {
                "sku_idx": i,
                "segment": fs.segments[i],
                "windows": len(e_ets),
                "ets_error": float(np.mean(e_ets)) if e_ets else np.nan,
                "champion_error": float(np.mean(e_ch)) if e_ch else np.nan,
            }
        )
    return pd.DataFrame(rows)
