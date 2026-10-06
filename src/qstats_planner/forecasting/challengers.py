"""As-of challenger: statsmodels ETS with parameters fitted by maximum likelihood.

The closed-loop candidates use fixed parameter grids (fast, and every origin comes out of one
pass). This challenger asks whether fitting the parameters per SKU and per origin would do
better. To compare like with like, both sides are out of sample:

    champion   the model chosen with data up to `holdout_weeks` before the plan date
    ETS        ETS(A, Ad, N) fitted by maximum likelihood at each origin, on the seasonally
               adjusted rate, with data up to that origin
    windows    origins in the hold-out (every `step` weeks), each scored on cumulative demand over
               the SKU's selection horizon, with the same scaled error used to choose champions;
               a window counts only if both models have a forecast for it

It runs once, in the live planning cycle; it is reported, not used to plan.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from qstats_planner.forecasting.pipeline import ForecastState, History


def truncate(h: History, weeks: int) -> History:
    """The history as it stood `weeks` weeks before its end."""
    k = h.Y.shape[0] - weeks
    return History(h.Y[:k], h.imputed[:k], h.observed[:k], h.valid[:k], h.exposure[:k], h.season[:k],
                   h.launch_week, h.seasonal_group)


def ets_challenger(h: History, fs: ForecastState, early: ForecastState, holdout_weeks: int = 26, step: int = 4,
                   min_weeks: int = 26) -> pd.DataFrame:
    """h: the history the backtest was fitted on; fs: its fit (with backtest); early: the fit as of
    `holdout_weeks` before the end (its champions are scored out of sample)."""
    bt = fs.backtest
    if bt is None:
        raise ValueError("needs the backtest (fit with keep_backtest=True)")
    W, n = h.Y.shape
    hsel = fs.extra["selection_horizon"]
    champ = early.selection.champion
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
        for w in range(W - holdout_weeks, W - hh, step):
            if not bt.scored[hh][w, i] or not np.isfinite(mean_level[w, i]) or mean_level[w, i] <= 0:
                continue
            f_ch = bt.forecast[hh][champ[i], w, i]
            if not np.isfinite(f_ch):
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
            e_ch.append(abs(f_ch - bt.actual[hh][w, i]) / scale)
        rows.append(
            {
                "sku_idx": i,
                "segment": fs.segments[i],
                "champion_model": early.models[champ[i]].name,
                "windows": len(e_ets),
                "ets_error": float(np.mean(e_ets)) if e_ets else np.nan,
                "champion_error": float(np.mean(e_ch)) if e_ch else np.nan,
            }
        )
    return pd.DataFrame(rows)
