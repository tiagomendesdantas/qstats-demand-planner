"""Rolling-origin (walk-forward) backtests. Never a random split.

From every origin w (end of a week) each model forecasts the following weeks using only data up
to w; the forecast is compared with what happened. Because model states are computed once for
the whole history (see `models.run_states`), the forecasts from all origins come out of one pass.

Two error views are kept:
    cumulative  total demand over the next h weeks, the quantity a replenishment decision rests
                on (h = the SKU's protection interval: lead time + review period)
    weekly      each single week ahead, for the usual WAPE / MAE / RMSE / bias diagnostics

Windows whose actuals are mostly reconstructed (imputed share above a threshold) are not scored,
so the reconstruction method does not grade the forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qstats_planner.forecasting.models import ModelSpec, States, phi_sums


@dataclass
class Backtest:
    horizons: tuple[int, ...]
    forecast: dict[int, np.ndarray]  # h -> (M, W, n) cumulative forecast over (w, w+h]
    actual: dict[int, np.ndarray]  # h -> (W, n) cumulative actual
    scored: dict[int, np.ndarray]  # h -> (W, n) bool: window complete, mostly observed, model available
    week1_forecast: np.ndarray  # (M, W, n) one-week-ahead forecast made at w for week w+1
    week1_actual: np.ndarray  # (W, n)


def run_backtest(
    models: list[ModelSpec],
    states: States,
    Y: np.ndarray,
    imputed_units: np.ndarray,
    es_plain: np.ndarray,
    es_seas: np.ndarray,
    exposure: np.ndarray,
    first_origin: np.ndarray,
    horizons: tuple[int, ...],
    max_imputed_share: float,
) -> Backtest:
    """Y (W, n) adjusted units; es_* (W, n) exposure x season for the same weeks."""
    M, W, n = states.level.shape
    H = max(horizons)
    ps = phi_sums(states.phi, H)
    cum_f = np.zeros((M, W, n))
    out_f: dict[int, np.ndarray] = {}
    week1 = np.full((M, W, n), np.nan)
    # shift forward by h weeks, padding with NaN; works when h exceeds the history length
    pad = lambda a, h: np.concatenate([a[h:], np.full((min(h, len(a)), *a.shape[1:]), np.nan)], axis=0)  # noqa: E731
    for h in range(1, H + 1):
        es_p, es_s = pad(es_plain, h), pad(es_seas, h)
        step = np.empty((M, W, n))
        for m, spec in enumerate(models):
            if spec.family == "snaive":
                past = (
                    np.concatenate([np.full((52 - h, n), np.nan), Y[: W - (52 - h)]], axis=0)
                    if 52 - h < W
                    else np.full((W, n), np.nan)
                )
                e_then = np.concatenate([np.full(52 - h, np.nan), exposure[: W - (52 - h)]]) if 52 - h < W else np.full(W, np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    step[m] = np.where(e_then[:, None] > 0, past * es_p / e_then[:, None], np.nan)
            else:
                es = es_s if spec.seasonal else es_p
                rate = states.level[m] + states.trend[m] * ps[m, h - 1]
                step[m] = np.maximum(rate, 0) * es
        if h == 1:
            week1 = step.copy()
        cum_f = cum_f + step
        if h in horizons:
            out_f[h] = cum_f.copy()

    cs_y = np.concatenate([np.zeros((1, n)), np.cumsum(Y, axis=0)], axis=0)
    cs_i = np.concatenate([np.zeros((1, n)), np.cumsum(imputed_units, axis=0)], axis=0)
    out_a, out_s = {}, {}
    w_idx = np.arange(W)
    for h in horizons:
        end = np.minimum(w_idx + 1 + h, W)
        complete = (w_idx + h) < W
        a = cs_y[end] - cs_y[w_idx + 1]
        imp = cs_i[end] - cs_i[w_idx + 1]
        share = np.where(a > 0, imp / np.maximum(a, 1e-9), 0.0)
        out_a[h] = a
        out_s[h] = complete[:, None] & (w_idx[:, None] >= first_origin[None, :]) & (share <= max_imputed_share)
    week1_actual = np.concatenate([Y[1:], np.full((1, n), np.nan)], axis=0)
    return Backtest(tuple(horizons), out_f, out_a, out_s, week1, week1_actual)


def weekly_metrics(fc: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> dict:
    """MAE, WAPE, RMSE and bias of one-week-ahead forecasts over the masked cells."""
    m = mask & np.isfinite(fc) & np.isfinite(actual)
    if not m.any():
        return {"mae": np.nan, "wape": np.nan, "rmse": np.nan, "bias": np.nan, "n": 0}
    e = fc[m] - actual[m]
    tot = actual[m].sum()
    return {
        "mae": float(np.abs(e).mean()),
        "wape": float(np.abs(e).sum() / tot) if tot > 0 else np.nan,
        "rmse": float(np.sqrt((e**2).mean())),
        "bias": float(e.sum() / tot) if tot > 0 else np.nan,
        "n": int(m.sum()),
    }
