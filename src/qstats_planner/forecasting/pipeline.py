"""Forecasting pipeline: reconstructed history in, champion forecasts with uncertainty out.

Used, unchanged, in two places: inside the weekly closed-loop simulation (World Q) and in the
live planning cycle the app shows. What is backtested is what runs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qstats_planner.demand.segmentation import assign_segments, segment_features
from qstats_planner.forecasting.backtest import Backtest, run_backtest, weekly_metrics
from qstats_planner.forecasting.models import ModelSpec, candidate_models, run_states, weekly_paths
from qstats_planner.forecasting.selection import Selection, scaled_errors, select
from qstats_planner.forecasting.uncertainty import REPRESENTATIVE, ErrorTable, build_error_table, recent_level

SELECTION_HORIZONS = (6, 8, 11, 14)


@dataclass
class History:
    """Weekly planner-side history (complete weeks up to the plan date)."""

    Y: np.ndarray            # (W, n) reconstructed weekly units
    imputed: np.ndarray      # (W, n) units of Y that were imputed
    observed: np.ndarray     # (W, n) units actually sold
    valid: np.ndarray        # (W, n) week may update a model
    exposure: np.ndarray     # (W,) trading days
    season: np.ndarray       # (W, n) seasonal prior factor (ones when the prior is off)
    launch_week: np.ndarray  # (n,)
    seasonal_group: np.ndarray  # (n,)


@dataclass
class ForecastState:
    models: list[ModelSpec]
    selection: Selection
    errors: ErrorTable
    segments: np.ndarray
    features: pd.DataFrame
    fitted_week: int
    backtest: Backtest | None = None
    extra: dict = field(default_factory=dict)


def _adjusted(h: History, seasonal: bool) -> np.ndarray:
    s = h.season if seasonal else np.ones_like(h.season)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = h.Y / (h.exposure[:, None] * s)
    return np.where(h.valid, y, np.nan)


def selection_horizon(protection_weeks: np.ndarray) -> np.ndarray:
    hs = np.array(SELECTION_HORIZONS)
    return hs[np.abs(hs[None, :] - np.asarray(protection_weeks)[:, None]).argmin(axis=1)]


def fit(h: History, protection_weeks: np.ndarray, cfg: dict, use_prior: bool,
        previous: ForecastState | None = None, keep_backtest: bool = False) -> ForecastState:
    f = cfg["forecasting"]
    W, n = h.Y.shape
    models = candidate_models(cfg, with_prior=use_prior)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_plain, y_seas = _adjusted(h, False), _adjusted(h, True)
        states = run_states(models, y_plain, y_seas, h.valid)
        es_plain = np.repeat(h.exposure[:, None], n, axis=1)
        es_seas = es_plain * h.season
        horizons = tuple(sorted(set(SELECTION_HORIZONS) | set(REPRESENTATIVE.values())))
        first_origin = h.launch_week + 8
        bt = run_backtest(models, states, h.Y, h.imputed, es_plain, es_seas, h.exposure, first_origin,
                          horizons, f["max_imputed_share_in_scoring_window"])
        weeks_since = W - h.launch_week
        feat = segment_features(h.Y, h.valid, y_seas, weeks_since)
        segments = assign_segments(feat, h.seasonal_group, cfg)
        hsel = selection_horizon(protection_weeks)
        err = np.full((len(models), W, n), np.nan)
        recent = np.arange(W) >= W - 52
        cs = np.cumsum(np.where(h.valid, h.Y, 0.0), axis=0)
        cn = np.cumsum(h.valid, axis=0)
        mean_level = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)   # as of each origin
        for hh in SELECTION_HORIZONS:
            cols = hsel == hh
            if not cols.any():
                continue
            scale = mean_level * hh
            e = scaled_errors(bt.forecast[hh], bt.actual[hh], bt.scored[hh] & recent[:, None], scale)
            err[:, :, cols] = e[:, :, cols]
        prev = previous.selection.champion if previous is not None else None
        if previous is not None and [m.name for m in previous.models] != [m.name for m in models]:
            prev = None
        sel = select(models, err, segments, hsel, prev, cfg, f["min_origins_for_selection"])
        errors = build_error_table(bt, sel.champion, segments, recent_level(h.Y, h.valid), f["min_errors_for_sku_quantiles"])
    return ForecastState(models, sel, errors, segments, feat, W - 1, bt if keep_backtest else None,
                         {"selection_error": err if keep_backtest else None, "selection_horizon": hsel})


def forecast(h: History, fs: ForecastState, cfg: dict, use_prior: bool, future_exposure: np.ndarray,
             future_season: np.ndarray, horizon_weeks: int) -> np.ndarray:
    """(H, n) champion unit forecasts for the next `horizon_weeks` weeks, from the latest week."""
    W, n = h.Y.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_plain, y_seas = _adjusted(h, False), _adjusted(h, True)
        champs = np.unique(fs.selection.champion)
        models = [fs.models[c] for c in champs]
        states = run_states(models, y_plain, y_seas, h.valid)
        es_plain = np.repeat(future_exposure[:horizon_weeks, None], n, axis=1)
        es_seas = es_plain * future_season[:horizon_weeks]
        paths = weekly_paths(states, models, W - 1, horizon_weeks, es_plain, es_seas, h.Y, h.exposure)
    out = np.zeros((horizon_weeks, n))
    for k, c in enumerate(champs):
        cols = fs.selection.champion == c
        out[:, cols] = paths[k][:, cols]
    # a SKU with no usable history yet: carry its mean weekly rate forward
    bad = ~np.isfinite(out)
    if bad.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.nanmean(np.where(h.valid, h.Y / h.exposure[:, None], np.nan), axis=0)
        fill = np.nan_to_num(rate)[None, :] * es_plain
        out = np.where(bad, fill, out)
    return np.maximum(out, 0)


def to_daily(weekly_fc: np.ndarray, future_days: pd.DatetimeIndex, trading: np.ndarray, season_daily: np.ndarray) -> np.ndarray:
    """Spread each week's units over its trading days, in proportion to the daily season factor."""
    H, n = weekly_fc.shape
    D = H * 7
    w = (trading[:D].astype(float)[:, None]) * season_daily[:D]
    wk = w.reshape(H, 7, n).sum(axis=1)
    share = np.where(wk[:, None, :] > 0, w.reshape(H, 7, n) / np.maximum(wk[:, None, :], 1e-12), 0)
    return (share * weekly_fc[:, None, :]).reshape(D, n)


def diagnostics(h: History, fs: ForecastState, product_idx: np.ndarray | None = None) -> pd.DataFrame:
    """One row per SKU x model: one-week-ahead MAE/WAPE/RMSE/bias over the last 52 scored weeks,
    and the cumulative-horizon scaled error used for selection."""
    bt = fs.backtest
    if bt is None:
        raise ValueError("fit(..., keep_backtest=True) is needed for diagnostics")
    W, n = h.Y.shape
    recent = (np.arange(W) >= W - 53)[:, None]
    ok = bt.scored[1] & recent
    rows = []
    err = fs.extra["selection_error"]
    for m, spec in enumerate(fs.models):
        for i in range(n):
            met = weekly_metrics(bt.week1_forecast[m][:, i], bt.week1_actual[:, i], ok[:, i])
            rows.append({"sku_idx": i, "model": spec.name, "champion": fs.selection.champion[i] == m,
                         "cum_scaled_error": float(np.nanmean(err[m, :, i])) if np.isfinite(err[m, :, i]).any() else np.nan,
                         **met})
    return pd.DataFrame(rows)
