"""Forecast uncertainty from backtest errors (no distributional assumption).

For the champion of each SKU, every scored origin gives a scaled error of cumulative demand over
h weeks,

    e = (actual - forecast) / (level x h)

where `level` is the larger of the SKU's mean weekly units over its last 26 usable weeks and over
its whole history, both as of the origin. Scaling by the SKU's level (not by the forecast) keeps
the errors bounded when a forecast is near zero just before a large order, which is common in
lumpy demand; the whole-history term does the same for a SKU that was dormant for months. Errors are pooled by segment and
horizon bucket (one SKU alone has too few), compressed to a grid of quantiles, and demand over h
weeks is then

    D = max(0, forecast + level x h x e)

These intervals are estimates. Their realised coverage is measured, not assumed (see
`evaluation.calibration`), and the app reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BUCKETS = ((1, 1), (2, 4), (5, 8), (9, 13), (14, 20), (21, 26))
REPRESENTATIVE = {1: 1, 2: 3, 5: 6, 9: 11, 14: 17, 21: 24}
GRID = (np.arange(200) + 0.5) / 200


def bucket_of(h: float) -> int:
    for k, (lo, hi) in enumerate(BUCKETS):
        if h <= hi:
            return k
    return len(BUCKETS) - 1


@dataclass
class ErrorTable:
    samples: dict[tuple[str, int], np.ndarray]   # (segment, bucket) -> scaled errors (quantile grid)
    counts: dict[tuple[str, int], int]

    def errors(self, segment: str, h_weeks: float) -> np.ndarray:
        b = bucket_of(h_weeks)
        for key in ((segment, b), ("ALL", b)):
            if key in self.samples:
                return self.samples[key]
        return np.zeros(1)


def recent_level(Y: np.ndarray, valid: np.ndarray, weeks: int = 26) -> np.ndarray:
    """(W, n) error scale as of each week: max(mean of the last `weeks` usable weeks, mean of all
    usable weeks so far, 1 unit)."""
    v = np.where(valid, Y, 0.0)
    cs = np.concatenate([np.zeros((1, Y.shape[1])), np.cumsum(v, axis=0)])
    cn = np.concatenate([np.zeros((1, Y.shape[1])), np.cumsum(valid, axis=0)])
    W = Y.shape[0]
    lo = np.maximum(np.arange(1, W + 1) - weeks, 0)
    s = cs[1:] - cs[lo]
    c = cn[1:] - cn[lo]
    recent = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    full = np.where(cn[1:] > 0, cs[1:] / np.maximum(cn[1:], 1), np.nan)
    return np.where(np.isfinite(full), np.maximum(np.fmax(recent, full), 1.0), np.nan)


def build_error_table(bt, champion: np.ndarray, segments: np.ndarray, level: np.ndarray, min_samples: int) -> ErrorTable:
    samples, counts = {}, {}
    n = len(champion)
    for k, (lo, _hi) in enumerate(BUCKETS):
        h = REPRESENTATIVE[lo]
        if h not in bt.forecast:
            continue
        f = bt.forecast[h][champion, :, np.arange(n)].T          # (W, n) champion forecast
        scale = level * h
        ok = bt.scored[h] & np.isfinite(f) & (scale > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.where(ok, (bt.actual[h] - f) / scale, np.nan)
        for seg in np.unique(segments):
            v = e[:, segments == seg]
            v = v[np.isfinite(v)]
            if len(v) >= min_samples:
                samples[(seg, k)] = np.quantile(v, GRID)
                counts[(seg, k)] = len(v)
        allv = e[np.isfinite(e)]
        if len(allv):
            samples[("ALL", k)] = np.quantile(allv, GRID)
            counts[("ALL", k)] = len(allv)
    return ErrorTable(samples, counts)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    c = c / c[-1]
    return np.interp(qs, c, v)
