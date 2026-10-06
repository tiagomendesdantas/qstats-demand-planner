"""Demand over the protection interval, as a distribution.

The protection interval is the supplier lead time L plus the review period R: an order placed now
must cover demand until the order placed at the next review arrives. L is uncertain (its
distribution comes from `lead_time`), and demand over any fixed horizon is uncertain (backtest
errors, `forecasting.uncertainty`). The two combine as a mixture, computed without simulation:

    P(D <= d) = sum_k p(L = l_k) * P( F(x_k) + level * x_k/7 * e <= d ),   x_k = l_k + R

where F(x) is the cumulative point forecast over the next x days, `level` the SKU's recent mean
weekly units and e the pooled scaled errors at that horizon. Quantiles of D are read from the
weighted mixture of those points (floored at zero). Approximations: the lead-time distribution is
compressed to at most 16 points, errors are stored as 200 quantiles per segment and horizon
bucket, and lead time is assumed independent of forecast error.

    safety stock     = Q_alpha(D) - E[D]
    order-up-to S    = Q_alpha(D)                     (alpha = cycle service level)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qstats_planner.forecasting.uncertainty import ErrorTable, weighted_quantiles
from qstats_planner.inventory.lead_time import LeadTimeDistribution


@dataclass
class LTDResult:
    mean: np.ndarray  # (n,) E[D]
    quantiles: dict  # q -> (n,)
    target: np.ndarray  # (n,) Q_alpha
    samples: list  # per SKU (values, weights), for service and shortfall calculations


def cumulative_forecast(daily_fc: np.ndarray, days: np.ndarray) -> np.ndarray:
    """daily_fc (D, n) expected units per day ahead; returns (len(days), n) cumulative totals."""
    cs = np.concatenate([np.zeros((1, daily_fc.shape[1])), np.cumsum(daily_fc, axis=0)], axis=0)
    d = np.clip(np.asarray(days, int), 0, daily_fc.shape[0])
    return cs[d]


def _compact(dist: LeadTimeDistribution, max_points: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Merge a lead-time distribution onto at most `max_points` support points (by quantile)."""
    if len(dist.days) <= max_points:
        return dist.days, dist.prob
    c = np.cumsum(dist.prob)
    edges = np.linspace(0, 1, max_points + 1)[1:-1]
    groups = np.searchsorted(edges, c - dist.prob / 2)
    days = np.array([np.average(dist.days[groups == g], weights=dist.prob[groups == g]) for g in np.unique(groups)])
    prob = np.array([dist.prob[groups == g].sum() for g in np.unique(groups)])
    return np.round(days), prob


def lead_time_demand(
    daily_fc: np.ndarray,
    lt: list[LeadTimeDistribution],
    review_days: float,
    segments: np.ndarray,
    errors: ErrorTable,
    alpha: np.ndarray,
    level: np.ndarray,
    qs=(0.5, 0.8, 0.9, 0.95),
    extra_days: np.ndarray | None = None,
) -> LTDResult:
    n = daily_fc.shape[1]
    mean = np.zeros(n)
    target = np.zeros(n)
    quant = {q: np.zeros(n) for q in qs}
    samples = []
    extra = np.zeros(n) if extra_days is None else extra_days
    lvl = np.nan_to_num(level)
    for i in range(n):
        days, prob = _compact(lt[i])
        horizon = days + review_days + extra[i]
        F = cumulative_forecast(daily_fc[:, [i]], horizon)[:, 0]  # (K,)
        vals, wts = [], []
        for k in range(len(horizon)):
            e = errors.errors(segments[i], horizon[k] / 7.0)
            vals.append(np.maximum(F[k] + lvl[i] * horizon[k] / 7.0 * e, 0.0))
            wts.append(np.full(len(e), prob[k] / len(e)))
        v, w = np.concatenate(vals), np.concatenate(wts)
        samples.append((v, w))
        mean[i] = float((v * w).sum() / w.sum())
        qv = weighted_quantiles(v, w, np.array([*qs, alpha[i]]))
        for j, q in enumerate(qs):
            quant[q][i] = qv[j]
        target[i] = qv[-1]
    return LTDResult(mean, quant, target, samples)


def service_level(samples: tuple[np.ndarray, np.ndarray], stock: float) -> float:
    """P(D <= stock): the cycle service level a given position delivers."""
    v, w = samples
    return float(w[v <= stock].sum() / w.sum())


def expected_shortfall(samples: tuple[np.ndarray, np.ndarray], stock: float) -> float:
    """E[(D - stock)+]: expected units short over the protection interval."""
    v, w = samples
    return float((np.maximum(v - stock, 0) * w).sum() / w.sum())


def normal_approximation(
    mean_daily: np.ndarray, sd_daily: np.ndarray, lt_mean: np.ndarray, lt_sd: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Textbook cross-check: SS = z * sqrt(L sigma_d^2 + d^2 sigma_L^2)."""
    return z * np.sqrt(lt_mean * sd_daily**2 + mean_daily**2 * lt_sd**2)
