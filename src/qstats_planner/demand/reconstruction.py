"""Censored-demand detection and reconstruction.

Sales are not demand when stock runs out. The planner sees sales and inventory snapshots, never
true demand, and must decide which days were constrained and what demand probably was.

Channels. The two DCs fulfil each other's customers when one is out, so they are pooled into one
"direct" channel; Amazon FBA is its own channel. A day is censored for a channel when that
channel could not serve all demand.

Availability status of a channel-day (planner-side, from snapshots only):
    CONFIRMED_STOCKOUT    nothing sellable at the start of the day
    LIKELY_CONSTRAINED    sold out during the day (closing sellable = 0, some sales)
    UNKNOWN_AVAILABILITY  inventory snapshot missing; a run of zero-sales days that would be
                          unlikely at the usual rate is treated as a stockout
    PROMOTION/LIQUIDATION simulated event windows (announced to the planner)
    NORMAL

Methods (transparent, no deep learning):
    no_adjustment       sales as recorded (the baseline every method must beat)
    pre_post_velocity   mean daily sales on clean days before (and after) the episode
    local_profile       same windows, adjusted for the weekday profile and the seasonal prior
    model_expectation   exponentially weighted level of clean deseasonalised sales at the
                        episode start (what a level model expected), times the day's profile
    censored_gamma      local_profile, plus the information that demand on a sold-out day was at
                        least what sold: an EM loop replaces those days by E[Y | Y >= sales]
                        under a gamma demand distribution with the SKU's dispersion (shape
                        bounded below; the bound was set on the dev SKUs)

`two_sided=False` uses only data before each episode (the estimate available on the day);
`two_sided=True` also uses data after it, up to the as-of date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincc

NOT_ACTIVE, NORMAL, CONFIRMED, LIKELY, UNKNOWN, PROMOTION, LIQUIDATION = range(7)
STATUS_NAMES = {
    NOT_ACTIVE: "NOT_ACTIVE",
    NORMAL: "NORMAL",
    CONFIRMED: "CONFIRMED_STOCKOUT",
    LIKELY: "LIKELY_CONSTRAINED",
    UNKNOWN: "UNKNOWN_AVAILABILITY",
    PROMOTION: "PROMOTION",
    LIQUIDATION: "LIQUIDATION",
}
METHODS = ("no_adjustment", "pre_post_velocity", "local_profile", "model_expectation", "censored_gamma")
DIRECT, AMAZON = 0, 1


@dataclass
class ChannelData:
    sales: np.ndarray  # (T, n, 2)
    opening: np.ndarray  # (T, n, 2) NaN where the snapshot is missing
    closing: np.ndarray
    active: np.ndarray  # (T, n, 2) bool: trading day, SKU launched, channel exists
    warmup: np.ndarray  # (T,) bool: stock was ample, every day is NORMAL
    event: np.ndarray  # (T, n) int: 0 none, PROMOTION, LIQUIDATION
    season: np.ndarray  # (T, n) seasonal prior factor per day


@dataclass
class Reconstruction:
    status: np.ndarray  # (T, n, 2) int8
    adjusted: np.ndarray  # (T, n, 2) units
    imputed: np.ndarray  # (T, n, 2) bool
    confidence: np.ndarray  # (T, n, 2) 0-1 for imputed days, NaN otherwise
    method: str


def channel_data(view, season_daily: np.ndarray) -> ChannelData:
    """Pool the two DCs and keep FBA separate, from what the planner can see."""
    s = view.sales
    sales = np.stack([s[:, :, 0] + s[:, :, 1], s[:, :, 2]], axis=2)
    o, c = view.opening_available, view.closing_available
    opening = np.stack([o[:, :, 0] + o[:, :, 1], o[:, :, 2]], axis=2)
    closing = np.stack([c[:, :, 0] + c[:, :, 1], c[:, :, 2]], axis=2)
    T, n = s.shape[:2]
    day = np.arange(T)[:, None]
    launched = (view.launch_day[None, :] >= 0) & (day >= view.launch_day[None, :])
    fba = view.products["fba_enabled"].to_numpy(bool)
    active = np.stack([launched, launched & fba[None, :]], axis=2) & view.trading[:, None, None]
    warm = np.arange(T) <= view.first_planning_day
    event = np.zeros((T, n), dtype=np.int8)
    ev = view.events()
    for e in ev.itertuples() if len(ev) else []:
        if e.start_day <= view.t:
            event[e.start_day : min(e.end_day, view.t) + 1, e.sku_idx] = PROMOTION if e.kind == "PROMOTION" else LIQUIDATION
    return ChannelData(sales, opening, closing, active, warm, event, season_daily[:T])


# --------------------------------------------------------------------------- status


def classify(cd: ChannelData, unknown_zero_run_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Status per channel-day, and a mask of days whose demand must be estimated."""
    T, n, C = cd.sales.shape
    st = np.full((T, n, C), NOT_ACTIVE, dtype=np.int8)
    act = cd.active
    st[act] = NORMAL
    post = act & ~cd.warmup[:, None, None]
    missing = np.isnan(cd.opening) | np.isnan(cd.closing)
    st[post & missing] = UNKNOWN
    known = post & ~missing
    st[known & (np.nan_to_num(cd.opening) <= 0)] = CONFIRMED
    st[known & (np.nan_to_num(cd.opening) > 0) & (np.nan_to_num(cd.closing) <= 0) & (cd.sales > 0)] = LIKELY
    ev = np.repeat(cd.event[:, :, None], C, axis=2)
    st[(st == NORMAL) & (ev > 0)] = ev[(st == NORMAL) & (ev > 0)]

    # UNKNOWN days: a zero-sales run that would be unlikely at the usual rate is a stockout.
    clean = st == NORMAL
    p0 = np.where(clean.sum(0) > 0, ((cd.sales == 0) & clean).sum(0) / np.maximum(clean.sum(0), 1), 1.0)
    zero_unknown = (st == UNKNOWN) & (cd.sales == 0)
    run = _run_length(zero_unknown, act)
    with np.errstate(divide="ignore"):
        logp = run * np.log(np.clip(p0, 1e-6, 1))[None]
    inferred_out = zero_unknown & (logp < np.log(unknown_zero_run_p))
    censored = (st == CONFIRMED) | (st == LIKELY) | inferred_out
    return st, censored


def _run_length(flag: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Length of the run of `flag` each day belongs to, counting active days only (inactive days
    such as Saturdays neither break nor extend a run)."""
    T = flag.shape[0]
    out = np.zeros(flag.shape, dtype=float)
    cur = np.zeros(flag.shape[1:])
    start = np.zeros(flag.shape, dtype=float)
    for t in range(T):  # forward: run length so far
        cur = np.where(flag[t], cur + 1, np.where(active[t], 0, cur))
        start[t] = cur
    cur = np.zeros(flag.shape[1:])
    for t in range(T - 1, -1, -1):  # backward: carry the run's total length to every day of it
        cur = np.where(flag[t], np.maximum(cur, start[t]), np.where(active[t], 0, cur))
        out[t] = np.where(flag[t], cur, 0)
    return out


# --------------------------------------------------------------------------- helpers


def _episodes(censored: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every day, the first and last day of the censored episode it belongs to."""
    T = censored.shape[0]
    idx = np.arange(T)[:, None, None]
    # a censored episode continues across inactive days (weekends) but breaks on a clean active day
    breaker = active & ~censored
    last_break = np.maximum.accumulate(np.where(breaker, idx, -1), axis=0)
    start = np.where(censored, last_break + 1, -1)
    next_break = np.minimum.accumulate(np.where(breaker, idx, T)[::-1], axis=0)[::-1]
    end = np.where(censored, next_break - 1, -1)
    return start, end


def _window_sums(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum of `values` over days [lo, hi] (inclusive), per element, along axis 0."""
    T = values.shape[0]
    cs = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)], axis=0)
    lo_c = np.clip(lo, 0, T)
    hi_c = np.clip(hi + 1, 0, T)
    hi_c = np.maximum(hi_c, lo_c)
    take = lambda a: np.take_along_axis(cs, a.astype(int), axis=0)  # noqa: E731
    return take(hi_c) - take(lo_c)


def weekday_profile(cd: ChannelData, clean: np.ndarray, days_dow: np.ndarray) -> np.ndarray:
    """Pooled weekday profile of clean sales per trading day (mean 1 over trading weekdays)."""
    prof = np.ones(7)
    s = np.where(clean, cd.sales, 0).sum(axis=(1, 2))
    n = clean.sum(axis=(1, 2))
    for k in range(7):
        m = days_dow == k
        if n[m].sum() > 0:
            prof[k] = s[m].sum() / n[m].sum()
    traded = [k for k in range(7) if n[days_dow == k].sum() > 0]
    prof = prof / prof[traded].mean() if traded else prof
    return prof


def gamma_tail_mean(mu: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[Y | Y >= s] for Y ~ Gamma(shape k, mean mu)."""
    mu = np.maximum(mu, 1e-6)
    theta = mu / k
    x = s / theta
    num = gammaincc(k + 1, x)
    den = np.maximum(gammaincc(k, x), 1e-12)
    out = mu * num / den
    return np.where(np.isfinite(out), np.maximum(out, s), s)


# --------------------------------------------------------------------------- reconstruct


def reconstruct(
    cd: ChannelData,
    method: str,
    days_dow: np.ndarray,
    two_sided: bool = True,
    window: int = 28,
    min_clean: int = 8,
    unknown_zero_run_p: float = 0.02,
    em_iterations: int = 4,
    gamma_min_shape: float = 0.3,
) -> Reconstruction:
    status, censored = classify(cd, unknown_zero_run_p)
    sales = cd.sales
    if method == "no_adjustment":
        return Reconstruction(status, sales.copy(), np.zeros_like(censored), np.full(sales.shape, np.nan), method)

    clean = status == NORMAL
    act = cd.active
    start, end = _episodes(censored, act)
    T = sales.shape[0]
    wd = weekday_profile(cd, clean, days_dow)
    prof = wd[days_dow][:, None, None] * cd.season[:, :, None]  # expected relative level
    prof = np.where(act, prof, 0.0)

    # windows: `window` calendar days x 1.4 (to hold ~`window` trading days) before / after
    span = int(window * 7 / 6)
    pre_lo, pre_hi = start - span, start - 1
    post_lo, post_hi = end + 1, np.minimum(end + span, T - 1)

    def rates(y: np.ndarray, use: np.ndarray, weight: np.ndarray):
        num = _window_sums(np.where(use, y, 0), pre_lo, pre_hi)
        den = _window_sums(np.where(use, weight, 0), pre_lo, pre_hi)
        cnt = _window_sums(use.astype(float), pre_lo, pre_hi)
        if two_sided:
            num = num + _window_sums(np.where(use, y, 0), post_lo, post_hi)
            den = den + _window_sums(np.where(use, weight, 0), post_lo, post_hi)
            cnt = cnt + _window_sums(use.astype(float), post_lo, post_hi)
        return num, den, cnt

    if method == "pre_post_velocity":
        num, den, cnt = rates(sales, clean, np.ones_like(sales))
        est = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0) * act
    elif method in ("local_profile", "censored_gamma"):
        num, den, cnt = rates(sales, clean, prof)
        level = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        est = level * prof
        if method == "censored_gamma":
            likely = (status == LIKELY) & censored
            k = _dispersion(sales, clean, gamma_min_shape)
            y = sales.astype(float).copy()
            for _ in range(em_iterations):
                y_lik = gamma_tail_mean(est, k[None], sales)
                y = np.where(likely, y_lik, sales)
                use = clean | likely
                num, den, cnt = rates(y, use, prof)
                level = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
                est = level * prof
            est = np.where(likely, gamma_tail_mean(est, k[None], sales), est)
    elif method == "model_expectation":
        # exponentially weighted level of clean deseasonalised sales, read at the episode start
        half_life = 20.0
        a = 1 - 0.5 ** (1 / half_life)
        lvl = np.full(sales.shape[1:], np.nan)
        levels = np.empty(sales.shape)
        ratio = np.where(prof > 0, sales / np.maximum(prof, 1e-9), np.nan)
        for t in range(T):
            upd = clean[t]
            first = upd & np.isnan(lvl)
            lvl = np.where(first, ratio[t], lvl)
            lvl = np.where(upd & ~first, a * ratio[t] + (1 - a) * lvl, lvl)
            levels[t] = lvl
        at_start = np.take_along_axis(levels, np.clip(start - 1, 0, T - 1), axis=0)
        est = np.nan_to_num(at_start) * prof
        cnt = _window_sums(clean.astype(float), pre_lo, pre_hi)
    else:
        raise ValueError(f"unknown method {method}")

    est = np.where(censored, est, 0.0)
    adjusted = np.where(censored, np.maximum(sales, est), sales)
    ep_len = np.where(censored, end - start + 1, 0)
    conf = np.where(censored, np.clip(cnt / max(min_clean * 2.5, 1), 0, 1) * np.exp(-ep_len / 60.0), np.nan)
    conf = np.where(censored & (cnt < min_clean), conf * 0.5, conf)
    return Reconstruction(status, adjusted, censored, conf, method)


def _dispersion(sales: np.ndarray, clean: np.ndarray, min_shape: float = 0.3) -> np.ndarray:
    """Gamma shape k per SKU-channel from clean days (method of moments, bounded)."""
    n = clean.sum(axis=0)
    m = np.where(n > 0, np.where(clean, sales, 0).sum(axis=0) / np.maximum(n, 1), 0)
    v = np.where(n > 1, np.where(clean, (sales - m[None]) ** 2, 0).sum(axis=0) / np.maximum(n - 1, 1), 0)
    k = np.where(v > 0, m**2 / np.maximum(v, 1e-9), 1.0)
    return np.clip(k, min_shape, 20.0)
