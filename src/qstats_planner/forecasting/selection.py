"""Champion selection: per segment, not per SKU.

With one or two years of weekly history and a 10-15 week protection interval, a single SKU has
only two or three non-overlapping backtest windows: far too few to choose among ~25 candidates
without fitting noise. So the champion is chosen per segment, on errors pooled across its SKUs:

    score(model, segment) = mean over SKUs and scored origins of
                            |cumulative forecast - cumulative actual| / (mean weekly units x h)

Rules (all thresholds in the config):
    parsimony   a simpler model within `parsimony_margin` of the best wins
    hysteresis  an incumbent champion is replaced only if the new one is better by `switch_margin`
    SKU override a SKU keeps its own best model only if it beats the segment champion by
                `sku_override_margin` on at least `sku_override_min_blocks` non-overlapping windows
    fallback    a segment with too few scored windows uses the pooled all-SKU champion
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qstats_planner.forecasting.models import ModelSpec


@dataclass
class Selection:
    champion: np.ndarray  # (n,) model index per SKU
    reason: list[str]  # (n,) why
    segment_scores: pd.DataFrame  # segment x model scaled error
    sku_scores: np.ndarray  # (M, n) mean scaled error per SKU (NaN if unscored)


def scaled_errors(forecast: np.ndarray, actual: np.ndarray, scored: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(M, W, n) |F - A| / scale, NaN where not scored or no scale."""
    ok = scored & (scale > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.abs(forecast - actual[None]) / scale[None]
    return np.where(ok[None] & np.isfinite(forecast), e, np.nan)


def _pick(scores: pd.Series, models: list[ModelSpec], parsimony: float) -> int:
    s = scores.dropna()
    if s.empty:
        return -1
    best = s.min()
    ok = s[s <= best * (1 + parsimony)].index
    return int(min(ok, key=lambda m: (models[m].complexity, s[m])))


def select(
    models: list[ModelSpec],
    err: np.ndarray,
    segments: np.ndarray,
    horizon: np.ndarray,
    previous: np.ndarray | None,
    cfg: dict,
    min_origins: int,
) -> Selection:
    f = cfg["forecasting"]
    M, W, n = err.shape
    sku_mean = np.nanmean(err, axis=1) if W else np.full((M, n), np.nan)  # (M, n)
    counts = np.isfinite(err).sum(axis=1)
    # pooled scores per segment
    seg_rows = {}
    for seg in np.unique(segments):
        idx = segments == seg
        tot = np.nansum(np.where(np.isfinite(err[:, :, idx]), err[:, :, idx], 0), axis=(1, 2))
        cnt = np.isfinite(err[:, :, idx]).sum(axis=(1, 2))
        seg_rows[seg] = np.where(cnt >= min_origins * max(1, idx.sum() // 4), tot / np.maximum(cnt, 1), np.nan)
    all_tot = np.nansum(np.where(np.isfinite(err), err, 0), axis=(1, 2))
    all_cnt = np.isfinite(err).sum(axis=(1, 2))
    pooled = pd.Series(np.where(all_cnt > 0, all_tot / np.maximum(all_cnt, 1), np.nan))
    seg_scores = pd.DataFrame(seg_rows, index=[m.name for m in models]).T

    pooled_pick = _pick(pooled, models, f["parsimony_margin"])
    if pooled_pick < 0:
        pooled_pick = next(i for i, m in enumerate(models) if m.family == "ses")
    champion = np.full(n, pooled_pick)
    reason = [""] * n
    for seg in np.unique(segments):
        idx = np.where(segments == seg)[0]
        s = pd.Series(seg_rows[seg])
        pick = _pick(s, models, f["parsimony_margin"])
        why = f"segment champion ({seg})"
        if pick < 0:
            pick, why = pooled_pick, f"too few scored windows in {seg}: all-SKU champion"
        champion[idx] = pick
        for i in idx:
            reason[i] = why

    # SKU override on non-overlapping windows
    for i in range(n):
        h = max(int(horizon[i]), 1)
        blocks = err[:, ::h, i]
        nb = np.isfinite(blocks).sum(axis=1)
        own = (
            np.where(
                nb >= f["sku_override_min_blocks"], np.nanmean(np.where(np.isfinite(blocks), blocks, np.nan), axis=1), np.nan
            )
            if blocks.size
            else np.full(M, np.nan)
        )
        if np.all(np.isnan(own)) or np.isnan(own[champion[i]]):
            continue
        best = int(np.nanargmin(own))
        if own[best] < own[champion[i]] * (1 - f["sku_override_margin"]):
            champion[i] = best
            reason[i] = f"SKU's own best on {int(nb[best])} non-overlapping windows"

    # hysteresis against the incumbent
    if previous is not None:
        for i in range(n):
            p = previous[i]
            if p < 0 or p == champion[i]:
                continue
            seg_s = seg_scores.loc[segments[i]] if segments[i] in seg_scores.index else pooled
            new, old = seg_s.iloc[champion[i]], seg_s.iloc[p]
            if np.isfinite(old) and np.isfinite(new) and new > old * (1 - f["switch_margin"]):
                champion[i] = p
                reason[i] = (
                    reason[i].replace("segment champion", "incumbent kept (switch margin not met)", 1)
                    if "segment" in reason[i]
                    else "incumbent kept (switch margin not met)"
                )
    del counts
    return Selection(champion, reason, seg_scores, sku_mean)
