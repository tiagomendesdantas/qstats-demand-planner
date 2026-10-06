"""Champion selection: per segment, not per SKU.

With one or two years of weekly history and a 7-11 week protection interval, a single SKU has only
three to eight non-overlapping backtest windows in the last 52 weeks: few to choose among ~25
candidates without fitting noise. So the champion is chosen per segment, on errors pooled across
its SKUs:

    score(model, segment) = mean over SKUs and scored origins of
                            |cumulative forecast - cumulative actual| / (mean weekly units x h)

Rules (all thresholds in the config):
    parsimony   a simpler model within `parsimony_margin` of the best wins
    hysteresis  an incumbent champion is replaced only if the new one is better by `switch_margin`
    SKU override a SKU keeps its own best model only if it beats the segment champion by
                `sku_override_margin` on at least `sku_override_min_blocks` non-overlapping windows
                that both models were scored on
    fallback    a segment with too few scored windows uses the pooled all-SKU champion
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qstats_planner.forecasting.models import ModelSpec


@dataclass
class Selection:
    champion: np.ndarray  # (n,) model index per SKU
    reason: list[str]  # (n,) why
    segment_scores: pd.DataFrame  # segment x model scaled error
    sku_scores: np.ndarray  # (M, n) mean scaled error per SKU (NaN if unscored)
    segment_champion: dict = field(default_factory=dict)  # segment -> model index, before SKU overrides


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
    seg_champion = {}
    for seg in np.unique(segments):
        idx = np.where(segments == seg)[0]
        s = pd.Series(seg_rows[seg])
        pick = _pick(s, models, f["parsimony_margin"])
        why = f"segment champion ({seg})"
        if pick < 0:
            pick, why = pooled_pick, f"too few scored windows in {seg}: all-SKU champion"
        seg_champion[str(seg)] = pick
        champion[idx] = pick
        for i in idx:
            reason[i] = why

    # SKU override on non-overlapping windows. Two models are compared only on the windows where both
    # were scored (seasonal naive, for one, has no forecast where last year's week was closed), so a
    # model never wins on a window its rival was not scored on.
    min_blocks = f["sku_override_min_blocks"]

    def own_blocks(i: int) -> np.ndarray:
        return err[:, :: max(int(horizon[i]), 1), i]  # (M, non-overlapping windows)

    def paired(blocks: np.ndarray, a: int, b: int) -> tuple[float, float, int]:
        """Mean error of models a and b over the windows both were scored on, and how many."""
        both = np.isfinite(blocks[a]) & np.isfinite(blocks[b])
        k = int(both.sum())
        if k < min_blocks:
            return np.nan, np.nan, k
        return float(blocks[a, both].mean()), float(blocks[b, both].mean()), k

    for i in range(n):
        blocks = own_blocks(i)
        c = champion[i]
        if blocks.size == 0 or not np.isfinite(blocks[c]).any():
            continue
        best, best_ratio, best_k = -1, np.inf, 0
        for m in range(M):
            if m == c:
                continue
            em, ec, k = paired(blocks, m, c)
            if np.isfinite(em) and ec > 0 and em / ec < best_ratio:
                best, best_ratio, best_k = m, em / ec, k
        if best >= 0 and best_ratio < 1 - f["sku_override_margin"]:
            champion[i] = best
            reason[i] = f"SKU's own best on {best_k} non-overlapping windows"

    # hysteresis against the incumbent: on the windows where both are scored, when there are enough
    # (so a SKU override is not judged on segment scores it can never beat), else on segment scores
    if previous is not None:
        for i in range(n):
            p = previous[i]
            if p < 0 or p == champion[i]:
                continue
            new, old, _ = paired(own_blocks(i), champion[i], p)
            basis = "the SKU's own windows"
            if not (np.isfinite(new) and np.isfinite(old)):
                seg_s = seg_scores.loc[segments[i]] if segments[i] in seg_scores.index else pooled
                new, old, basis = seg_s.iloc[champion[i]], seg_s.iloc[p], "segment windows"
            if np.isfinite(old) and np.isfinite(new) and new > old * (1 - f["switch_margin"]):
                champion[i] = p
                reason[i] = f"incumbent kept (the new pick is not {f['switch_margin']:.0%} better on {basis})"
    del counts
    return Selection(champion, reason, seg_scores, sku_mean, seg_champion)
