"""Representative SKU populations, chosen systematically and without looking ahead.

Candidates and their profiles are computed on the first `selection_weeks` of data only. A SKU that
sold well in year one and died in year two stays in the population: that is a cost a planner
really faces. New products are drawn by launch date alone (launched in the window after the fork),
never by how they sold.

Two disjoint populations come out: `demo` (reported) and `dev` (every tuning decision is made on
it, so the demo results are not tuned on themselves).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.demand.features import sku_metrics
from qstats_planner.utils.rng import stream

# Share of the non-new population per profile. Rare-but-important profiles are over-represented
# relative to the catalogue, on purpose: the demo has to show how each is handled.
PROFILE_QUOTAS = {
    "SMOOTH": 0.28,
    "ERRATIC": 0.17,
    "INTERMITTENT": 0.04,
    "LUMPY": 0.15,
    "SEASONAL_KEYWORD": 0.11,
    "DECLINING": 0.09,
    "GROWING": 0.08,
    "SHORT_HISTORY": 0.08,
}


def assign_profile(m: pd.DataFrame, first_seen_week: pd.Series, cfg: dict) -> pd.Series:
    trend = cfg["population"]["profile_trend_threshold"]
    short = first_seen_week >= cfg["population"]["selection_weeks"] // 2
    prof = np.select(
        [short, m["seasonal_keyword"], m["trend_strength"] <= -trend, m["trend_strength"] >= trend],
        ["SHORT_HISTORY", "SEASONAL_KEYWORD", "DECLINING", "GROWING"],
        default=m["sb_class"].to_numpy(),
    )
    return pd.Series(prof, index=m.index)


def _take(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n SKUs spread evenly over the pool's volume terciles."""
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    band = pd.qcut(pool["total_units"].rank(method="first"), 3, labels=False) if len(pool) >= 3 else 0
    pool = pool.assign(_band=band)
    picks = []
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    for b, k in enumerate(per):
        part = pool[pool["_band"] == b]
        idx = rng.permutation(len(part))[: min(k, len(part))]
        picks.append(part.iloc[idx])
    out = pd.concat(picks)
    if len(out) < n:  # a thin tercile: fill from the rest of the pool
        rest = pool.drop(out.index)
        out = pd.concat([out, rest.iloc[rng.permutation(len(rest))[: n - len(out)]]])
    return out.drop(columns="_band")


def select_populations(daily: pd.DataFrame, life: pd.DataFrame, data_start: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    pop = cfg["population"]
    seed = cfg["random_seed"]
    sel_end = pd.Timestamp(data_start) + pd.Timedelta(weeks=pop["selection_weeks"]) - pd.Timedelta(days=1)
    fork_start = pd.Timestamp(data_start) + pd.Timedelta(weeks=cfg["simulation"]["fork_week"] - 1)

    m = sku_metrics(daily, cfg, end=sel_end)
    life = life.set_index("sku")
    first = m["sku"].map(life["first_seen_date"])
    first_week = ((first - pd.Timestamp(data_start)).dt.days // 7 + 1).astype(int)
    latest_launch = pop["selection_weeks"] - pop["min_weeks_of_history_at_fork"]
    alive_at_end = m["sku"].map(life["last_seen_date"]) > sel_end - pd.Timedelta(weeks=4)
    eligible = (m["total_units"] >= pop["min_units_in_selection_window"]) & (first_week <= latest_launch) & alive_at_end
    cand = m[eligible].copy()
    cand["profile"] = assign_profile(cand, first_week[eligible], cfg)

    n_new = int(round(pop["demo_sku_count"] * pop["new_product_share"]))
    n_dev_new = int(round(pop["dev_sku_count"] * pop["new_product_share"]))
    chosen = []
    remaining = cand
    for name, total in (("demo", pop["demo_sku_count"] - n_new), ("dev", pop["dev_sku_count"] - n_dev_new)):
        rng = stream(seed, "population", name)
        picks = []
        for profile, share in PROFILE_QUOTAS.items():
            k = int(round(total * share))
            picks.append(_take(remaining[remaining["profile"] == profile], k, rng))
        got = pd.concat(picks)
        short = total - len(got)
        if short > 0:
            rest = remaining.drop(got.index)
            got = pd.concat([got, rest.iloc[rng.permutation(len(rest))[:short]]])
        got = got.iloc[:total].assign(population=name)
        chosen.append(got)
        remaining = remaining.drop(got.index)

    # New products: launched after the fork, drawn by launch date only.
    lo, hi = pop["new_product_launch_weeks"]
    launch_week = (life["first_seen_date"] - pd.Timestamp(data_start)).dt.days // 7 + 1
    new_pool = life[(launch_week >= lo) & (launch_week <= hi)].index.difference(cand.index)
    new_pool = pd.Index(sorted(set(new_pool) - set(m.loc[eligible, "sku"])))
    rng = stream(seed, "population", "new")
    order = new_pool[rng.permutation(len(new_pool))]
    for name, k, offset in (("demo", n_new, 0), ("dev", n_dev_new, n_new)):
        skus = order[offset : offset + k]
        chosen.append(pd.DataFrame({"sku": skus, "profile": "NEW_PRODUCT", "population": name}))

    out = pd.concat(chosen, ignore_index=True)
    out["first_seen_date"] = out["sku"].map(life["first_seen_date"])
    out["last_seen_date"] = out["sku"].map(life["last_seen_date"])
    out["disappeared"] = out["sku"].map(life["disappeared"])
    out["fork_start"] = fork_start
    return out
