"""Builds the simulated worlds: one shared history under Legacy, then a fork per policy variant."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from qstats_planner.forecasting.seasonal_prior import SeasonalPrior
from qstats_planner.replenishment.policy import PlanSettings
from qstats_planner.simulation.engine import Engine
from qstats_planner.simulation.environment import Environment, build_environment
from qstats_planner.simulation.policies.legacy import LegacyPlanner
from qstats_planner.simulation.policies.qstats import QStatsPlanner
from qstats_planner.utils.config import resolve

LEGACY_REF = "legacy_30d"
QSTATS_REF = "qstats"
LEGACY_FAMILY = ["legacy_30d", "legacy_15d", "legacy_45d", "legacy_60d", "legacy_75d", "legacy_90d", "legacy_120d"]
QSTATS_FAMILY = ["qstats", "qstats_sl70", "qstats_sl80", "qstats_sl90", "qstats_sl95", "qstats_sl98"]
FACTORIAL = [(u, s, p) for u in (0, 1) for s in (0, 1) for p in (0, 1)]


def factorial_name(u: int, s: int, p: int) -> str:
    if (u, s, p) == (1, 1, 1):
        return QSTATS_REF
    return f"ablation_u{u}s{s}p{p}"


def variant_names() -> list[str]:
    names = LEGACY_FAMILY + ["legacy_30d_prior"] + QSTATS_FAMILY[1:] + [factorial_name(*f) for f in FACTORIAL]
    return list(dict.fromkeys(names))


@dataclass
class Inputs:
    pop: pd.DataFrame
    lines: pd.DataFrame
    descriptions: pd.Series
    trading: pd.DatetimeIndex
    daily: pd.DataFrame


def load_inputs(cfg: dict) -> Inputs:
    d = resolve(cfg["paths"]["processed_dir"])
    daily = pd.read_parquet(d / "daily_demand.parquet")
    return Inputs(
        pop=pd.read_parquet(d / "population.parquet"),
        lines=pd.read_parquet(d / "demand_lines.parquet"),
        descriptions=daily.drop_duplicates("sku").set_index("sku")["description"],
        trading=pd.DatetimeIndex(pd.read_parquet(d / "trading_days.parquet")["date"]),
        daily=daily,
    )


def make_prior(inp: Inputs, env: Environment, cfg: dict) -> SeasonalPrior:
    kw = cfg["segmentation"]["seasonal_keywords"]
    sp = SeasonalPrior.fit(
        inp.daily,
        set(inp.pop["sku"]),
        env.days[0],
        cfg["population"]["selection_weeks"],
        kw,
        cfg["forecasting"]["seasonal_prior_shrink_k"],
    )
    return sp.for_skus(env.products["description"], kw)


def make_policy(name: str, cfg: dict, prior: SeasonalPrior):
    if name.startswith("legacy_"):
        days = float(name.split("_")[1].rstrip("d"))
        return LegacyPlanner(cfg, safety_days=days, seasonal_prior=prior if name.endswith("_prior") else None, name=name)
    if name == QSTATS_REF:
        return QStatsPlanner(cfg, prior, PlanSettings(), name=name)
    if name.startswith("qstats_sl"):
        return QStatsPlanner(cfg, prior, PlanSettings(service_level=int(name[-2:]) / 100), name=name)
    if name.startswith("ablation_"):
        u, s, p = (int(name[i]) for i in (10, 12, 14))
        return QStatsPlanner(cfg, prior, PlanSettings(uncensor=bool(u), seasonal=bool(s), probabilistic=bool(p)), name=name)
    raise ValueError(name)


def build_world(
    inp: Inputs, cfg: dict, population: str, supply_seed: int, scenario: str = "base"
) -> tuple[Environment, SeasonalPrior, Engine]:
    """The shared history: the Legacy planner runs the business until the fork."""
    env = build_environment(
        inp.pop,
        inp.lines,
        inp.descriptions,
        inp.trading,
        cfg,
        cfg["random_seed"],
        population,
        supply_seed=supply_seed,
        scenario=scenario,
    )
    prior = make_prior(inp, env, cfg)
    fork = (cfg["simulation"]["fork_week"] - 1) * 7
    base = Engine(env, cfg).run(LegacyPlanner(cfg, name=LEGACY_REF), until=fork)
    return env, prior, base


def run_variant(base: Engine, name: str, cfg: dict, prior: SeasonalPrior):
    eng = base.clone()
    pol = make_policy(name, cfg, prior)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eng.run(pol)
    return eng, pol


def world_frames(env: Environment, eng: Engine) -> dict[str, pd.DataFrame]:
    """Long tables of one world for the database (observable and hidden kept apart)."""
    T, n, L = eng.sales.shape
    idx = pd.MultiIndex.from_product([range(T), range(n), range(L)], names=["day", "sku_idx", "loc"])
    obs = pd.DataFrame(
        {
            "sales": eng.sales.ravel(),
            "opening_available": eng.opening_available.ravel(),
            "closing_available": eng.closing_available.ravel(),
            "on_hand": eng.closing_on_hand.ravel(),
            "receipts": eng.receipts.ravel(),
            "transfer_in": eng.transfer_in.ravel(),
            "transfer_out": eng.transfer_out.ravel(),
            "cross_ship": eng.cross_ship.ravel(),
        },
        index=idx,
    ).reset_index()
    hidden = pd.DataFrame(
        {"baseline": env.baseline.ravel(), "lost": eng.lost.ravel(), "pre_uplift": env.pre_uplift.ravel()}, index=idx
    ).reset_index()
    keep = (obs[["sales", "receipts", "transfer_in", "transfer_out"]].abs().sum(axis=1) > 0) | obs["on_hand"].notna()
    return {"observed": obs[keep], "hidden": hidden[(hidden["baseline"] > 0) | (hidden["lost"] > 0)]}
