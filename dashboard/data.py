"""Cached access to the planner database for the dashboard (the same Repository the API uses)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from qstats_planner.services.repository import Repository  # noqa: E402
from qstats_planner.utils.config import database_url, load_config  # noqa: E402


@st.cache_resource
def cfg() -> dict:
    return load_config()


@st.cache_resource
def repo() -> Repository:
    return Repository(database_url(cfg()))


@st.cache_data(ttl=600)
def table(name: str, *args) -> pd.DataFrame:
    return getattr(repo(), name)(*args)


@st.cache_data(ttl=600)
def kpis() -> dict:
    return repo().kpis()


@st.cache_data(ttl=600)
def eval_table(name: str) -> pd.DataFrame:
    return repo().eval_table(name)


@st.cache_data(ttl=600)
def eval_meta() -> dict:
    return repo().eval_meta()


def recommendations() -> pd.DataFrame:
    """Not cached: planner decisions change the status column."""
    return repo().recommendations()


@st.cache_resource(show_spinner="Loading today's state for scenarios…")
def scenario_service():
    from qstats_planner.services.scenario import ScenarioService

    return ScenarioService(cfg())


def plan_date() -> str:
    return pd.Timestamp(kpis()["plan_date"]).strftime("%a %d %b %Y")


@st.cache_data(ttl=600)
def verdicts() -> dict:
    """The comparison's verdicts in words, computed once for every page that states them."""
    import numpy as np

    from qstats_planner.evaluation.comparison import efficiency_vs_legacy
    from qstats_planner.simulation.runner import LEGACY_FAMILY

    seed = cfg()["random_seed"]
    meta = eval_meta().get("matched", {})
    m = meta[str(seed)]
    b0 = eval_table("bootstrap").query("seed == @seed")
    lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
    slo, shi = b0["legacy_extra_inventory_pct"].quantile([0.05, 0.95])
    sav, extra = m["inventory_saving_pct"], m["legacy_extra_inventory_pct"]
    if not np.isfinite(sav):
        v1 = "At today's service level, no reading: QStats's frontier does not reach the current fill rate."
    elif lo > 0:
        v1 = "At today's service level, QStats needs less inventory."
    elif hi < 0:
        v1 = "At today's service level, QStats needs more inventory."
    else:
        v1 = "At today's service level, no difference distinguishable from zero."
    if not np.isfinite(extra):
        v2 = "At QStats's service level, the legacy frontier runs out."
    elif slo > 0:
        v2 = "At QStats's service level, the current rule needs more inventory."
    elif shi < 0:
        v2 = "At QStats's service level, the current rule needs less inventory."
    else:
        who = "QStats" if extra > 0 else "the current rule"
        v2 = f"At QStats's service level the direction favours {who}, but not conclusively."
    head = eval_table("summary").query("subset == 'headline'")
    ref = head[head["seed"] == seed].set_index("variant")
    lf = ref.loc[[v for v in LEGACY_FAMILY if v in ref.index]]
    top = lf.loc[lf["fill_rate"].idxmax()]
    eff = {
        s_: efficiency_vs_legacy({v: dict(r) for v, r in g.set_index("variant").iterrows()}, LEGACY_FAMILY)
        for s_, g in head.groupby("seed")
    }
    # worlds where both the prior-only arm and full QStats can be read off the Legacy frontier
    both = [e for e in eff.values() if np.isfinite(e.get("legacy_30d_prior", np.nan)) and np.isfinite(e.get("qstats", np.nan))]
    prior_wins = sum(int(e["legacy_30d_prior"] >= e["qstats"]) for e in both)
    return {
        "seed": seed,
        "matched": meta,
        "m": m,
        "interval": (lo, hi),
        "interval_secondary": (slo, shi),
        "primary": v1,
        "secondary": v2,
        "worlds": len(meta),
        "secondary_positive_worlds": sum(int(v["legacy_extra_inventory_pct"] > 0) for v in meta.values()),
        "prior_wins": prior_wins,
        "prior_comparable": len(both),
        # the legacy frontier's highest point, for when QStats runs beyond it
        "legacy_top": {
            "days": int(top.name.removeprefix("legacy_").removesuffix("d")),
            "fill": float(top["fill_rate"]),
            "inventory": float(top["average_inventory_value"]),
        },
        "qstats_inventory": float(ref.at["qstats", "average_inventory_value"]),
        "secondary_other_worlds": [
            v["legacy_extra_inventory_pct"]
            for s_, v in meta.items()
            if s_ != str(seed) and np.isfinite(v["legacy_extra_inventory_pct"])
        ],
    }
