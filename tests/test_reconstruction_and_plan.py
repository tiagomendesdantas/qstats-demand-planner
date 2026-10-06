"""Stockout detection, demand reconstruction, and the full planning cycle on a synthetic world."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qstats_planner.demand import reconstruction as R
from qstats_planner.evaluation.benchmark import score
from qstats_planner.replenishment import recommendations
from qstats_planner.replenishment.policy import PlanSettings, run_cycle
from qstats_planner.simulation.engine import Engine
from qstats_planner.simulation.policies.legacy import LegacyPlanner

SRC = Path(__file__).resolve().parents[1] / "src" / "qstats_planner"


def channel(T=140, rate=10.0, seed=0, out=(60, 75)):
    rng = np.random.default_rng(seed)
    demand = rng.poisson(rate, (T, 1, 2)).astype(float)
    opening = np.full((T, 1, 2), 1e6)
    opening[out[0] : out[1], 0, 0] = 0.0
    sales = np.minimum(demand, opening)
    closing = opening - sales
    act = np.ones((T, 1, 2), bool)
    cd = R.ChannelData(sales, opening, closing, act, np.zeros(T, bool), np.zeros((T, 1), np.int8), np.ones((T, 1)))
    return cd, demand


def test_status_classification():
    cd, _ = channel()
    cd.opening[30, 0, 0], cd.closing[30, 0, 0], cd.sales[30, 0, 0] = 5, 0, 5  # sold out during the day
    cd.opening[40, 0, 0] = np.nan  # snapshot missing
    st, cen = R.classify(cd, 0.02)
    assert st[65, 0, 0] == R.CONFIRMED and st[30, 0, 0] == R.LIKELY and st[40, 0, 0] == R.UNKNOWN
    assert st[10, 0, 0] == R.NORMAL and cen[65, 0, 0] and not cen[10, 0, 0]


def test_zero_run_in_unknown_block_is_inferred_as_stockout():
    cd, _ = channel(out=(0, 0))
    cd.opening[50:60, 0, 0] = np.nan
    cd.sales[50:60, 0, 0] = 0  # ten zero days at a rate of 10/day
    _, cen = R.classify(cd, 0.02)
    assert cen[50:60, 0, 0].all()


@pytest.mark.parametrize("method", ["pre_post_velocity", "local_profile", "model_expectation", "censored_gamma"])
def test_reconstruction_recovers_lost_demand(method):
    cd, demand = channel()
    rec = R.reconstruct(cd, method, np.arange(140) % 7, two_sided=True)
    m = score(cd.sales, rec.adjusted, demand, rec.imputed, cd.active)
    assert rec.imputed[60:75, 0, 0].all() and not rec.imputed[:60].any()
    assert 0.8 < m["recovery_pct"] < 1.2  # a stationary series: close to full recovery
    assert np.array_equal(rec.adjusted[~rec.imputed], cd.sales[~rec.imputed])  # observed days untouched


def test_no_adjustment_is_the_floor():
    cd, demand = channel()
    rec = R.reconstruct(cd, "no_adjustment", np.arange(140) % 7)
    assert np.array_equal(rec.adjusted, cd.sales)


def test_gamma_tail_mean():
    mu, k = np.array([10.0]), np.array([2.0])
    assert R.gamma_tail_mean(mu, k, np.array([0.0]))[0] == pytest.approx(10.0, rel=1e-6)
    assert R.gamma_tail_mean(mu, k, np.array([15.0]))[0] > 15.0


def test_full_planning_cycle_and_recommendations(env, cfg, neutral_prior):
    eng = Engine(env, cfg).run(LegacyPlanner(cfg), until=150)
    view = eng.view()
    st = run_cycle(view, cfg, neutral_prior, PlanSettings(), keep_backtest=True)
    prod = view.products
    q = st.order_qty
    assert ((q == 0) | ((q % prod["case_pack"].to_numpy() == 0) & (q >= prod["moq"].to_numpy()))).all()
    assert (st.target >= st.ltd.quantiles[0.5] - 1e-9).all()  # order-up-to >= median demand
    need = st.position < st.target
    assert (q[~need] == 0).all()
    plan = recommendations.build(view, st, cfg, pd.Timestamp("2026-01-01"))
    recs = plan["recommendations"]
    required = [
        "recommendation_id",
        "sku",
        "location",
        "action",
        "severity",
        "recommended_quantity",
        "stockout_date",
        "confidence",
        "reason",
        "economic_impact",
        "created_at",
        "evidence",
        "expected_effect",
    ]
    assert set(required) <= set(recs.columns) and recs["recommendation_id"].is_unique
    assert recs["reason"].str.len().gt(10).all()
    assert set(recs["action"]) <= {
        "CRITICAL_STOCKOUT",
        "BUY",
        "EXPEDITE",
        "TRANSFER",
        "SEND_TO_FBA",
        "EXCESS",
        "LOW_MARGIN",
        "REVIEW_FORECAST",
        "STOCKOUT_CENSORED",
        "NO_ACTION",
    }
    assert recs["confidence"].isin(["HIGH", "MEDIUM", "LOW"]).all()


def test_low_margin_routes_a_buy_to_review(env, cfg, neutral_prior):
    env.products.loc[:, "contribution_margin_pct"] = 0.05
    eng = Engine(env, cfg).run(LegacyPlanner(cfg), until=150)
    view = eng.view()
    st = run_cycle(view, cfg, neutral_prior, PlanSettings(), keep_backtest=True)
    recs = recommendations.build(view, st, cfg, pd.Timestamp("2026-01-01"))["recommendations"]
    assert "BUY" not in set(recs["action"])
    if (st.order_qty > 0).any():
        assert "LOW_MARGIN" in set(recs["action"])


def test_fba_plan_respects_source_stock(env, cfg, neutral_prior):
    eng = Engine(env, cfg).run(LegacyPlanner(cfg), until=150)
    view = eng.view()
    st = run_cycle(view, cfg, neutral_prior, PlanSettings())
    for move in st.fba.attrs["transfers"]:
        assert move["qty"] <= view.position.available[move["sku_idx"], move["source"]]
        assert view.products.at[move["sku_idx"], "fba_enabled"]


PLANNER_PACKAGES = ["demand", "forecasting", "inventory", "replenishment", "optimization", "economics", "domain"]


def test_planner_code_depends_on_neither_the_simulation_nor_the_evaluation_layer():
    """The planner must run on a client's data unchanged, and must not be able to read the hidden truth."""
    for pkg in PLANNER_PACKAGES:
        for f in (SRC / pkg).glob("*.py"):
            tree = ast.parse(f.read_text())
            names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
            names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            assert not any(m.startswith("qstats_planner.evaluation") for m in names), f
            assert not any(m.startswith("qstats_planner.simulation") for m in names), f


def test_replayed_planners_see_the_world_only_through_the_planner_view():
    """The Legacy and QStats planners replayed in the comparison live in simulation/policies; they may
    take the Decisions container from the engine, nothing else from the simulation or evaluation."""
    for f in (SRC / "simulation" / "policies").glob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.ImportFrom) and n.module:
                assert not n.module.startswith("qstats_planner.evaluation"), f
                assert n.module != "qstats_planner.simulation.environment", f
                if n.module == "qstats_planner.simulation.engine":
                    assert [a.name for a in n.names] == ["Decisions"], f
            if isinstance(n, ast.Import):
                assert not any(a.name.startswith(("qstats_planner.evaluation", "qstats_planner.simulation")) for a in n.names), f


UCI_COLUMNS = ("Invoice", "StockCode", "InvoiceDate", "Customer ID")


def test_only_the_uci_adapter_knows_uci_column_names():
    for f in SRC.rglob("*.py"):
        if f.relative_to(SRC).as_posix() == "adapters/uci.py":
            continue
        text = f.read_text()
        for col in UCI_COLUMNS:
            assert f'"{col}"' not in text and f"'{col}'" not in text, (f, col)
