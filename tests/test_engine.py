"""Inventory simulation: conservation, censoring, forks and leakage."""

import numpy as np
import pytest

from qstats_planner.simulation.engine import Engine
from qstats_planner.simulation.environment import FBA
from qstats_planner.simulation.policies.base import network_position
from qstats_planner.simulation.policies.legacy import LegacyPlanner
from tests.conftest import make_env


def run(env, cfg, until=None):
    return Engine(env, cfg).run(LegacyPlanner(cfg), until=until)


def test_inventory_is_conserved(env, cfg):
    e = run(env, cfg)
    start = e.warm_end
    oh = np.nan_to_num(e.closing_on_hand)
    for t in range(start + 1, e.n_days):
        flow = (
            oh[t - 1]
            + e.receipts[t]
            + e.transfer_in[t]
            - e.transfer_out[t]
            - e.write_off[t]
            - e.sales[t]
            + _cross_in(e, t)
            - _cross_out(e, t)
        )
        assert np.allclose(oh[t], flow), t


def _cross_out(e, t):
    """Units a DC shipped to the other region (they leave this DC's stock)."""
    return e.cross_ship[t]


def _cross_in(e, t):
    """Sales are recorded at the customer's region; units shipped by the other DC are added back
    to the region that recorded the sale."""
    out = np.zeros_like(e.cross_ship[t])
    out[:, 0] = e.cross_ship[t][:, 1]
    out[:, 1] = e.cross_ship[t][:, 0]
    return out


def test_censoring_identity(env, cfg):
    env.baseline[80:] *= 3  # a demand jump the smoothing rule lags behind
    e = run(env, cfg)
    assert np.allclose(env.baseline, e.sales + e.lost)
    assert (e.lost >= -1e-9).all() and (e.closing_on_hand[e.warm_end :] >= -1e-9).all()
    assert e.lost[: e.warm_end].sum() == 0  # warm-up: ample stock
    assert e.lost[e.warm_end :].sum() > 0  # the system does create stockouts


def test_sales_never_exceed_what_was_sellable(env, cfg):
    e = run(env, cfg)
    s, oa = e.sales[e.warm_end :], e.opening_available[e.warm_end :]
    fba = s[:, :, FBA]
    assert (fba <= oa[:, :, FBA] + 1e-9).all()
    direct = s[:, :, 0] + s[:, :, 1]
    assert (direct <= oa[:, :, 0] + oa[:, :, 1] + 1e-9).all()


def test_fork_is_identical_to_running_straight_through(env, cfg):
    a = run(env, cfg)
    b = run(env, cfg, until=100)
    c = b.clone().run(LegacyPlanner(cfg))
    assert np.array_equal(a.sales, c.sales) and np.array_equal(np.nan_to_num(a.closing_on_hand), np.nan_to_num(c.closing_on_hand))
    assert [p["po_id"] for p in a.pos] == [p["po_id"] for p in c.pos]


def test_future_demand_does_not_change_past_decisions(cfg):
    """Scramble the hidden baseline after day t: every decision up to t must be unchanged."""
    t = 120
    e1 = make_env(cfg, seed=3)
    e2 = make_env(cfg, seed=3)
    rng = np.random.default_rng(99)
    e2.baseline[t + 1 :] = rng.permutation(e2.baseline[t + 1 :].ravel()).reshape(e2.baseline[t + 1 :].shape) * 3
    a, b = run(e1, cfg), run(e2, cfg)
    pa = [(p["sku_idx"], p["qty"], p["order_day"]) for p in a.pos if p["order_day"] <= t]
    pb = [(p["sku_idx"], p["qty"], p["order_day"]) for p in b.pos if p["order_day"] <= t]
    assert pa == pb
    assert np.array_equal(a.sales[: t + 1], b.sales[: t + 1])


def test_planner_view_hides_the_evaluation_layer(env, cfg):
    e = run(env, cfg, until=80)
    v = e.view()
    for name in ("baseline", "lost", "env", "pre_uplift"):
        assert not hasattr(v, name)
    po = v.purchase_orders()
    assert po.loc[po["status"] != "RECEIVED", "actual_day"].isna().all()  # arrival unknown until it happens
    with pytest.raises(ValueError):
        v.sales[0, 0, 0] = 1.0  # read-only


def test_inventory_position_counts_open_orders(env, cfg):
    e = run(env, cfg, until=90)
    v = e.view()
    ip = network_position(v, reserved_weight=1.0)
    p = v.position
    open_q = v.open_purchase_orders().groupby("sku_idx")["quantity"].sum().reindex(range(v.n_sku), fill_value=0).to_numpy()
    assert np.allclose(ip, p.on_hand.sum(axis=1) + p.fba_inbound + open_q)
    assert (network_position(v, reserved_weight=0.5) <= ip + 1e-9).all()


def test_fba_transfer_flow(env, cfg):
    e = run(env, cfg)
    assert len(e.transfers) > 0
    tr = e.transfers[0]
    i, q = tr["sku_idx"], tr["qty"]
    assert e.transfer_out[tr["ship_day"], i, tr["source"]] >= q
    assert e.transfer_in[tr["arrive_day"], i, FBA] >= q
    assert tr["arrive_day"] > tr["ship_day"] >= tr["decided_day"]
    assert e.env.products.at[i, "fba_enabled"]
