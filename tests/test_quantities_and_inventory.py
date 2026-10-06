import numpy as np
import pandas as pd
import pytest

from qstats_planner.forecasting.uncertainty import ErrorTable
from qstats_planner.inventory.lead_time import LeadTimeDistribution, estimate, kaplan_meier
from qstats_planner.inventory.lead_time_demand import expected_shortfall, lead_time_demand, service_level
from qstats_planner.inventory.projection import expected_lost, project, scheduled_receipts, stockout_day
from qstats_planner.optimization.containers import ContainerLine, GreedyContainerSolver
from qstats_planner.replenishment.order_quantity import round_order, round_orders, split_by_share


def test_spec_example_moq_and_case_pack():
    assert round_order(1047, case_pack=24, moq=600) == 1056


def test_moq_binds_and_rounds_up_to_case():
    assert round_order(100, case_pack=24, moq=600) == 600
    assert round_order(100, case_pack=25, moq=610) == 625
    assert round_order(0, case_pack=24, moq=600) == 0
    assert round_orders(np.array([0, 1, 49]), np.array([12, 12, 12]), np.array([24, 24, 24])).tolist() == [0, 24, 60]


def test_split_by_share_whole_cases():
    e, w = split_by_share(np.array([120]), np.array([0.6]), np.array([12]), np.array([True]))
    assert e[0] + w[0] == 120 and e[0] % 12 == 0 and w[0] % 12 == 0 and e[0] == 72


def test_kaplan_meier_uses_censored_open_orders():
    t = np.array([10.0, 12.0, 14.0, 30.0])
    full, _ = kaplan_meier(t, np.ones(4, bool))
    d, p = kaplan_meier(t, np.array([True, True, True, False]))
    assert np.isclose(p.sum(), 1) and d.max() == 30.0  # the open order keeps the tail
    lt = LeadTimeDistribution("X", d, p, 3, 1, 12)
    assert lt.quantile(0.9) == 30.0 and len(full) == 4


def test_lead_time_shrinks_to_quote_with_few_receipts():
    obs = pd.DataFrame({"supplier_id": ["S"] * 2, "lead_time_days": [80.0, 82.0], "event": [True, True]})
    few = estimate(obs, "S", quoted=50, k=8)
    many = estimate(pd.concat([obs] * 20), "S", quoted=50, k=8)
    assert few.quantile(0.5) < 70 < many.quantile(0.5)


def _table(errors):
    return ErrorTable({("REGULAR", b): np.asarray(errors, float) for b in range(6)}, {})


def test_lead_time_demand_deterministic_case():
    daily = np.full((200, 1), 10.0)
    lt = [LeadTimeDistribution("S", np.array([20.0]), np.array([1.0]), 10, 0, 20)]
    r = lead_time_demand(daily, lt, 7, np.array(["REGULAR"]), _table([0.0]), np.array([0.95]), np.array([70.0]))
    assert r.mean[0] == pytest.approx(270) and r.target[0] == pytest.approx(270)  # 27 days x 10


def test_lead_time_demand_mixture_and_safety_stock():
    daily = np.full((300, 1), 10.0)
    lt = [LeadTimeDistribution("S", np.array([20.0, 60.0]), np.array([0.8, 0.2]), 10, 0, 20)]
    r = lead_time_demand(daily, lt, 0, np.array(["REGULAR"]), _table([0.0]), np.array([0.9]), np.array([70.0]))
    assert r.mean[0] == pytest.approx(0.8 * 200 + 0.2 * 600)
    assert r.target[0] == pytest.approx(600)  # P90 sits in the long-lead-time branch
    ss = r.target[0] - r.mean[0]
    assert ss > 0
    assert service_level(r.samples[0], r.target[0]) >= 0.8
    assert expected_shortfall(r.samples[0], 1e9) == 0


def test_reorder_point_moves_with_demand():
    lt = [LeadTimeDistribution("S", np.array([30.0]), np.array([1.0]), 10, 0, 30)]
    low = lead_time_demand(
        np.full((200, 1), 5.0), lt, 7, np.array(["REGULAR"]), _table([-0.2, 0, 0.3]), np.array([0.95]), np.array([35.0])
    )
    high = lead_time_demand(
        np.full((200, 1), 15.0), lt, 7, np.array(["REGULAR"]), _table([-0.2, 0, 0.3]), np.array([0.95]), np.array([105.0])
    )
    assert high.target[0] > 2.5 * low.target[0]


def test_projection_and_stockout_date():
    open_po = pd.DataFrame({"sku_idx": [0], "quantity": [100], "expected_day": [14], "status": ["IN_TRANSIT"]})
    rec = scheduled_receipts(open_po, t=9, n=1, horizon=30)
    assert rec[4, 0] == 100
    path = project(np.array([30.0]), np.full((30, 1), 10.0), rec, 30)
    assert stockout_day(path)[0] == 2  # 30 units last three days
    lost = expected_lost(np.array([30.0]), np.full((30, 1), 10.0), rec, np.array([10]))
    assert lost[0] == pytest.approx(10.0)  # day 3 is lost; the receipt on day 4 covers the rest


def test_container_top_up_only_below_minimum_fill():
    lines = [ContainerLine("A", 0, 120, 12, 1.0, 2.0, 4.0, 30.0, True), ContainerLine("B", 1, 0, 12, 1.0, 2.0, 2.0, 30.0, True)]
    out = GreedyContainerSolver(top_up_to=0.65, max_cover_weeks=16).solve("S", lines, capacity_m3=68.0, min_fill=0.55)
    assert out["utilisation"].iloc[0] == pytest.approx(out["cube_m3"].sum() / 68.0)
    assert 0.55 <= out["utilisation"].iloc[0] <= 0.67 and out["top_up_units"].sum() > 0
    full = [ContainerLine("A", 0, 480, 12, 1.0, 2.0, 4.0, 30.0, True)]
    out2 = GreedyContainerSolver().solve("S", full, capacity_m3=68.0, min_fill=0.55)
    assert out2["top_up_units"].sum() == 0 and out2["utilisation"].iloc[0] == pytest.approx(40 / 68)
