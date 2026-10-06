"""Scenario simulator: rerun the planning cycle on today's state with different assumptions.

The champions and error tables fitted in the base plan are reused (no refit), so a scenario takes
about a second. The knobs are those of `PlanSettings` plus container capacity.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qstats_planner.inventory.lead_time_demand import service_level
from qstats_planner.inventory.projection import project, scheduled_receipts, stockout_day
from qstats_planner.optimization.containers import plan_containers
from qstats_planner.replenishment.policy import PlanSettings, run_cycle
from qstats_planner.simulation import runner
from qstats_planner.utils.config import resolve


@dataclass
class ScenarioResult:
    summary: dict
    by_sku: pd.DataFrame
    containers: pd.DataFrame


class ScenarioService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        inp = runner.load_inputs(cfg)
        self.env, self.prior, _ = runner.build_world(inp, cfg, "demo", cfg["random_seed"])
        sim = resolve(cfg["paths"]["simulation_dir"]) / "demo"
        with open(sim / f"world_{runner.LEGACY_REF}_seed{cfg['random_seed']}.pkl", "rb") as fh:
            self.world = pickle.load(fh)["engine"]
        self.world.env = self.env
        with open(sim / "plan_state.pkl", "rb") as fh:
            self.fs = pickle.load(fh)["forecast_state"]
        self.view = self.world.view(self.env.n_days - 1)

    def run(
        self,
        service_level_target: float | None = None,
        demand_growth_pct: float = 0.0,
        lead_time_multiplier: float = 1.0,
        supplier_variability: float = 1.0,
        container_capacity_m3: float | None = None,
        probabilistic: bool = True,
        safety_days: float = 30.0,
    ) -> ScenarioResult:
        settings = PlanSettings(
            service_level=service_level_target,
            demand_multiplier=1 + demand_growth_pct / 100,
            lead_time_multiplier=lead_time_multiplier,
            lead_time_spread=supplier_variability,
            probabilistic=probabilistic,
            safety_days=safety_days,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            st = run_cycle(self.view, self.cfg, self.prior, settings, previous=self.fs, refit=False)
        v = self.view
        prod = v.products
        n = v.n_sku
        cost = prod["unit_cost"].to_numpy(float)
        horizon = self.cfg["forecasting"]["planning_horizon_days"]
        p = v.position
        stock_now = p.on_hand.sum(axis=1) + p.fba_inbound
        rec = scheduled_receipts(v.open_purchase_orders(), v.t, n, horizon)
        lt50 = np.array([d.quantile(0.5) for d in st.sku_lead_time])
        rec_new = rec.copy()
        for i in np.where(st.order_qty > 0)[0]:
            rec_new[int(min(max(lt50[i], 1), horizon) - 1), i] += st.order_qty[i]
        so_before = stockout_day(project(stock_now, st.daily_fc, rec, horizon))
        path_after = project(stock_now, st.daily_fc, rec_new, horizon)
        so_after = stockout_day(path_after)
        avg_inv_13w = float((np.maximum(path_after[:91], 0).mean(axis=0) * cost).sum())
        risk = self.cfg["inventory"]["stockout_risk_weeks"] * 7
        weekly = st.daily_fc[:91].sum(axis=0) / 13
        svc = np.array([service_level(st.ltd.samples[i], st.position[i] + st.order_qty[i]) for i in range(n)])
        w = np.where(v.launched, weekly, 0)
        purchase_value = float((st.order_qty * cost).sum())
        inventory_value = float((p.on_hand.sum(axis=1) * cost).sum())
        safety_value = float((np.maximum(st.target - st.ltd.mean, 0) * cost).sum())
        cap = container_capacity_m3 or float(v.suppliers["container_capacity_m3"].iloc[0])
        sups = v.suppliers.assign(container_capacity_m3=cap)
        buy = pd.DataFrame({"sku_idx": np.where(st.order_qty > 0)[0], "recommended_quantity": st.order_qty[st.order_qty > 0]})
        cover_after = (stock_now + st.order_qty) / np.maximum(weekly, 1e-9)
        margin_ok = prod["contribution_margin_pct"].to_numpy() >= self.cfg["business"]["minimum_margin_pct"]
        _, csum = plan_containers(buy, prod, sups, cover_after, weekly, margin_ok, v.discontinued(), cfg=self.cfg)
        summary = {
            "purchase_lines": int((st.order_qty > 0).sum()),
            "purchase_units": int(st.order_qty.sum()),
            "purchase_value": purchase_value,
            "skus_at_risk_before_orders": int((v.launched & (so_before >= 0) & (so_before < risk)).sum()),
            "skus_at_risk_after_orders": int((v.launched & (so_after >= 0) & (so_after < risk)).sum()),
            "projected_service_level": float((svc * w).sum() / max(w.sum(), 1e-9)),
            "inventory_value_now": inventory_value,
            "projected_average_inventory_13w": avg_inv_13w,
            "safety_stock_value": safety_value,
            "working_capital_committed": inventory_value + purchase_value,
            "containers": int(csum["containers"].sum()) if len(csum) else 0,
            "mean_container_utilisation": float(csum["utilisation"].mean()) if len(csum) else float("nan"),
        }
        by_sku = pd.DataFrame(
            {
                "sku": prod["sku"],
                "order_qty": st.order_qty,
                "order_up_to": st.target,
                "service_after": svc,
                "purchase_value": st.order_qty * cost,
            }
        )
        return ScenarioResult(summary, by_sku, csum)
