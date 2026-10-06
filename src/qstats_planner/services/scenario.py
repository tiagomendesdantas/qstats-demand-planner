"""Scenario simulator: rerun the planning cycle on today's state with different assumptions.

The champions and error tables fitted in the base plan are reused (no refit), so a scenario takes
a few seconds. The knobs are those of `PlanSettings` plus container capacity. Recommendations, KPIs
and containers come from the same functions as the weekly plan, so the base run reproduces it.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qstats_planner.economics.impact import PURCHASE_ACTIONS, portfolio_kpis
from qstats_planner.optimization.containers import containers_for_plan
from qstats_planner.replenishment import recommendations
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
            # the same recommendation, KPI and container code as the weekly plan, so the base run
            # reproduces this week's plan exactly
            plan = recommendations.build(self.view, st, self.cfg, pd.Timestamp(0))
            k = portfolio_kpis(self.view, st, plan, self.cfg)
        v = self.view
        prod = v.products
        cost = prod["unit_cost"].to_numpy(float)
        sp = plan["sku_plan"]
        recs = plan["recommendations"]
        lines = recs[recs["action"].isin(PURCHASE_ACTIONS) & (recs["recommended_quantity"] > 0)].drop_duplicates("sku_idx")
        qty = np.zeros(v.n_sku, dtype=int)
        qty[lines["sku_idx"].to_numpy()] = lines["recommended_quantity"].to_numpy()
        sups = v.suppliers if container_capacity_m3 is None else v.suppliers.assign(container_capacity_m3=container_capacity_m3)
        _, csum = containers_for_plan(plan, prod, sups, self.cfg)
        risk = self.cfg["inventory"]["stockout_risk_weeks"] * 7
        so_after = sp["stockout_day_with_order"].to_numpy()
        summary = {
            "purchase_lines": k["purchase_lines"],
            "purchase_lines_for_review": k["purchase_lines_for_review"],
            "purchase_units": int(qty.sum()),
            "purchase_value": k["purchase_value"],
            "skus_at_risk_before_orders": k["skus_at_stockout_risk"],
            "skus_at_risk_after_orders": int((v.launched & (so_after >= 0) & (so_after < risk)).sum()),
            "projected_service_level": k["service_level_after_plan"],
            "inventory_value_now": k["inventory_value"],
            "projected_average_inventory_13w": float((np.maximum(plan["path_with_order"][:91], 0).mean(axis=0) * cost).sum()),
            "safety_stock_value": float((sp["safety_stock"].to_numpy() * cost).sum()),
            "working_capital_committed": k["inventory_value"] + k["purchase_value"],
            "containers": int(csum["containers"].sum()) if len(csum) else 0,
            "mean_container_utilisation": float(csum["utilisation"].mean()) if len(csum) else float("nan"),
        }
        by_sku = pd.DataFrame(
            {
                "sku": prod["sku"],
                "order_qty": qty,
                "order_up_to": st.target,
                "service_after": sp["service_after"].to_numpy(),
                "purchase_value": qty * cost,
            }
        )
        return ScenarioResult(summary, by_sku, csum)
