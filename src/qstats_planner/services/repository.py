"""Read and write access to the planner database. Used by both the API and the dashboard, so the
two always show the same numbers."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import insert, text, update

from qstats_planner.domain import tables


class Repository:
    def __init__(self, url: str):
        self.engine = tables.engine_for(url)

    def _q(self, sql: str, **params) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    # ------------------------------------------------------------------ master data
    def products(self) -> pd.DataFrame:
        return self._q("SELECT * FROM products ORDER BY sku_idx")

    def product(self, sku: str) -> dict | None:
        df = self._q("SELECT * FROM products WHERE sku = :sku", sku=sku)
        return None if df.empty else df.iloc[0].to_dict()

    def suppliers(self) -> pd.DataFrame:
        return self._q("SELECT * FROM suppliers ORDER BY supplier_id")

    def purchase_orders(self, sku: str | None = None, open_only: bool = False) -> pd.DataFrame:
        sql = "SELECT * FROM purchase_orders WHERE 1=1"
        if sku:
            sql += " AND sku = :sku"
        if open_only:
            sql += " AND status IN ('OPEN', 'IN_TRANSIT', 'DELAYED')"
        return self._q(sql + " ORDER BY order_date", sku=sku)

    def lead_time_distribution(self) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_lead_time_distribution")

    # ------------------------------------------------------------------ plan
    def kpis(self) -> dict:
        df = self._q("SELECT * FROM plan_kpis")
        return {r.key: json.loads(r.value) for r in df.itertuples()}

    def sku_plan(self) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_sku ORDER BY sku_idx")

    def recommendations(self, **filters) -> pd.DataFrame:
        sql = "SELECT * FROM plan_recommendations WHERE 1=1"
        params = {}
        for col in ("sku", "action", "severity", "supplier_id", "category", "location", "segment", "status"):
            if filters.get(col):
                sql += f" AND {col} = :{col}"
                params[col] = filters[col]
        return self._q(sql + " ORDER BY priority", **params)

    def recommendation(self, rec_id: str) -> dict | None:
        df = self._q("SELECT * FROM plan_recommendations WHERE recommendation_id = :r", r=rec_id)
        return None if df.empty else df.iloc[0].to_dict()

    def forecast(self, sku: str) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_forecast WHERE sku = :sku ORDER BY week", sku=sku)

    def history(self, sku: str) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_history WHERE sku = :sku ORDER BY week", sku=sku)

    def history_all(self) -> pd.DataFrame:
        return self._q("SELECT sku, week, observed, reconstructed, imputed FROM plan_history")

    def projection(self, sku: str) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_projection WHERE sku = :sku ORDER BY date", sku=sku)

    def projection_all(self) -> pd.DataFrame:
        return self._q("SELECT sku, date, projected, projected_with_order, safety_stock FROM plan_projection")

    def daily(self, sku: str) -> pd.DataFrame:
        return self._q("SELECT * FROM demand_observations WHERE sku = :sku ORDER BY date", sku=sku)

    def inventory(self, sku: str) -> pd.DataFrame:
        return self._q("SELECT * FROM inventory_snapshots WHERE sku = :sku ORDER BY date", sku=sku)

    def fba(self) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_fba ORDER BY sku_idx")

    def containers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._q("SELECT * FROM plan_containers"), self._q("SELECT * FROM plan_container_summary")

    def forecast_performance(self, sku: str | None = None) -> pd.DataFrame:
        if sku:
            return self._q("SELECT * FROM plan_forecast_performance WHERE sku = :sku", sku=sku)
        return self._q("SELECT * FROM plan_forecast_performance")

    def segment_scores(self) -> pd.DataFrame:
        return self._q("SELECT * FROM plan_segment_scores")

    # ------------------------------------------------------------------ evaluation layer
    def eval_table(self, name: str) -> pd.DataFrame:
        allowed = {
            "summary",
            "per_sku",
            "bootstrap",
            "calibration",
            "benchmark_scores",
            "benchmark_episodes",
            "benchmark_weekly",
            "world_weekly",
            "scenario_null",
            "scenario_optimistic_quotes",
        }
        if name not in allowed:
            raise ValueError(name)
        return self._q(f"SELECT * FROM eval_{name}")

    def eval_meta(self) -> dict:
        df = self._q("SELECT * FROM eval_meta")
        return {r.key: json.loads(r.value) for r in df.itertuples()}

    # ------------------------------------------------------------------ planner decisions
    def record_decision(
        self, rec_id: str, planner_action: str, quantity: int | None, comment: str, planner: str = "planner"
    ) -> dict:
        rec = self.recommendation(rec_id)
        if rec is None:
            raise KeyError(rec_id)
        if planner_action not in ("ACCEPT", "OVERRIDE", "REJECT"):
            raise ValueError(planner_action)
        if planner_action == "OVERRIDE" and (quantity is None or quantity < 0):
            raise ValueError("an override needs a non-negative quantity")
        row = {
            "recommendation_id": rec_id,
            "sku": rec["sku"],
            "action": rec["action"],
            "system_quantity": int(rec["recommended_quantity"]),
            "planner_action": planner_action,
            "override_quantity": int(quantity)
            if planner_action == "OVERRIDE"
            else (int(rec["recommended_quantity"]) if planner_action == "ACCEPT" else 0),
            "comment": comment or "",
            "planner": planner,
            "timestamp": datetime.now().replace(microsecond=0),
        }
        status = {"ACCEPT": "ACCEPTED", "OVERRIDE": "OVERRIDDEN", "REJECT": "REJECTED"}[planner_action]
        with self.engine.begin() as conn:
            conn.execute(insert(tables.overrides).values(**row))
            conn.execute(
                update(tables.recommendations).where(tables.recommendations.c.recommendation_id == rec_id).values(status=status)
            )
        return row

    def decisions(self) -> pd.DataFrame:
        return self._q("SELECT * FROM recommendation_overrides ORDER BY timestamp DESC")
