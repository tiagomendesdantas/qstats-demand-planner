"""Database schema (SQLAlchemy Core, portable types only).

Nothing here is SQLite-specific: pointing `paths.database_url` at PostgreSQL or Azure SQL
(`postgresql+psycopg://...`, `mssql+pyodbc://...`) creates the same tables. Operational tables
mirror the canonical contract; `plan_*` tables hold the output of a planning cycle; `eval_*`
tables hold the simulation's evaluation layer (baseline demand, benchmark scores) and are read only
by the evaluation pages, never by the planner.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

metadata = MetaData()

products = Table(
    "products",
    metadata,
    Column("sku", String(32), primary_key=True),
    Column("sku_idx", Integer),
    Column("description", String(200)),
    Column("category", String(64)),
    Column("supplier_id", String(16)),
    Column("unit_cost", Float),
    Column("selling_price", Float),
    Column("contribution_margin", Float),
    Column("contribution_margin_pct", Float),
    Column("fulfillment_cost_dc", Float),
    Column("fulfillment_cost_fba", Float),
    Column("advertising_cost", Float),
    Column("case_pack", Integer),
    Column("moq", Integer),
    Column("cube_per_case", Float),
    Column("abc_class", String(1)),
    Column("target_service_level", Float),
    Column("fba_enabled", Boolean),
    Column("preferred_source_dc", String(16)),
    Column("profile", String(32)),
)

suppliers = Table(
    "suppliers",
    metadata,
    Column("supplier_id", String(16), primary_key=True),
    Column("supplier_name", String(64)),
    Column("country", String(64)),
    Column("quoted_lead_time_days", Float),
    Column("minimum_order_value", Float),
    Column("minimum_container_fill", Float),
    Column("container_capacity_m3", Float),
    Column("lead_time_mean", Float),
    Column("lead_time_std", Float),
    Column("lead_time_p50", Float),
    Column("lead_time_p90", Float),
    Column("receipts", Integer),
    Column("open_orders", Integer),
)

locations = Table("locations", metadata, Column("location_id", String(16), primary_key=True), Column("kind", String(8)))

purchase_orders = Table(
    "purchase_orders",
    metadata,
    Column("po_id", String(16), primary_key=True),
    Column("sku", String(32)),
    Column("supplier_id", String(16)),
    Column("quantity", Integer),
    Column("qty_east", Integer),
    Column("qty_west", Integer),
    Column("order_date", Date),
    Column("expected_arrival", Date),
    Column("actual_arrival", Date),
    Column("status", String(12)),
    Column("policy", String(32)),
)

lead_time_observations = Table(
    "lead_time_observations",
    metadata,
    Column("po_id", String(32)),
    Column("supplier_id", String(16)),
    Column("lead_time_days", Float),
    Column("received", Boolean),
)

demand_observations = Table(
    "demand_observations",
    metadata,
    Column("date", Date),
    Column("sku", String(32)),
    Column("channel", String(16)),
    Column("units", Float),
    Column("reconstructed", Float),
    Column("availability_status", String(24)),
)

inventory_snapshots = Table(
    "inventory_snapshots",
    metadata,
    Column("date", Date),
    Column("sku", String(32)),
    Column("location", String(16)),
    Column("on_hand", Float),
    Column("available", Float),
    Column("sales", Float),
    Column("receipts", Float),
)

recommendations = Table(
    "plan_recommendations",
    metadata,
    Column("recommendation_id", String(24), primary_key=True),
    Column("priority", Integer),
    Column("sku", String(32)),
    Column("sku_idx", Integer),
    Column("description", String(200)),
    Column("category", String(64)),
    Column("supplier_id", String(16)),
    Column("segment", String(16)),
    Column("location", String(16)),
    Column("action", String(24)),
    Column("severity", String(10)),
    Column("recommended_quantity", Integer),
    Column("stockout_date", Date),
    Column("inventory_position", Float),
    Column("confidence", String(8)),
    Column("confidence_score", Float),
    Column("reason", Text),
    Column("economic_impact", Float),
    Column("evidence", Text),
    Column("expected_effect", Text),
    Column("created_at", DateTime),
    Column("priority_score", Float),
    Column("status", String(12)),
)

overrides = Table(
    "recommendation_overrides",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("recommendation_id", String(24)),
    Column("sku", String(32)),
    Column("action", String(24)),
    Column("system_quantity", Integer),
    Column("planner_action", String(10)),
    Column("override_quantity", Integer),
    Column("comment", Text),
    Column("planner", String(64)),
    Column("timestamp", DateTime),
)


def engine_for(url: str):
    return create_engine(url, future=True)


def create_schema(engine) -> None:
    metadata.drop_all(engine)
    metadata.create_all(engine)
