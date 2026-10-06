"""Canonical data contract.

This is the schema a client's systems map into. The planner reads only these tables and these
column names; it never sees a source system's column names. An adapter (UCI, CSV export,
PostgreSQL, Azure SQL) is responsible for producing frames that pass `validate_frame`.

The Pydantic models document each record and are reused by the API. The `TABLES` spec is what
`validate_frame` enforces on the pandas frames that move through the pipeline.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field


class LineType(StrEnum):
    SALE = "SALE"
    RETURN = "RETURN"
    ADJUSTMENT = "ADJUSTMENT"
    NON_MERCHANDISE = "NON_MERCHANDISE"


class AvailabilityStatus(StrEnum):
    NORMAL = "NORMAL"
    CONFIRMED_STOCKOUT = "CONFIRMED_STOCKOUT"
    LIKELY_CONSTRAINED = "LIKELY_CONSTRAINED"
    UNKNOWN_AVAILABILITY = "UNKNOWN_AVAILABILITY"
    PROMOTION = "PROMOTION"
    LIQUIDATION = "LIQUIDATION"


class POStatus(StrEnum):
    OPEN = "OPEN"
    IN_TRANSIT = "IN_TRANSIT"
    RECEIVED = "RECEIVED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"


class FbaState(StrEnum):
    AVAILABLE = "AVAILABLE"
    INBOUND = "INBOUND"
    RESERVED = "RESERVED"
    TRANSFER = "TRANSFER"


# --------------------------------------------------------------------------- records


class SalesLine(BaseModel):
    """One order line as the source recorded it (the input to demand building)."""

    order_id: str
    order_ts: datetime
    sku: str
    description: str | None = None
    quantity: float
    unit_price: float
    customer_id: str | None = None
    country: str | None = None
    line_type: LineType


class DemandObservation(BaseModel):
    date: date
    sku: str
    location: str
    units: float = Field(description="Units sold (fulfilled). Censored when stock ran out.")
    price: float | None = None
    promotion_flag: bool = False
    availability_status: AvailabilityStatus = AvailabilityStatus.NORMAL


class InventorySnapshot(BaseModel):
    timestamp: date
    sku: str
    location: str
    on_hand: float
    available: float
    reserved: float = 0.0
    inbound: float = 0.0
    transfer: float = 0.0


class PurchaseOrder(BaseModel):
    po_id: str
    sku: str
    supplier_id: str
    location: str
    quantity: float
    order_date: date
    expected_arrival: date
    actual_arrival: date | None = None
    status: POStatus


class Product(BaseModel):
    sku: str
    description: str
    category: str
    supplier_id: str
    unit_cost: float
    selling_price: float
    case_pack: int
    moq: int
    cube_per_case: float
    target_service_level: float
    fba_enabled: bool
    preferred_source_dc: str


class Supplier(BaseModel):
    supplier_id: str
    supplier_name: str
    country: str
    quoted_lead_time_days: float
    minimum_order_value: float
    minimum_container_fill: float


class LeadTimeObservation(BaseModel):
    po_id: str
    supplier_id: str
    order_date: date
    receipt_date: date | None
    lead_time_days: float | None


class Location(BaseModel):
    location_id: str
    kind: str


class Promotion(BaseModel):
    sku: str
    start_date: date
    end_date: date
    kind: AvailabilityStatus
    announced_date: date


# --------------------------------------------------------------------------- frame specs

TABLES: dict[str, dict[str, str]] = {
    "sales_lines": {
        "order_id": "string", "order_ts": "datetime", "sku": "string", "description": "string",
        "quantity": "float", "unit_price": "float", "customer_id": "string", "country": "string",
        "line_type": "string",
    },
    "demand_observations": {
        "date": "datetime", "sku": "string", "location": "string", "units": "float",
        "availability_status": "string", "promotion_flag": "bool",
    },
    "inventory_snapshots": {
        "date": "datetime", "sku": "string", "location": "string", "on_hand": "float",
        "available": "float", "reserved": "float", "inbound": "float", "transfer": "float",
    },
    "purchase_orders": {
        "po_id": "string", "sku": "string", "supplier_id": "string", "location": "string",
        "quantity": "float", "order_date": "datetime", "expected_arrival": "datetime",
        "status": "string",
    },
    "products": {
        "sku": "string", "description": "string", "category": "string", "supplier_id": "string",
        "unit_cost": "float", "selling_price": "float", "case_pack": "int", "moq": "int",
        "cube_per_case": "float", "target_service_level": "float", "fba_enabled": "bool",
        "preferred_source_dc": "string",
    },
    "suppliers": {
        "supplier_id": "string", "supplier_name": "string", "country": "string",
        "quoted_lead_time_days": "float", "minimum_order_value": "float",
    },
    "locations": {"location_id": "string", "kind": "string"},
}

_KIND_CHECK = {
    "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "float": pd.api.types.is_numeric_dtype,
    "int": pd.api.types.is_integer_dtype,
    "bool": pd.api.types.is_bool_dtype,
}


class ContractError(ValueError):
    pass


def validate_frame(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Raise ContractError when a frame does not carry the canonical columns and kinds."""
    spec = TABLES[table]
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ContractError(f"{table}: missing columns {missing}")
    wrong = [c for c, kind in spec.items() if not _KIND_CHECK[kind](df[c])]
    if wrong:
        raise ContractError(f"{table}: wrong kind for {[(c, str(df[c].dtype)) for c in wrong]}")
    return df
