"""What a planner is allowed to see.

`PlannerView` is the only object a planning policy receives. It exposes the business as its systems
would record it up to the plan date: observed sales (censored by stock), inventory snapshots (with
the gaps of a real feed), purchase orders as their status was known that day, received lead times,
announced events and master data. It holds no reference to the environment, so baseline demand,
lost sales, future arrival dates and true lead-time parameters are out of reach by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qstats_planner.utils.calendar import TradingCalendar

PUBLIC_PRODUCT_COLUMNS = [
    "sku",
    "description",
    "category",
    "supplier_id",
    "supplier_idx",
    "selling_price",
    "unit_cost",
    "case_pack",
    "moq",
    "cube_per_case",
    "abc_class",
    "target_service_level",
    "fba_enabled",
    "preferred_source_dc",
    "fulfillment_cost_dc",
    "fulfillment_cost_fba",
    "advertising_cost",
    "contribution_dc",
    "contribution_fba",
    "contribution_margin",
    "contribution_margin_pct",
]
PUBLIC_SUPPLIER_COLUMNS = [
    "supplier_id",
    "supplier_name",
    "country",
    "quoted_lead_time_days",
    "minimum_order_value",
    "minimum_container_fill",
    "container_capacity_m3",
]


def _ro(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.setflags(write=False)
    return v


@dataclass(frozen=True)
class Position:
    """Inventory position components at the plan date, per SKU and location."""

    on_hand: np.ndarray  # (n_sku, n_loc) physical units
    available: np.ndarray  # (n_sku, n_loc) sellable units
    fba_reserved: np.ndarray  # (n_sku,)
    fba_transfer: np.ndarray  # (n_sku,) units moving between Amazon fulfilment centres
    fba_inbound: np.ndarray  # (n_sku,) shipped from a DC, not yet received at Amazon
    dc_committed: np.ndarray  # (n_sku, 2) allocated to outbound FBA transfers, not yet shipped


class PlannerView:
    def __init__(
        self,
        *,
        t: int,
        days: pd.DatetimeIndex,
        calendar: TradingCalendar,
        trading: np.ndarray,
        products: pd.DataFrame,
        suppliers: pd.DataFrame,
        sales: np.ndarray,
        opening_available: np.ndarray,
        closing_available: np.ndarray,
        on_hand_hist: np.ndarray,
        snapshot_known: np.ndarray,
        position: Position,
        purchase_orders: list[dict],
        lead_time_history: pd.DataFrame,
        events: pd.DataFrame,
        transfers: list[dict],
        first_planning_day: int,
        launch_day_known: np.ndarray,
        review_period_days: int,
    ):
        self.t = t
        self.date = days[t]
        self.days = days[: t + 1]
        self.all_days = days  # dates only; the future calendar is public knowledge
        self.calendar = calendar
        self.trading = _ro(trading[: t + 1])
        self.products = products[PUBLIC_PRODUCT_COLUMNS].copy()
        self.suppliers = suppliers[PUBLIC_SUPPLIER_COLUMNS].copy()
        self.sales = _ro(sales[: t + 1])
        known = snapshot_known[: t + 1]
        self.opening_available = _ro(np.where(known, opening_available[: t + 1], np.nan))
        self.closing_available = _ro(np.where(known, closing_available[: t + 1], np.nan))
        self.on_hand_history = _ro(np.where(known, on_hand_hist[: t + 1], np.nan))
        self.snapshot_known = _ro(known)
        self.position = position
        # Only what a buyer could know at t is kept: PO status as of t (true arrival dates and
        # cancellations stay hidden until they happen), announced events without their uplift,
        # transfers with arrival dates only once they have arrived.
        self._po_table = _visible_purchase_orders(purchase_orders, t)
        self._lt_hist = lead_time_history[["po_id", "supplier_id", "lead_time_days"]].copy()
        if len(events):
            self._events = events[events["announce_day"] <= t].drop(columns=["uplift"], errors="ignore").copy()
        else:
            self._events = events.drop(columns=["uplift"], errors="ignore").copy()
        tr = pd.DataFrame(
            [x for x in transfers if x["decided_day"] <= t],
            columns=["sku_idx", "qty", "source", "decided_day", "ship_day", "arrive_day", "policy"],
        )
        tr.loc[tr["arrive_day"] > t, "arrive_day"] = np.nan
        self._transfers = tr
        self.first_planning_day = first_planning_day
        self.launched = _ro(launch_day_known <= t)
        self.launch_day = _ro(np.where(launch_day_known <= t, launch_day_known, -1))
        self.review_period_days = review_period_days

    @property
    def n_sku(self) -> int:
        return len(self.products)

    # ------------------------------------------------------------------ purchase orders

    def purchase_orders(self) -> pd.DataFrame:
        """Every PO with the status the buyer would have seen at the plan date."""
        return self._po_table.copy()

    def open_purchase_orders(self) -> pd.DataFrame:
        po = self.purchase_orders()
        return po[po["status"].isin(["OPEN", "IN_TRANSIT", "DELAYED"])]

    def lead_time_observations(self) -> pd.DataFrame:
        """Received POs (events) and still-open POs (right-censored at their age today)."""
        hist = self._lt_hist.assign(event=True, order_day=np.nan)
        po = self.purchase_orders()
        po = po[po["status"].isin(["RECEIVED", "OPEN", "IN_TRANSIT", "DELAYED"])]
        recv = po["status"] == "RECEIVED"
        sim = pd.DataFrame(
            {
                "po_id": po["po_id"],
                "supplier_id": po["supplier_id"],
                "lead_time_days": np.where(recv, po["actual_day"].fillna(0) - po["order_day"], self.t - po["order_day"]).astype(
                    float
                ),
                "event": recv.to_numpy(),
                "order_day": po["order_day"].astype(float),
            }
        )
        return pd.concat([hist[["po_id", "supplier_id", "lead_time_days", "event", "order_day"]], sim], ignore_index=True)

    # ------------------------------------------------------------------ events and transfers

    def events(self) -> pd.DataFrame:
        return self._events.copy()

    def discontinued(self) -> np.ndarray:
        """SKUs with an announced liquidation: no new purchases."""
        out = np.zeros(self.n_sku, dtype=bool)
        e = self.events()
        if not e.empty:
            out[e.loc[e["kind"] == "LIQUIDATION", "sku_idx"].to_numpy(int)] = True
        return out

    def transfers(self) -> pd.DataFrame:
        return self._transfers.copy()


PO_COLUMNS = [
    "po_id", "sku_idx", "supplier_id", "qty_east", "qty_west", "quantity", "order_day", "expected_day",
    "actual_day", "status", "policy",
]


def _visible_purchase_orders(pos: list[dict], t: int) -> pd.DataFrame:
    """POs as the buyer saw them at t: status from what had happened by t; the actual arrival
    day only for received orders; a cancellation only once the supplier announced it."""
    rows = []
    for po in pos:
        if po["order_day"] > t:
            continue
        ship = po["order_day"] + po["production_days"]
        if po["cancelled"] and t >= ship:
            status, actual = "CANCELLED", None
        elif not po["cancelled"] and po["arrival_day"] <= t:
            status, actual = "RECEIVED", po["arrival_day"]
        elif t > po["expected_day"]:
            status, actual = "DELAYED", None
        elif t >= ship:
            status, actual = "IN_TRANSIT", None
        else:
            status, actual = "OPEN", None
        rows.append((po["po_id"], po["sku_idx"], po["supplier_id"], po["qty"][0], po["qty"][1], po["qty"][0] + po["qty"][1],
                     po["order_day"], po["expected_day"], actual, status, po.get("policy", "")))
    return pd.DataFrame(rows, columns=PO_COLUMNS)
