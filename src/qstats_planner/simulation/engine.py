"""Daily inventory simulation.

Order of events within a day (tested in tests/test_engine.py):
    1. Purchase-order receipts land at the DCs.
    2. FBA transfers that were picked ship from their DC; transfers in transit arrive at Amazon
       (a share goes through an Amazon fulfilment-centre transfer before it can sell).
    3. FBA RESERVED units are set for the day; a small share of them is written off.
    4. Demand arrives. Each DC serves its own region first; a DC that cannot serve its region
       ships from the other DC when that DC has stock left (a cross-DC shipment, at an extra
       fulfilment cost). Amazon demand is served only from Amazon stock. Sales = units served;
       the rest is lost (not backordered) and recorded for the evaluation layer only.
    5. At the end of each Sunday, the policy plans: purchase orders and FBA transfers.

Inventory is conserved: opening on hand + receipts + transfers in - transfers out - sales - write-
offs = closing on hand, for every SKU, location and day.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from qstats_planner.simulation.environment import DCS, EAST, FBA, LOCATIONS, Environment
from qstats_planner.simulation.view import PlannerView, Position


@dataclass
class Decisions:
    orders: list[tuple[int, int, int]] = field(default_factory=list)  # (sku, qty_east, qty_west)
    transfers: list[tuple[int, int, int]] = field(default_factory=list)  # (sku, qty, source_dc)
    notes: dict = field(default_factory=dict)


class Engine:
    def __init__(self, env: Environment, cfg: dict):
        self.env = env
        self.cfg = cfg
        sim = cfg["simulation"]
        n_d, n_s, n_l = env.baseline.shape
        self.n_days, self.n_sku, self.n_loc = n_d, n_s, n_l
        self.warm_end = sim["warmup_weeks"] * 7  # first day with real inventory
        self.first_planning_day = self.warm_end - 1  # the Sunday that closes the warm-up
        self.review = sim["review_period_days"]
        self.pick_days = sim["transfer_pick_days"]
        self.prod_share = cfg["suppliers"]["production_share_of_lead_time"]
        self.reserved_loss = cfg["simulation"]["fba_reserved_loss_share"]

        # state
        self.on_hand = np.zeros((n_s, n_l))
        self.fba_reserved = np.zeros(n_s)
        self.fba_transfer = np.zeros(n_s)
        self.fba_inbound = np.zeros(n_s)
        self.dc_committed = np.zeros((n_s, 2))
        self.pos: list[dict] = []
        self.transfers: list[dict] = []
        self.fc_release: dict[int, list[tuple[int, float]]] = {}
        self._arrivals: dict[int, list[int]] = {}  # day -> PO indices arriving
        self._ships: dict[int, list[int]] = {}  # day -> transfer indices shipping
        self._lands: dict[int, list[int]] = {}  # day -> transfer indices arriving at FBA
        self.day = 0

        # history (observable)
        z = lambda: np.zeros((n_d, n_s, n_l))  # noqa: E731
        self.sales = z()
        self.opening_available = z()
        self.closing_available = z()
        self.closing_on_hand = z()
        self.receipts = z()
        self.transfer_in = z()
        self.transfer_out = z()
        self.write_off = z()
        self.cross_ship = z()  # units shipped from this DC to the other DC's region
        self.reserved_hist = np.zeros((n_d, n_s))
        self.transfer_hist = np.zeros((n_d, n_s))
        self.inbound_hist = np.zeros((n_d, n_s))
        self.committed_hist = np.zeros((n_d, n_s, 2))
        # history (hidden: evaluation layer only)
        self.lost = z()
        self.decision_log: list[dict] = []

    # ------------------------------------------------------------------ helpers

    def clone(self) -> Engine:
        """Deep copy of the whole state, used to fork two worlds from one history."""
        env = self.env
        self.env = None
        twin = copy.deepcopy(self)
        self.env = env
        twin.env = env
        return twin

    def position(self) -> Position:
        avail = self._available()
        return Position(
            on_hand=self.on_hand.copy(),
            available=avail,
            fba_reserved=self.fba_reserved.copy(),
            fba_transfer=self.fba_transfer.copy(),
            fba_inbound=self.fba_inbound.copy(),
            dc_committed=self.dc_committed.copy(),
        )

    def _available(self) -> np.ndarray:
        a = self.on_hand.copy()
        a[:, DCS[0]] -= self.dc_committed[:, 0]
        a[:, DCS[1]] -= self.dc_committed[:, 1]
        a[:, FBA] -= self.fba_reserved + self.fba_transfer
        return np.maximum(a, 0.0)

    def view(self, t: int | None = None) -> PlannerView:
        t = self.day - 1 if t is None else t
        env = self.env
        known = ~env.unknown_mask
        known[: self.warm_end] = False  # warm-up: stock was ample, snapshots are not kept
        return PlannerView(
            t=t,
            days=env.days,
            calendar=env.calendar,
            trading=env.trading,
            products=env.products,
            suppliers=env.suppliers,
            sales=self.sales,
            opening_available=self.opening_available,
            closing_available=self.closing_available,
            on_hand_hist=self.closing_on_hand,
            snapshot_known=known,
            position=self.position(),
            purchase_orders=self.pos,
            lead_time_history=env.lead_time_history,
            events=env.events,
            transfers=self.transfers,
            first_planning_day=self.first_planning_day,
            launch_day_known=env.launch_day,
            review_period_days=self.review,
        )

    # ------------------------------------------------------------------ decisions

    def place_order(self, sku: int, qty_east: int, qty_west: int, policy: str) -> None:
        if qty_east + qty_west <= 0:
            return
        env = self.env
        lt, cancelled = env.lead_time(sku, self.day)
        sup = env.suppliers.iloc[int(env.products.at[sku, "supplier_idx"])]
        quoted = int(sup["quoted_lead_time_days"])
        self.pos.append(
            {
                "po_id": f"PO-{len(self.pos) + 1:06d}",
                "sku_idx": sku,
                "supplier_id": sup["supplier_id"],
                "qty": (int(qty_east), int(qty_west)),
                "order_day": self.day,
                "expected_day": self.day + quoted,
                "arrival_day": self.day + lt,
                "lead_time": lt,
                "production_days": int(round(self.prod_share * lt)),
                "cancelled": cancelled,
                "policy": policy,
            }
        )
        if not cancelled:
            self._arrivals.setdefault(self.day + lt, []).append(len(self.pos) - 1)

    def place_transfer(self, sku: int, qty: int, source: int, policy: str) -> None:
        avail = self.on_hand[sku, source] - self.dc_committed[sku, source]
        qty = int(min(qty, max(avail, 0)))
        if qty <= 0:
            return
        self.dc_committed[sku, source] += qty
        transit = int(self.env.fba_transit[min(self.day // 7, self.env.fba_transit.shape[0] - 1), sku])
        ship = self.day + self.pick_days
        self.transfers.append(
            {
                "sku_idx": sku,
                "qty": qty,
                "source": source,
                "decided_day": self.day,
                "ship_day": ship,
                "arrive_day": ship + transit,
                "policy": policy,
            }
        )
        self._ships.setdefault(ship, []).append(len(self.transfers) - 1)
        self._lands.setdefault(ship + transit, []).append(len(self.transfers) - 1)

    def apply(self, d: Decisions, policy: str) -> None:
        for sku, qe, qw in d.orders:
            self.place_order(sku, qe, qw, policy)
        for sku, qty, src in d.transfers:
            self.place_transfer(sku, qty, src, policy)
        self.decision_log.append(
            {"day": self.day, "policy": policy, "orders": len(d.orders), "transfers": len(d.transfers), **d.notes}
        )

    # ------------------------------------------------------------------ one day

    def step(self) -> None:
        env, t = self.env, self.day
        # launch stock for SKUs launched after the warm-up (identical in every world)
        launching = np.where((env.launch_day == t) & (env.launch_stock > 0))[0]
        for i in launching:
            q = env.launch_stock[i]
            east = np.floor(q * 0.6 / env.products.at[i, "case_pack"]) * env.products.at[i, "case_pack"]
            self.on_hand[i, EAST] += east
            self.on_hand[i, DCS[1]] += q - east
            self.receipts[t, i, EAST] += east
            self.receipts[t, i, DCS[1]] += q - east

        # 1. receipts
        for k_po in self._arrivals.pop(t, []):
            po = self.pos[k_po]
            i = po["sku_idx"]
            for k, dc in enumerate(DCS):
                self.on_hand[i, dc] += po["qty"][k]
                self.receipts[t, i, dc] += po["qty"][k]
        # 2. transfers ship and arrive
        for k_tr in self._ships.pop(t, []):
            tr = self.transfers[k_tr]
            i, q, src = tr["sku_idx"], tr["qty"], tr["source"]
            self.dc_committed[i, src] -= q
            self.on_hand[i, src] -= q
            self.transfer_out[t, i, src] += q
            self.fba_inbound[i] += q
        for k_tr in self._lands.pop(t, []):
            tr = self.transfers[k_tr]
            i, q = tr["sku_idx"], tr["qty"]
            self.fba_inbound[i] -= q
            self.on_hand[i, FBA] += q
            self.transfer_in[t, i, FBA] += q
            moving = np.floor(0.3 * q)
            if moving > 0:
                self.fba_transfer[i] += moving
                self.fc_release.setdefault(t + 4, []).append((i, moving))
        for i, q in self.fc_release.pop(t, []):
            self.fba_transfer[i] -= q
        # 3. FBA reserved units and write-offs
        free = np.maximum(self.on_hand[:, FBA] - self.fba_transfer, 0)
        self.fba_reserved = np.floor(free * env.fba_unavailable_share[t])
        loss = np.floor(self.fba_reserved * self.reserved_loss / 7 + 0.5 * (self.fba_reserved > 20))
        loss = np.minimum(loss, self.fba_reserved)
        self.on_hand[:, FBA] -= loss
        self.fba_reserved -= loss
        self.write_off[t, :, FBA] = loss
        # 4. demand
        if t < self.warm_end:
            avail = np.full((self.n_sku, self.n_loc), np.inf)
        else:
            avail = self._available()
        self.opening_available[t] = np.where(np.isinf(avail), np.nan, avail)
        demand = env.baseline[t] if env.trading[t] else np.zeros((self.n_sku, self.n_loc))
        sold = np.minimum(demand, avail)
        if t >= self.warm_end:
            left = avail - sold
            unmet = demand - sold
            e, w_ = DCS
            to_w = np.minimum(unmet[:, w_], left[:, e])  # EAST ships to WEST customers
            to_e = np.minimum(unmet[:, e], left[:, w_])  # WEST ships to EAST customers
            self.cross_ship[t, :, e] = to_w
            self.cross_ship[t, :, w_] = to_e
            shipped = sold.copy()
            shipped[:, e] += to_w
            shipped[:, w_] += to_e
            sold = sold.copy()
            sold[:, w_] += to_w
            sold[:, e] += to_e
            self.on_hand -= shipped
        self.sales[t] = sold
        self.lost[t] = demand - sold
        self.closing_available[t] = np.nan if t < self.warm_end else self._available()
        self.closing_on_hand[t] = np.nan if t < self.warm_end else self.on_hand
        self.reserved_hist[t] = self.fba_reserved
        self.transfer_hist[t] = self.fba_transfer
        self.inbound_hist[t] = self.fba_inbound
        self.committed_hist[t] = self.dc_committed
        self.day += 1

    def run(self, policy, until: int | None = None) -> Engine:
        """Advance day by day to `until` (exclusive), planning at the end of every Sunday."""
        until = self.n_days if until is None else until
        while self.day < until:
            t = self.day
            self.step()
            if t == self.first_planning_day:
                init = policy.initial_stock(self.view(t))
                self.on_hand[:] = init
                self.closing_on_hand[t] = init
                self.closing_available[t] = self._available()
            if t >= self.first_planning_day and (t + 1) % self.review == 0:
                self.apply(policy.plan(self.view(t)), policy.name)
        return self


LOC_NAMES = LOCATIONS
