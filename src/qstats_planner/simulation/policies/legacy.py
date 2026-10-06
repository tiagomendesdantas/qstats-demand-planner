"""The Legacy planner: a reasonable, conventional replenishment process.

It is what many importers run in a spreadsheet or an ERP module, configured sensibly:
    forecast      simple exponential smoothing of weekly sales (alpha tuned on the dev SKUs);
                  closed weeks are skipped; sales are taken as demand (no stockout correction)
    lead time     the supplier's quoted lead time, treated as certain
    safety stock  N days of forecast demand (30 by default)
    policy        weekly review; when the inventory position (on hand everywhere + in transit to
                  Amazon + open POs) is below lead-time demand + review-period demand + safety
                  stock, order up to that level, with the same MOQ and case-pack rounding as QStats
    Amazon FBA    replenishment time = pick + average transit + review period; when FBA cover
                  falls below that + 14 days, send enough for that + 30 days

It is not a straw man: same review cadence, same rounding, same visibility of open POs, and its
smoothing constant is tuned the same way the QStats parameters are.
"""

from __future__ import annotations

import numpy as np

from qstats_planner.replenishment.order_quantity import round_orders, round_up_to_pack, split_by_share
from qstats_planner.simulation.engine import Decisions
from qstats_planner.simulation.environment import EAST, FBA, WEST
from qstats_planner.simulation.policies.base import (
    dc_east_share,
    network_position,
    ses_levels,
    weekly,
    weekly_trading_days,
)
from qstats_planner.simulation.view import PlannerView


class LegacyPlanner:
    def __init__(
        self,
        cfg: dict,
        safety_days: float | None = None,
        alpha: float | None = None,
        seasonal_prior=None,
        name: str | None = None,
    ):
        lg = cfg["legacy"]
        self.alpha = lg["ses_alpha"] if alpha is None else alpha
        self.safety_days = lg["safety_days"] if safety_days is None else safety_days
        self.fba_cover = lg["fba_cover_days"]
        self.fba_trigger = lg["fba_trigger_extra_days"]
        sim = cfg["simulation"]
        self.fba_replenish_days = sim["transfer_pick_days"] + float(np.mean(sim["fba_transit_days"])) + sim["review_period_days"]
        self.prior = seasonal_prior  # optional: "Legacy + seasonal prior" comparison row
        self.name = name or f"legacy_{int(self.safety_days)}d"
        self.history: list[dict] = []

    # ------------------------------------------------------------------ forecasts

    def _daily_rate(self, view: PlannerView, weekly_units: np.ndarray, horizon_days: np.ndarray) -> np.ndarray:
        """Expected units per calendar day over the coming `horizon_days`, per SKU."""
        exposure = weekly_trading_days(view)
        valid = (exposure > 0)[:, None] & (np.arange(len(exposure))[:, None] >= view.launch_day[None, :] // 7)
        if self.prior is None:
            lvl = ses_levels(weekly_units, self.alpha, valid)[-1]
            return np.nan_to_num(lvl) / 7.0
        # with the seasonal prior: smooth the seasonally adjusted series, re-apply the prior ahead
        s_hist = self.prior.weekly_factors(view, past=True)
        lvl = ses_levels(weekly_units / s_hist, self.alpha, valid)[-1]
        ahead = self.prior.mean_factor_ahead(view, horizon_days)
        return np.nan_to_num(lvl) * ahead / 7.0

    # ------------------------------------------------------------------ policy

    def initial_stock(self, view: PlannerView) -> np.ndarray:
        prod = view.products
        net = weekly(view.sales, view.t).sum(axis=2)
        loc = weekly(view.sales, view.t)
        lt = prod["supplier_idx"].map(view.suppliers["quoted_lead_time_days"]).to_numpy(float)
        rate = self._daily_rate(view, net, lt)
        fba_rate = self._fba_rate(view, loc)
        target = rate * (lt + view.review_period_days + self.safety_days)
        fba = np.where(prod["fba_enabled"], fba_rate * (self.fba_cover + self.fba_replenish_days), 0)
        dc_total = np.maximum(target - fba, 0)
        east = dc_east_share(loc, (prod["preferred_source_dc"] == "EAST_DC").to_numpy())
        out = np.zeros((view.n_sku, 3))
        out[:, EAST] = np.round(dc_total * east)
        out[:, WEST] = np.round(dc_total * (1 - east))
        out[:, FBA] = np.round(fba)
        out[~view.launched] = 0
        return out

    def _fba_rate(self, view: PlannerView, loc_weekly: np.ndarray) -> np.ndarray:
        exposure = weekly_trading_days(view)
        valid = (exposure > 0)[:, None] & (np.arange(len(exposure))[:, None] >= view.launch_day[None, :] // 7)
        return np.nan_to_num(ses_levels(loc_weekly[:, :, FBA], self.alpha, valid)[-1]) / 7.0

    def plan(self, view: PlannerView) -> Decisions:
        prod = view.products
        loc = weekly(view.sales, view.t)
        net = loc.sum(axis=2)
        lt = prod["supplier_idx"].map(view.suppliers["quoted_lead_time_days"]).to_numpy(float)
        R = view.review_period_days
        rate = self._daily_rate(view, net, lt + R)
        target = rate * (lt + R + self.safety_days)
        ip = network_position(view, reserved_weight=1.0)
        need = (ip < target) & view.launched & ~view.discontinued()
        raw = np.where(need, target - ip, 0)
        qty = round_orders(raw, prod["case_pack"].to_numpy(), prod["moq"].to_numpy())
        pref_east = (prod["preferred_source_dc"] == "EAST_DC").to_numpy()
        east, west = split_by_share(qty, dc_east_share(loc, pref_east), prod["case_pack"].to_numpy(), pref_east)
        d = Decisions()
        d.orders = [(int(i), int(east[i]), int(west[i])) for i in np.where(qty > 0)[0]]
        self.history.append({"t": view.t, "weekly_fc_1": rate * 7.0, "target": target.copy()})

        # Amazon FBA: days-of-cover rule
        fba_rate = self._fba_rate(view, loc)
        p = view.position
        fba_pos = p.on_hand[:, FBA] + p.fba_inbound + p.dc_committed.sum(axis=1)
        trigger = fba_rate * (self.fba_replenish_days + self.fba_trigger)
        send = np.where(
            prod["fba_enabled"] & (fba_pos < trigger) & (fba_rate > 0),
            fba_rate * (self.fba_cover + self.fba_replenish_days) - fba_pos,
            0,
        )
        send = round_up_to_pack(send, prod["case_pack"].to_numpy())
        avail = p.available
        for i in np.where(send > 0)[0]:
            src = EAST if pref_east[i] else WEST
            if avail[i, src] < send[i] and avail[i, 1 - src] > avail[i, src]:
                src = 1 - src
            d.transfers.append((int(i), int(send[i]), int(src)))
        return d
