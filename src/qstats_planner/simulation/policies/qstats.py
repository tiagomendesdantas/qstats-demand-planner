"""The QStats planner as a simulation policy: the live planning cycle, run every week."""

from __future__ import annotations

import numpy as np

from qstats_planner.replenishment.policy import PlanSettings, run_cycle
from qstats_planner.simulation.engine import Decisions
from qstats_planner.simulation.environment import EAST, FBA, WEST
from qstats_planner.simulation.view import PlannerView


class QStatsPlanner:
    def __init__(self, cfg: dict, prior, settings: PlanSettings | None = None, name: str | None = None):
        self.cfg = cfg
        self.prior = prior
        self.settings = settings or PlanSettings()
        self.name = name or "qstats"
        self.state = None
        self.reselect = cfg["forecasting"]["reselect_every_weeks"]
        self.history: list[dict] = []

    def _cycle(self, view: PlannerView):
        W = (view.t + 1) // 7
        prev = self.state.forecast_state if self.state is not None else None
        refit = prev is None or (W - 1 - prev.fitted_week) >= self.reselect
        st = run_cycle(view, self.cfg, self.prior, self.settings, prev, refit=refit)
        self.state = st
        return st

    def initial_stock(self, view: PlannerView) -> np.ndarray:
        st = self._cycle(view)
        out = np.zeros((view.n_sku, 3))
        fba = st.fba["fba_target"].to_numpy()
        dc = np.maximum(st.target - fba, 0)
        east = self.cfg["business"]["locations"][0]["region_share"]
        out[:, EAST] = np.round(dc * east)
        out[:, WEST] = np.round(dc * (1 - east))
        out[:, FBA] = np.round(fba)
        out[~view.launched] = 0
        return out

    def plan(self, view: PlannerView) -> Decisions:
        st = self._cycle(view)
        d = Decisions()
        d.orders = [(int(i), int(st.east_qty[i]), int(st.west_qty[i])) for i in np.where(st.order_qty > 0)[0]]
        d.transfers = [(r["sku_idx"], r["qty"], r["source"]) for r in st.fba.attrs["transfers"]]
        # what the planner believed when it decided: used for calibration in the evaluation layer
        self.history.append({
            "t": view.t,
            "ltd_mean": st.ltd.mean.copy(),
            "ltd_q": {q: v.copy() for q, v in st.ltd.quantiles.items()},
            "lt_p50": np.array([d_.quantile(0.5) for d_ in st.sku_lead_time]),
            "segments": st.forecast_state.segments.copy(),
            "champion": st.forecast_state.selection.champion.copy(),
            "weekly_fc_1": st.weekly_fc[0].copy(),
        })
        return d

