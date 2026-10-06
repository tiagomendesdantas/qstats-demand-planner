"""Shared plumbing for planning policies (weekly arrays, positions, DC shares)."""

from __future__ import annotations

import numpy as np

from qstats_planner.simulation.environment import EAST, FBA, WEST
from qstats_planner.simulation.view import PlannerView


def weekly(daily: np.ndarray, t: int) -> np.ndarray:
    """Sum complete Monday-Sunday weeks up to day t (inclusive). Day 0 is a Monday."""
    n_weeks = (t + 1) // 7
    arr = np.nan_to_num(daily[: n_weeks * 7])
    return arr.reshape(n_weeks, 7, *arr.shape[1:]).sum(axis=1)


def weekly_trading_days(view: PlannerView) -> np.ndarray:
    n_weeks = (view.t + 1) // 7
    return view.trading[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1).astype(float)


def network_position(view: PlannerView, reserved_weight: float = 1.0) -> np.ndarray:
    """On hand everywhere + in transit to Amazon + open purchase orders.

    `reserved_weight` < 1 discounts FBA RESERVED units (some never come back to sellable).
    """
    p = view.position
    on_hand = p.on_hand.sum(axis=1) - (1 - reserved_weight) * p.fba_reserved
    open_po = view.open_purchase_orders().groupby("sku_idx")["quantity"].sum()
    po = np.zeros(view.n_sku)
    po[open_po.index.to_numpy(int)] = open_po.to_numpy()
    return on_hand + p.fba_inbound + po


def dc_east_share(sales_weekly_loc: np.ndarray, preferred_east: np.ndarray, weeks: int = 13) -> np.ndarray:
    """Share of the replenishment that should land at EAST_DC: DC demand by region, with FBA
    demand attributed to the DC that supplies Amazon."""
    recent = sales_weekly_loc[-weeks:].sum(axis=0) if len(sales_weekly_loc) else np.zeros((0, 3))
    east = recent[:, EAST] + np.where(preferred_east, recent[:, FBA], 0)
    west = recent[:, WEST] + np.where(preferred_east, 0, recent[:, FBA])
    total = east + west
    return np.where(total > 0, east / np.maximum(total, 1e-9), 0.6)


def ses_levels(y: np.ndarray, alpha: float, valid: np.ndarray) -> np.ndarray:
    """Simple exponential smoothing, run over weeks (rows) for every SKU (columns) at once.

    Weeks marked invalid (before launch, or no trading days) leave the level unchanged.
    Returns the level after each week; NaN until a SKU's first valid week.
    """
    level = np.full(y.shape[1], np.nan)
    out = np.full(y.shape, np.nan)
    for w in range(y.shape[0]):
        v = valid[w]
        first = v & np.isnan(level)
        level = np.where(first, y[w], level)
        upd = v & ~first
        level = np.where(upd, alpha * y[w] + (1 - alpha) * level, level)
        out[w] = level
    return out
