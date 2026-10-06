"""Forward inventory projection (180 days) and projected stockout dates.

Projected sellable stock on day d = sellable stock now + receipts expected by d - expected demand
through d. Open POs land on their expected date (a DELAYED PO, already past its date, is assumed
one week out). Transfers to Amazon move stock inside the network and do not change the network
total. A second path adds the recommended order at the median lead time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scheduled_receipts(open_pos: pd.DataFrame, t: int, n: int, horizon: int) -> np.ndarray:
    """(horizon, n) units expected to arrive on each future day (day 0 = tomorrow)."""
    rec = np.zeros((horizon, n))
    for po in open_pos.itertuples():
        day = po.expected_day if po.status != "DELAYED" else t + 7
        off = int(max(day, t + 1) - (t + 1))
        if off < horizon:
            rec[off, int(po.sku_idx)] += po.quantity
    return rec


def project(stock_now: np.ndarray, daily_fc: np.ndarray, receipts: np.ndarray, horizon: int) -> np.ndarray:
    """(horizon, n) expected sellable stock at the end of each future day (may go negative:
    negative = expected unmet demand, shown as zero in the app)."""
    d = daily_fc[:horizon]
    return stock_now[None, :] + np.cumsum(receipts[:horizon] - d, axis=0)


def stockout_day(path: np.ndarray) -> np.ndarray:
    """Index of the first day the projection is at or below zero; -1 if never in the horizon."""
    hit = path <= 0
    first = hit.argmax(axis=0)
    return np.where(hit.any(axis=0), first, -1)


def expected_lost(stock_now: np.ndarray, daily_fc: np.ndarray, receipts: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Expected units of demand lost (not backordered) over the next `days` days per SKU, given
    today's stock and the receipts already scheduled. Uses expected daily demand, so it is a
    point estimate: lumpy demand makes the true figure vary around it."""
    n = len(stock_now)
    horizon = int(np.max(days)) if len(days) else 0
    stock = np.asarray(stock_now, float).copy()
    lost = np.zeros(n)
    for d in range(horizon):
        stock = stock + receipts[d]
        short = np.maximum(daily_fc[d] - stock, 0)
        lost += np.where(d < days, short, 0)
        stock = np.maximum(stock - daily_fc[d], 0)
    return lost
