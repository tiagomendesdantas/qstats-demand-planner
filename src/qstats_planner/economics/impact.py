"""Portfolio economics of a plan, computed from the plan state (nothing hard-coded).

inventory value          on hand at every location x landed unit cost
purchase recommendations purchase lines and their value at cost
SKUs at stockout risk    projected to run out within `stockout_risk_weeks` without a new order
contribution at risk     expected units short over lead time + review at today's position,
                         x contribution per unit
excess inventory value   on-hand stock beyond `excess_weeks_of_cover` of forecast demand, at cost
                         (per SKU in sku_plan.excess_value; open orders are not counted)
service level            demand-weighted probability of covering lead time + review demand,
                         at today's position and after the recommended orders
forecast WAPE / bias     champion one-week-ahead errors over the last 26 scored weeks
inventory turns          trailing 13-week cost of goods sold, annualised / inventory value
"""

from __future__ import annotations

import numpy as np

PURCHASE_ACTIONS = ["BUY", "REVIEW_FORECAST", "LOW_MARGIN"]  # every purchase line, approved or awaiting review


def portfolio_kpis(view, st, plan: dict, cfg: dict) -> dict:
    inv, biz = cfg["inventory"], cfg["business"]
    prod = view.products
    sp = plan["sku_plan"]
    recs = plan["recommendations"]
    cost = prod["unit_cost"].to_numpy(float)
    cm = prod["contribution_margin"].to_numpy(float)
    launched = view.launched
    on_hand = view.position.on_hand.sum(axis=1)
    weekly = sp["weekly_demand"].to_numpy()
    risk_days = inv["stockout_risk_weeks"] * 7
    so = sp["stockout_day"].to_numpy()
    at_risk = launched & (so >= 0) & (so < risk_days)
    buys = recs[recs["action"].isin(PURCHASE_ACTIONS) & (recs["recommended_quantity"] > 0)].drop_duplicates("sku_idx")
    w = np.where(launched, weekly, 0)
    bt = st.forecast_state.backtest
    wape = bias = np.nan
    if bt is not None:
        champ = st.forecast_state.selection.champion
        n = len(champ)
        f = bt.week1_forecast[champ, :, np.arange(n)].T
        a = bt.week1_actual
        m = np.isfinite(f) & np.isfinite(a) & bt.scored[1]
        m[: max(0, m.shape[0] - 27)] = False
        if m.any():
            wape = float(np.abs(f[m] - a[m]).sum() / a[m].sum())
            bias = float((f[m] - a[m]).sum() / a[m].sum())
    sold13 = view.sales[-91:].sum(axis=(0, 2))
    value = float((on_hand * cost).sum())
    by_action = recs["action"].value_counts().to_dict()
    cat = prod["category"].to_numpy()
    risk_by_cat = {c: int(at_risk[cat == c].sum()) for c in np.unique(cat)}
    value_by_cat = {c: float((on_hand * cost)[cat == c].sum()) for c in np.unique(cat)}
    return {
        "skus_monitored": int(launched.sum()),
        "inventory_value": value,
        "inventory_units": float(on_hand.sum()),
        "purchase_lines": int(len(buys)),
        "purchase_lines_for_review": int((buys["action"] != "BUY").sum()),
        "purchase_value": float((buys["recommended_quantity"] * cost[buys["sku_idx"].to_numpy()]).sum()),
        "skus_at_stockout_risk": int(at_risk.sum()),
        "contribution_at_risk": float((sp["shortfall_before"].to_numpy() * cm)[launched].sum()),
        "excess_inventory_value": float(sp["excess_value"].sum()),
        "service_level_now": float((sp["service_before"].to_numpy() * w).sum() / max(w.sum(), 1e-9)),
        "service_level_after_plan": float((sp["service_after"].to_numpy() * w).sum() / max(w.sum(), 1e-9)),
        "forecast_wape_26w": wape,
        "forecast_bias_26w": bias,
        "inventory_turns": float((sold13 * cost).sum() * 4 / value) if value > 0 else np.nan,
        "recommendations_by_action": by_action,
        "stockout_risk_by_category": risk_by_cat,
        "inventory_value_by_category": value_by_cat,
        "holding_cost_rate": biz["holding_cost_annual_pct"],
        "minimum_margin_pct": biz["minimum_margin_pct"],
        "excess_weeks_of_cover": inv["excess_weeks_of_cover"],
        "stockout_risk_weeks": inv["stockout_risk_weeks"],
    }
