"""Exception-based recommendations: what to buy, move or review, why, and what it is worth.

Every recommendation answers WHAT (action and quantity), WHY (one sentence from the numbers),
EVIDENCE (the inputs that drove it), EXPECTED EFFECT (service before -> after) and CONFIDENCE.
Nothing here is a hard-coded improvement: every figure comes from the plan state.

Actions
    CRITICAL_STOCKOUT  out of stock now, or projected out before any order placed today can arrive
    BUY                inventory position below the order-up-to level
    EXPEDITE           projected out before an open PO's expected arrival
    TRANSFER           one DC nearly out while the other holds far more than it needs
    SEND_TO_FBA        Amazon stock below its target; units move from a DC
    EXCESS             more than `excess_weeks_of_cover` of stock
    LOW_MARGIN         a BUY on a SKU under the margin threshold: routed to review, not bought
    REVIEW_FORECAST    a BUY whose forecast confidence is LOW
    STOCKOUT_CENSORED  recent sales were held down by stockouts; demand was reconstructed
    NO_ACTION
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from qstats_planner.domain.locations import EAST, FBA, WEST
from qstats_planner.inventory.lead_time_demand import expected_shortfall, service_level
from qstats_planner.inventory.projection import expected_lost, project, scheduled_receipts, stockout_day
from qstats_planner.replenishment.order_quantity import round_up_to_pack

SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}
NO_DEMAND_DAILY = 0.01  # a forecast under one unit per hundred days is read as no forecast demand
DC_NAMES = {EAST: "EAST_DC", WEST: "WEST_DC"}


def _weeks(x: float) -> str:
    return "over a year" if x > 52 else f"{x:.1f} weeks"


def confidence_score(
    history_weeks: np.ndarray, segments: np.ndarray, imputed_share: np.ndarray, selection_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """0-1 score and HIGH / MEDIUM / LOW label. Documented in docs/forecasting_methodology.md."""
    s = np.ones(len(segments))
    s -= np.where(history_weeks < 26, 0.35, np.where(history_weeks < 52, 0.15, 0.0))
    s -= np.where(np.isin(segments, ["NEW_PRODUCT"]), 0.15, 0.0)
    s -= np.where(np.isin(segments, ["INTERMITTENT", "VOLATILE"]), 0.10, 0.0)
    s -= np.clip(imputed_share - 0.10, 0, 0.5) * 0.8
    err = np.nan_to_num(selection_error, nan=0.6)
    s -= np.clip(err - 0.35, 0, 0.6) * 0.5
    s = np.clip(s, 0, 1)
    label = np.where(s >= 0.70, "HIGH", np.where(s >= 0.45, "MEDIUM", "LOW"))
    return s, label


def build(view, st, cfg: dict, created_at: pd.Timestamp) -> dict:
    rc, inv, biz = cfg["recommendations"], cfg["inventory"], cfg["business"]
    prod = view.products
    n = view.n_sku
    t = view.t
    plan_date = view.date + pd.Timedelta(days=1)  # the plan applies from Monday
    horizon = cfg["forecasting"]["planning_horizon_days"]
    p = view.position
    fs = st.forecast_state

    lt_p50 = np.array([d.quantile(0.5) for d in st.sku_lead_time])
    lt_p90 = np.array([d.quantile(0.9) for d in st.sku_lead_time])
    mean_ltd = st.ltd.mean
    ss = np.maximum(st.target - mean_ltd, 0)
    on_hand = p.on_hand.sum(axis=1)  # physical units at every location
    stock_now = on_hand - (1 - inv["reserved_planning_weight"]) * p.fba_reserved + p.fba_inbound
    open_po = view.open_purchase_orders()
    on_order = open_po.groupby("sku_idx")["quantity"].sum().reindex(range(n), fill_value=0).to_numpy()
    rec = scheduled_receipts(open_po, t, n, horizon)
    path = project(stock_now, st.daily_fc, rec, horizon)
    rec_new = rec.copy()
    for i in np.where(st.order_qty > 0)[0]:
        off = int(min(max(lt_p50[i], 1), horizon) - 1)
        rec_new[off, i] += st.order_qty[i]
    path_new = project(stock_now, st.daily_fc, rec_new, horizon)
    so = stockout_day(path)
    so_new = stockout_day(path_new)
    # demand expected to be lost before an order placed today could land, given the open pipeline
    lost_before_new = expected_lost(stock_now, st.daily_fc, rec, np.minimum(lt_p50, horizon).astype(int))
    # What this week's order is worth, on the same timing-aware basis: the demand it serves from its
    # arrival until the next weekly order can land (one review period later) that would otherwise
    # be lost. Demand before it lands is the CRITICAL line's loss, so the two never overlap. It is an
    # expected-value projection (demand at its forecast), so it understates what safety stock buys.
    lands = np.minimum(np.maximum(lt_p50, 1), horizon).astype(int) - 1  # the day rec_new adds the order
    window_end = np.minimum(lands + view.review_period_days, horizon)
    protected_units = expected_lost(stock_now, st.daily_fc, rec, window_end) - expected_lost(
        stock_now, st.daily_fc, rec_new, window_end
    )
    # the next 13 weeks for the portfolio figures: open POs on their dates, unmet demand lost
    days13 = np.full(n, min(91, horizon))
    demand_13w = st.daily_fc[: days13[0]].sum(axis=0)
    lost_13w_now = expected_lost(stock_now, st.daily_fc, rec, days13)
    lost_13w_plan = expected_lost(stock_now, st.daily_fc, rec_new, days13)
    weekly13 = st.daily_fc[:91].sum(axis=0) / 13
    no_fc = weekly13 < 7 * NO_DEMAND_DAILY
    per_week = np.where(no_fc, 1.0, weekly13)
    cover_now = np.where(no_fc, np.inf, (stock_now + on_order) / per_week)
    cover_after = np.where(no_fc, np.inf, (stock_now + on_order + st.order_qty) / per_week)
    h = st.history
    imputed_share = h.imputed[-8:].sum(axis=0) / np.maximum(h.Y[-8:].sum(axis=0), 1e-9)
    hist_weeks = h.valid.sum(axis=0)
    sel_err = (
        np.nanmean(fs.extra["selection_error"][fs.selection.champion, :, np.arange(n)], axis=1)
        if fs.extra.get("selection_error") is not None
        else np.full(n, np.nan)
    )
    conf_score, conf = confidence_score(hist_weeks, fs.segments, imputed_share, sel_err)
    cm = prod["contribution_margin"].to_numpy(float)
    cm_pct = prod["contribution_margin_pct"].to_numpy(float)
    cost = prod["unit_cost"].to_numpy(float)
    low_margin = cm_pct < biz["minimum_margin_pct"]
    discontinued = view.discontinued()
    next_po = (
        open_po.assign(eday=np.where(open_po["status"] == "DELAYED", t + 7, open_po["expected_day"]))
        .sort_values("eday")
        .groupby("sku_idx")
        .first()
    )

    svc_before = np.array([service_level(st.ltd.samples[i], st.position[i]) for i in range(n)])
    svc_after = np.array([service_level(st.ltd.samples[i], st.position[i] + st.order_qty[i]) for i in range(n)])
    short_before = np.array([expected_shortfall(st.ltd.samples[i], st.position[i]) for i in range(n)])
    short_after = np.array([expected_shortfall(st.ltd.samples[i], st.position[i] + st.order_qty[i]) for i in range(n)])
    protected = protected_units * cm

    day = lambda k: (plan_date + pd.Timedelta(days=int(k))) if k >= 0 else pd.NaT  # noqa: E731
    rows = []

    def add(i, action, severity, qty, location, why, evidence, effect, impact, stockout=None):
        rows.append(
            {
                "sku_idx": int(i),
                "sku": prod.at[i, "sku"],
                "description": prod.at[i, "description"],
                "category": prod.at[i, "category"],
                "supplier_id": prod.at[i, "supplier_id"],
                "segment": fs.segments[i],
                "location": location,
                "action": action,
                "severity": severity,
                "recommended_quantity": int(qty),
                "stockout_date": stockout if stockout is not None else day(so[i]),
                "inventory_position": float(st.position[i]),
                "confidence": conf[i],
                "confidence_score": round(float(conf_score[i]), 2),
                "reason": why,
                "economic_impact": round(float(impact), 2),
                "evidence": json.dumps(
                    {
                        k: (round(float(v), 2) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in evidence.items()
                    }
                ),
                "expected_effect": effect,
                "created_at": created_at,
            }
        )

    for i in range(n):
        if not view.launched[i]:
            continue
        base_ev = {
            "Inventory position": st.position[i],
            "On hand (network)": float(p.on_hand[i].sum()),
            "On order": on_order[i],
            "P50 demand over lead time + review": st.ltd.quantiles[0.5][i],
            "P90 demand over lead time + review": st.ltd.quantiles[0.9][i],
            "Order-up-to level": st.target[i],
            "Safety stock": ss[i],
            "Lead time P50 (days)": lt_p50[i],
            "Lead time P90 (days)": lt_p90[i],
            "Service target": float(st.alpha[i]),
            "MOQ": int(prod.at[i, "moq"]),
            "Case pack": int(prod.at[i, "case_pack"]),
            "Forecast model": fs.models[fs.selection.champion[i]].name,
            "Segment": fs.segments[i],
        }
        qty = int(st.order_qty[i])
        so_date = day(so[i])
        arrival_new = day(lt_p50[i])
        effect = f"Cycle service over lead time + review: {svc_before[i]:.0%} -> {svc_after[i]:.0%}"
        acted = False

        # out of stock now / before any new order could land
        net_avail = p.available[i].sum()
        unavoidable = so[i] >= 0 and so[i] < lt_p50[i]
        if qty > 0 or net_avail <= 0 or unavoidable:
            if net_avail <= 0 or (unavoidable and so[i] <= 14):
                gap_days = max(int(lt_p50[i] - max(so[i], 0)), 0)
                lost = lost_before_new[i]
                why = (
                    "Out of stock in the network now."
                    if net_avail <= 0
                    else f"Projected to run out on {so_date:%b %d}, {gap_days} days before an order placed today could arrive."
                )
                if i in next_po.index:
                    npo = next_po.loc[i]
                    why += (
                        f" Next receipt: {npo['po_id']}, {int(npo['quantity']):,} units, due "
                        f"{day(npo['eday'] - t - 1):%b %d}" + (" (already late)." if npo["status"] == "DELAYED" else ".")
                    )
                else:
                    why += " No purchase order is open."
                if qty > 0:
                    why += f" A purchase of {qty:,} units is recommended on its own line."
                add(
                    i,
                    "CRITICAL_STOCKOUT",
                    "CRITICAL",
                    0,
                    "NETWORK",
                    why,
                    {**base_ev, "Expected units lost before a new order lands": lost},
                    f"About {lost:,.0f} units of demand expected to be lost before a new order could land (open POs counted)",
                    -lost * cm[i],  # a loss in the status quo, not value protected (as EXCESS carries its cost)
                )
                acted = True

        if qty > 0 and not discontinued[i]:
            ev = {
                **base_ev,
                "Raw requirement": st.raw_requirement[i],
                "Recommended (rounded)": qty,
                "To EAST_DC": int(st.east_qty[i]),
                "To WEST_DC": int(st.west_qty[i]),
                "Expected arrival": f"{arrival_new:%b %d}",
                "Purchase value": qty * cost[i],
            }
            why = (
                f"Inventory position {st.position[i]:,.0f} is below the order-up-to level {st.target[i]:,.0f} "
                f"(demand over {lt_p50[i] + view.review_period_days:.0f} days at the {st.alpha[i]:.0%} quantile)."
            )
            if so[i] >= 0:
                why += f" Without an order, stock runs out on {so_date:%b %d}."
            if qty > st.raw_requirement[i] * 1.25:
                why += f" MOQ / case pack round the order up from {st.raw_requirement[i]:,.0f} to {qty:,}."
            if low_margin[i]:
                add(
                    i,
                    "LOW_MARGIN",
                    "MEDIUM",
                    qty,
                    "NETWORK",
                    f"Would buy {qty:,} units, but contribution margin is {cm_pct[i]:.0%}, under the "
                    f"{biz['minimum_margin_pct']:.0%} threshold. Review price, cost or channel before buying.",
                    ev,
                    effect,
                    protected[i],
                )
            elif conf[i] == "LOW" and rc["low_confidence_review"]:
                add(
                    i,
                    "REVIEW_FORECAST",
                    "MEDIUM",
                    qty,
                    "NETWORK",
                    why + f" Forecast confidence is LOW ({fs.segments[i].lower().replace('_', ' ')}, "
                    f"{int(hist_weeks[i])} weeks of history, {imputed_share[i]:.0%} of recent demand reconstructed).",
                    ev,
                    effect,
                    protected[i],
                )
            else:
                sev = "HIGH" if (so[i] >= 0 and so[i] < lt_p50[i] + 14) or protected[i] > 2000 else "MEDIUM"
                add(i, "BUY", sev, qty, "NETWORK", why, ev, effect, protected[i])
            acted = True

        # expedite an open PO that lands after the projected stockout
        if i in next_po.index and so[i] >= 0:
            po = next_po.loc[i]
            gap = int(po["eday"] - (t + 1) - so[i])
            if gap > rc["expedite_gap_days"]:
                lost = float(expected_lost(stock_now[[i]], st.daily_fc[:, [i]], rec[:, [i]], np.array([so[i] + gap]))[0])
                add(
                    i,
                    "EXPEDITE",
                    "HIGH",
                    int(po["quantity"]),
                    "NETWORK",
                    f"Projected out on {so_date:%b %d}; open PO {po['po_id']} is due {gap} days later.",
                    {
                        **base_ev,
                        "PO": po["po_id"],
                        "PO status": po["status"],
                        "PO expected": f"{day(po['eday'] - t - 1):%b %d}",
                        "Days short": gap,
                    },
                    f"Pulling the PO forward covers about {lost:,.0f} units of demand",
                    lost * cm[i],
                )
                acted = True

        # inter-DC transfer
        direct_week = weekly13[i] * (1 - st.fba_share[i])
        if direct_week >= 7 * NO_DEMAND_DAILY:
            east_share = biz["locations"][0]["region_share"]
            need = np.array([direct_week * east_share, direct_week * (1 - east_share)])
            cover_dc = p.available[i, [EAST, WEST]] / np.maximum(need, 1e-9)
            for poor, rich in ((EAST, WEST), (WEST, EAST)):
                pi, ri = (0, 1) if poor == EAST else (1, 0)
                if cover_dc[pi] < 2 and cover_dc[ri] > 8:
                    q = min(p.available[i, rich] - 6 * need[ri], 4 * need[pi] - p.available[i, poor])
                    q = int(round_up_to_pack(np.array([q]), np.array([prod.at[i, "case_pack"]]))[0])
                    # a case that would sit beyond the excess ceiling at the receiving DC is not worth moving
                    if 0 < q <= inv["excess_weeks_of_cover"] * need[pi]:
                        saving = q * biz["cross_dc_extra_cost_usd"]
                        add(
                            i,
                            "TRANSFER",
                            "MEDIUM",
                            q,
                            DC_NAMES[poor],
                            f"{DC_NAMES[poor]} has {_weeks(cover_dc[pi])} of cover, {DC_NAMES[rich]} has "
                            f"{_weeks(cover_dc[ri])}. Move {q:,} units instead of cross-shipping orders.",
                            {
                                **base_ev,
                                f"{DC_NAMES[poor]} available": float(p.available[i, poor]),
                                f"{DC_NAMES[rich]} available": float(p.available[i, rich]),
                            },
                            f"Avoids about ${saving:,.0f} of cross-DC shipping",
                            saving,
                        )
                        acted = True

        # Amazon FBA
        f = st.fba.loc[i]
        if f["fba_enabled"] and f["recommended"] > 0:
            moves = [r for r in st.fba.attrs["transfers"] if r["sku_idx"] == i]
            sent = sum(r["qty"] for r in moves)
            if sent > 0:
                src = DC_NAMES[moves[0]["source"]]
                fba_daily = st.daily_fc[:, i] * st.fba_share[i]
                fba_rate = fba_daily[:28].mean()
                no_demand = fba_rate < NO_DEMAND_DAILY
                dos = (p.available[i, FBA] + p.fba_transfer[i]) / fba_rate if not no_demand else np.inf
                remaining = {DC_NAMES[r["source"]]: float(p.available[i, r["source"]] - r["qty"]) for r in moves}
                fba_so = stockout_day(
                    project(
                        np.array([p.available[i, FBA] + p.fba_transfer[i] + p.fba_inbound[i]]),
                        fba_daily[:, None],
                        np.zeros((horizon, 1)),
                        horizon,
                    )
                )[0]
                short_b = expected_shortfall(_fba_samples(st, i), f["fba_position"])
                short_a = expected_shortfall(_fba_samples(st, i), f["fba_position"] + sent)
                small = (short_b - short_a) < 1.0
                unit_fba = prod.at[i, "contribution_fba"]
                if no_demand:
                    why = (
                        f"No Amazon demand is forecast; the target of {f['fba_target']:,.0f} is the forecast-error "
                        f"allowance alone. Review before sending {sent:,} from {src}."
                    )
                else:
                    why = (
                        f"Amazon has {dos:.0f} days of supply; replenishment takes up to "
                        f"{st.fba.attrs['transit_p90']:.0f} days (pick + transit). Send {sent:,} from {src}."
                    )
                    if small:
                        why += " It avoids less than one unit of expected shortfall."
                if unit_fba < 0:
                    why += f" Each Amazon sale loses ${-unit_fba:,.2f}: review the listing."
                if no_demand or small or unit_fba < 0:
                    sev = "LOW"
                else:
                    sev = "HIGH" if dos < st.fba.attrs["transit_p90"] else "MEDIUM"
                add(
                    i,
                    "SEND_TO_FBA",
                    sev,
                    sent,
                    "AMAZON_FBA",
                    why,
                    {
                        "FBA available": float(p.available[i, FBA]),
                        "FBA inbound": float(p.fba_inbound[i]),
                        "FBA reserved": float(p.fba_reserved[i]),
                        "FBA transfer": float(p.fba_transfer[i]),
                        "FBA days of supply": "no forecast demand" if no_demand else dos,
                        "FBA target": f["fba_target"],
                        "FBA position": f["fba_position"],
                        "Amazon share of demand": st.fba_share[i],
                        "Source": src,
                        **{f"{k} after transfer": v for k, v in remaining.items()},
                    },
                    f"Expected Amazon units short over the replenishment window: {short_b:,.1f} -> {short_a:,.1f}",
                    (short_b - short_a) * prod.at[i, "contribution_fba"],
                    stockout=day(fba_so) if fba_so >= 0 else pd.NaT,
                )
                acted = True

        # excess
        if no_fc[i] or cover_now[i] > inv["excess_weeks_of_cover"]:
            excess_units = stock_now[i] + on_order[i] - inv["excess_weeks_of_cover"] * weekly13[i]
            val = excess_units * cost[i]
            if val >= rc["excess_value_min_usd"] and qty == 0:
                if no_fc[i]:
                    lead = (
                        f"No demand is forecast for the next 13 weeks: ${val:,.0f} of stock and open orders "
                        "has no expected sale"
                    )
                else:
                    cover_txt = f"{cover_now[i]:.0f} weeks" if cover_now[i] <= 104 else "More than two years"
                    lead = (
                        f"{cover_txt} of cover against a {inv['excess_weeks_of_cover']}-week ceiling: "
                        f"${val:,.0f} tied up beyond it"
                    )
                add(
                    i,
                    "EXCESS",
                    "LOW",
                    int(excess_units),
                    "NETWORK",
                    lead + "." + (" Liquidation announced." if discontinued[i] else ""),
                    {
                        **base_ev,
                        "Weeks of cover": "no forecast demand" if no_fc[i] else cover_now[i],
                        "Excess units": excess_units,
                        "Excess value": val,
                    },
                    f"Carrying cost of the excess: about ${val * biz['holding_cost_annual_pct']:,.0f} a year",
                    -val * biz["holding_cost_annual_pct"],
                )
                acted = True

        # censored history (information for the planner)
        if imputed_share[i] > 0.20:
            add(
                i,
                "STOCKOUT_CENSORED",
                "INFO",
                0,
                "NETWORK",
                f"{imputed_share[i]:.0%} of the last 8 weeks' demand was reconstructed: stockouts held sales "
                f"down ({h.observed[-8:, i].sum():,.0f} sold vs {h.Y[-8:, i].sum():,.0f} estimated demand).",
                {
                    "Observed sales, 8 weeks": float(h.observed[-8:, i].sum()),
                    "Reconstructed demand, 8 weeks": float(h.Y[-8:, i].sum()),
                    "Imputed share": float(imputed_share[i]),
                },
                "The forecast uses reconstructed demand, not raw sales",
                0.0,
            )

        if not acted and imputed_share[i] <= 0.20:
            add(
                i,
                "NO_ACTION",
                "INFO",
                0,
                "NETWORK",
                "Position covers demand over lead time + review at the service target.",
                base_ev,
                effect,
                0.0,
            )

    out = pd.DataFrame(rows)
    out["priority_score"] = out["severity"].map(SEVERITY_RANK) * 1e6 + out["economic_impact"].abs().clip(upper=9.99e5)
    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    out.insert(0, "recommendation_id", [f"REC-{plan_date:%Y%m%d}-{k + 1:04d}" for k in range(len(out))])
    out["priority"] = np.arange(1, len(out) + 1)
    sku_frame = pd.DataFrame(
        {
            "sku_idx": np.arange(n),
            "segment": fs.segments,
            "champion_model": [fs.models[c].name for c in fs.selection.champion],
            "selection_reason": fs.selection.reason,
            "confidence": conf,
            "confidence_score": conf_score,
            "lt_p50": lt_p50,
            "lt_p90": lt_p90,
            "ltd_mean": mean_ltd,
            "ltd_p50": st.ltd.quantiles[0.5],
            "ltd_p80": st.ltd.quantiles[0.8],
            "ltd_p90": st.ltd.quantiles[0.9],
            "ltd_p95": st.ltd.quantiles[0.95],
            "safety_stock": ss,
            "order_up_to": st.target,
            "reorder_point": st.target,
            "inventory_position": st.position,
            "stock_now": stock_now,
            "on_order": on_order,
            "on_hand": on_hand,
            # one excess definition for the KPIs and the inventory pages: physical stock beyond the ceiling
            "excess_value": np.maximum(on_hand - inv["excess_weeks_of_cover"] * weekly13, 0) * cost,
            "recommended_quantity": st.order_qty,
            "raw_requirement": st.raw_requirement,
            "stockout_day": so,
            "stockout_day_with_order": so_new,
            "weekly_demand": weekly13,
            "weeks_of_cover": cover_now,
            "weeks_of_cover_after": cover_after,
            "service_before": svc_before,
            "service_after": svc_after,
            "shortfall_before": short_before,
            "shortfall_after": short_after,
            "protected_units": protected_units,
            "demand_13w": demand_13w,
            "lost_13w_now": lost_13w_now,
            "lost_13w_with_plan": lost_13w_plan,
            "imputed_share_8w": imputed_share,
            "history_weeks": hist_weeks,
            "fba_share": st.fba_share,
            "low_margin": low_margin,
            "discontinued": discontinued,
        }
    )
    return {
        "recommendations": out,
        "sku_plan": sku_frame,
        "path": path,
        "path_with_order": path_new,
        "receipts": rec,
        "receipts_with_order": rec_new,
        "plan_date": plan_date,
    }


def _fba_samples(st, i):
    """Approximate demand samples for Amazon over its replenishment window: the network's
    lead-time-demand samples scaled by Amazon's share and by the ratio of the two windows.
    Used only to size the economic effect of a transfer, not to decide it."""
    v, w = st.ltd.samples[i]
    share = st.fba_share[i]
    days = st.fba.attrs["transit_p90"] + 7
    lt_days = st.sku_lead_time[i].quantile(0.5) + 7
    return v * share * min(1.0, days / max(lt_days, 1)), w
