"""Legacy vs QStats: outcomes of the same simulated business, scored on the same SKU-days.

Evaluation layer: reads hidden baseline demand and lost units, which no planner ever sees.

Scoring window: from the fork plus the longest quoted lead time (orders QStats placed have had
time to arrive) to the end of the data. Everything is computed per SKU first, so the totals can
be resampled by SKU (paired bootstrap) and so both worlds are always scored on identical SKU-days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.simulation.environment import FBA

PER_SKU_COLUMNS = [
    "demand", "sold", "lost", "lost_contribution", "inventory_value_days", "inventory_units_days",
    "cogs", "stockout_days", "active_days", "excess_value", "purchase_value", "purchase_units",
    "in_transit_value_days", "ending_position_value", "cross_ship_units", "fc_abs_error",
    "fc_error", "fc_actual", "days",
]


def scoring_window(env, cfg) -> tuple[int, int]:
    fork = (cfg["simulation"]["fork_week"] - 1) * 7
    return fork + int(env.suppliers["quoted_lead_time_days"].max()), env.n_days


def per_sku(env, eng, policy_history: list[dict], cfg: dict, start: int, end: int) -> pd.DataFrame:
    p = env.products
    cost = p["unit_cost"].to_numpy()
    days = end - start
    B, S, L = env.baseline[start:end], eng.sales[start:end], eng.lost[start:end]
    contrib = p[["contribution_dc", "contribution_dc", "contribution_fba"]].to_numpy(float)
    on_hand = np.nan_to_num(eng.closing_on_hand[start:end]).sum(axis=2) + eng.inbound_hist[start:end]
    # stockout days: channel-days (direct = both DCs pooled, Amazon) in the SKU's life, on trading days
    oa = np.nan_to_num(eng.opening_available[start:end])
    direct_out = (oa[:, :, 0] + oa[:, :, 1]) <= 0
    fba_out = oa[:, :, FBA] <= 0
    life = np.zeros(B.shape[:2], bool)
    for i in range(env.n_sku):
        life[max(env.launch_day[i] - start, 0): max(env.end_day[i] - start + 1, 0), i] = True
    act = life & env.trading[start:end, None]
    fba_on = p["fba_enabled"].to_numpy(bool)
    stockout = (direct_out & act).sum(0) + (fba_out & act & fba_on[None]).sum(0)
    active = act.sum(0) + (act & fba_on[None]).sum(0)
    # excess at the end: owned stock beyond 26 weeks of the last 13 weeks' true demand rate
    rate = B[-91:].sum(axis=(0, 2)) / 91.0
    end_stock = on_hand[-1]
    excess_units = np.maximum(end_stock - rate * 26 * 7, 0)
    # purchases decided from the fork on
    fork = (cfg["simulation"]["fork_week"] - 1) * 7
    pu = np.zeros(env.n_sku)
    transit_val = np.zeros(env.n_sku)
    open_end = np.zeros(env.n_sku)
    for po in eng.pos:
        q = po["qty"][0] + po["qty"][1]
        if po["order_day"] >= fork:
            pu[po["sku_idx"]] += q
        if po["cancelled"]:
            continue
        ship = po["order_day"] + po["production_days"]
        lo, hi = max(ship, start), min(po["arrival_day"], end)
        if hi > lo:
            transit_val[po["sku_idx"]] += q * (hi - lo)
        if po["order_day"] < end <= po["arrival_day"]:
            open_end[po["sku_idx"]] += q
    # forecasts: one-week-ahead network units vs true weekly demand, weeks inside the window
    fae, fe, fa = np.zeros(env.n_sku), np.zeros(env.n_sku), np.zeros(env.n_sku)
    for h in policy_history:
        t = h["t"]
        if t + 1 >= start and t + 8 <= end:
            actual = env.baseline[t + 1: t + 8].sum(axis=(0, 2))
            f = np.nan_to_num(h["weekly_fc_1"])
            fae += np.abs(f - actual)
            fe += f - actual
            fa += actual
    cross = eng.cross_ship[start:end].sum(axis=(0, 2))
    out = pd.DataFrame({
        "demand": B.sum(axis=(0, 2)), "sold": S.sum(axis=(0, 2)), "lost": L.sum(axis=(0, 2)),
        "lost_contribution": (L * contrib[None]).sum(axis=(0, 2)),
        "inventory_value_days": on_hand.sum(0) * cost, "inventory_units_days": on_hand.sum(0),
        "cogs": S.sum(axis=(0, 2)) * cost, "stockout_days": stockout, "active_days": active,
        "excess_value": excess_units * cost, "purchase_value": pu * cost, "purchase_units": pu,
        "in_transit_value_days": transit_val * cost,
        "ending_position_value": (end_stock + open_end) * cost,
        "cross_ship_units": cross, "fc_abs_error": fae, "fc_error": fe, "fc_actual": fa,
        "days": np.full(env.n_sku, days),
    })
    out.insert(0, "sku", p["sku"].to_numpy())
    return out


def summarise(df: pd.DataFrame, cfg: dict, mask: np.ndarray | None = None) -> dict:
    d = df if mask is None else df[mask]
    days = float(d["days"].iloc[0]) if len(d) else 1.0
    inv = d["inventory_value_days"].sum() / days
    wc = inv + d["in_transit_value_days"].sum() / days
    cogs = d["cogs"].sum()
    return {
        "skus": len(d),
        "fill_rate": d["sold"].sum() / max(d["demand"].sum(), 1e-9),
        "in_stock_rate": 1 - d["stockout_days"].sum() / max(d["active_days"].sum(), 1),
        "stockout_days": int(d["stockout_days"].sum()),
        "lost_units": d["lost"].sum(),
        "lost_contribution": d["lost_contribution"].sum(),
        "average_inventory_units": d["inventory_units_days"].sum() / days,
        "average_inventory_value": inv,
        "inventory_turns": (cogs * 365 / days) / inv if inv > 0 else np.nan,
        "excess_inventory_value": d["excess_value"].sum(),
        "purchase_value": d["purchase_value"].sum(),
        "working_capital": wc,
        "holding_cost": inv * cfg["business"]["holding_cost_annual_pct"] * days / 365,
        "ending_position_value": d["ending_position_value"].sum(),
        "cross_dc_cost": d["cross_ship_units"].sum() * cfg["business"]["cross_dc_extra_cost_usd"],
        "wape": d["fc_abs_error"].sum() / max(d["fc_actual"].sum(), 1e-9),
        "forecast_bias": d["fc_error"].sum() / max(d["fc_actual"].sum(), 1e-9),
        "demand_units": d["demand"].sum(),
    }


# --------------------------------------------------------------------------- frontier


def interpolate_inventory(points: pd.DataFrame, fill: float) -> float:
    """Inventory a policy family needs for a given fill rate, by linear interpolation along its
    frontier (no extrapolation: NaN outside the range it was run at)."""
    pts = points.sort_values("fill_rate")
    x, y = pts["fill_rate"].to_numpy(), pts["average_inventory_value"].to_numpy()
    if len(x) == 0 or fill < x.min() - 1e-12 or fill > x.max() + 1e-12:
        return np.nan
    return float(np.interp(fill, x, y))


def matched_comparison(summaries: dict[str, dict], legacy_ref: str, legacy_family: list[str], qstats_family: list[str]) -> dict:
    ref = summaries[legacy_ref]
    q_pts = pd.DataFrame([summaries[k] for k in qstats_family])
    l_pts = pd.DataFrame([summaries[k] for k in legacy_family])
    q_inv = interpolate_inventory(q_pts, ref["fill_rate"])
    # and the other way: Legacy's inventory at QStats's headline fill rate
    head = summaries[qstats_family[0]]
    l_inv = interpolate_inventory(l_pts, head["fill_rate"])
    return {
        "legacy_fill": ref["fill_rate"], "legacy_inventory": ref["average_inventory_value"],
        "qstats_inventory_at_legacy_fill": q_inv,
        "inventory_saving_pct": 1 - q_inv / ref["average_inventory_value"] if np.isfinite(q_inv) else np.nan,
        "qstats_fill": head["fill_rate"], "qstats_inventory": head["average_inventory_value"],
        "legacy_inventory_at_qstats_fill": l_inv,
        "legacy_extra_inventory_pct": l_inv / head["average_inventory_value"] - 1 if np.isfinite(l_inv) else np.nan,
    }


def bootstrap(per_variant: dict[str, pd.DataFrame], cfg: dict, legacy_ref: str, legacy_family: list[str],
              qstats_family: list[str], n_boot: int = 1000, seed: int = 20261006) -> pd.DataFrame:
    """Paired bootstrap over SKUs: every resample uses the same SKUs in every variant."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_variant.values())))
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sums = {k: summarise(v.iloc[idx], cfg) for k, v in per_variant.items() if k in set(legacy_family) | set(qstats_family)}
        m = matched_comparison(sums, legacy_ref, legacy_family, qstats_family)
        h, r = sums[qstats_family[0]], sums[legacy_ref]
        rows.append({**m, "fill_diff": h["fill_rate"] - r["fill_rate"],
                     "inventory_diff": h["average_inventory_value"] - r["average_inventory_value"],
                     "lost_contribution_diff": h["lost_contribution"] - r["lost_contribution"]})
    return pd.DataFrame(rows)


def efficiency_vs_legacy(summaries: dict[str, dict], legacy_family: list[str]) -> dict[str, float]:
    """For every variant: inventory the Legacy frontier needs at that variant's fill rate, relative
    to the variant's own inventory, minus one (positive = the variant holds less stock for the
    same service). Controls for the operating point, so ablation arms can be compared."""
    l_pts = pd.DataFrame([summaries[k] for k in legacy_family])
    out = {}
    for k, s in summaries.items():
        need = interpolate_inventory(l_pts, s["fill_rate"])
        out[k] = need / s["average_inventory_value"] - 1 if np.isfinite(need) else np.nan
    return out
