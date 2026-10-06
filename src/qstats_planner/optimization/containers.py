"""Container planning: turn one supplier's purchase lines into full containers.

MVP solver: a transparent greedy heuristic.
    1. Place every recommended line (whole cases).
    2. Only if the last container is below the supplier's minimum fill, top it up to `top_up_to`
       with extra cases of that supplier's SKUs, lowest weeks of cover first, skipping low-margin
       and discontinued SKUs and never taking a SKU above `max_cover_weeks` of cover. Filling a box
       for its own sake converts working capital into stock, so the top-up stops at the minimum.
    3. Report utilisation, unused capacity and purchase value per container.

`ContainerSolver` is the seam: an OR-Tools / PuLP / scipy.optimize model (maximise the value of
avoided shortfall subject to cube capacity and MOQ) can replace `GreedyContainerSolver` without
changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class ContainerLine:
    sku: str
    sku_idx: int
    recommended_units: int
    case_pack: int
    cube_per_case: float
    unit_cost: float
    weeks_of_cover_after: float  # after the recommended order
    weekly_demand: float
    eligible_top_up: bool


class ContainerSolver(Protocol):
    def solve(self, supplier_id: str, lines: list[ContainerLine], capacity_m3: float, min_fill: float) -> pd.DataFrame: ...


class GreedyContainerSolver:
    def __init__(self, top_up_to: float = 0.65, max_cover_weeks: float = 16.0):
        self.top_up_to = top_up_to
        self.max_cover_weeks = max_cover_weeks

    def solve(self, supplier_id: str, lines: list[ContainerLine], capacity_m3: float, min_fill: float) -> pd.DataFrame:
        if not lines:
            return pd.DataFrame()
        qty = {ln.sku_idx: ln.recommended_units for ln in lines}
        top = dict.fromkeys(qty, 0)
        cube = lambda: sum((qty[ln.sku_idx] + top[ln.sku_idx]) / ln.case_pack * ln.cube_per_case for ln in lines)  # noqa: E731
        total = cube()
        n_cont = max(1, int(np.ceil(total / capacity_m3 - 1e-9)))
        last_fill = (total - (n_cont - 1) * capacity_m3) / capacity_m3
        target = (n_cont - 1) * capacity_m3 + self.top_up_to * capacity_m3 if last_fill < min_fill else total
        cover = {ln.sku_idx: ln.weeks_of_cover_after for ln in lines}
        cands = [ln for ln in lines if ln.eligible_top_up and ln.weekly_demand > 0]
        while cands and total < target:
            ln = min(cands, key=lambda x: cover[x.sku_idx])
            add_cube = ln.cube_per_case
            new_cover = cover[ln.sku_idx] + ln.case_pack / ln.weekly_demand
            if total + add_cube > n_cont * capacity_m3 or new_cover > self.max_cover_weeks:
                cands.remove(ln)
                continue
            top[ln.sku_idx] += ln.case_pack
            cover[ln.sku_idx] = new_cover
            total += add_cube
        rows = []
        for ln in lines:
            units = qty[ln.sku_idx] + top[ln.sku_idx]
            if units <= 0:
                continue
            rows.append(
                {
                    "supplier_id": supplier_id,
                    "sku": ln.sku,
                    "sku_idx": ln.sku_idx,
                    "recommended_units": qty[ln.sku_idx],
                    "top_up_units": top[ln.sku_idx],
                    "units": units,
                    "cases": units // ln.case_pack,
                    "cube_m3": units / ln.case_pack * ln.cube_per_case,
                    "value": units * ln.unit_cost,
                    "weeks_of_cover_after": cover[ln.sku_idx],
                }
            )
        out = pd.DataFrame(rows)
        out["containers"] = n_cont
        util = total / (n_cont * capacity_m3)
        out["utilisation"] = util
        out["below_minimum_fill"] = util < min_fill
        return out


def plan_containers(
    recs: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    cover_after: np.ndarray,
    weekly_demand: np.ndarray,
    margin_ok: np.ndarray,
    discontinued: np.ndarray,
    solver: ContainerSolver | None = None,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """recs: one row per SKU with a purchase quantity (BUY lines)."""
    if solver is None:
        sc = (cfg or {}).get("suppliers", {})
        solver = GreedyContainerSolver(sc.get("container_top_up_to", 0.65), sc.get("container_top_up_max_cover_weeks", 16))
    lines_by_sup: dict[str, list[ContainerLine]] = {}
    buy = recs.set_index("sku_idx")["recommended_quantity"] if len(recs) else pd.Series(dtype=float)
    for i, p in products.iterrows():
        units = int(buy.get(i, 0))
        lines_by_sup.setdefault(p["supplier_id"], []).append(
            ContainerLine(
                p["sku"],
                i,
                units,
                int(p["case_pack"]),
                float(p["cube_per_case"]),
                float(p["unit_cost"]),
                float(cover_after[i]),
                float(weekly_demand[i]),
                bool(margin_ok[i] and not discontinued[i]),
            )
        )
    plans, summary = [], []
    sup = suppliers.set_index("supplier_id")
    for sid, lines in lines_by_sup.items():
        if not any(ln.recommended_units > 0 for ln in lines):
            continue
        res = solver.solve(sid, lines, float(sup.at[sid, "container_capacity_m3"]), float(sup.at[sid, "minimum_container_fill"]))
        if res.empty:
            continue
        plans.append(res)
        summary.append(
            {
                "supplier_id": sid,
                "supplier_name": sup.at[sid, "supplier_name"],
                "country": sup.at[sid, "country"],
                "containers": int(res["containers"].iloc[0]),
                "utilisation": float(res["utilisation"].iloc[0]),
                "unused_capacity_pct": float(1 - res["utilisation"].iloc[0]),
                "cube_m3": float(res["cube_m3"].sum()),
                "purchase_value": float(res["value"].sum()),
                "recommended_value": float((res["recommended_units"] * res["value"] / res["units"]).sum()),
                "top_up_value": float((res["top_up_units"] * res["value"] / res["units"]).sum()),
                "skus": int((res["units"] > 0).sum()),
                "minimum_order_value": float(sup.at[sid, "minimum_order_value"]),
                "meets_minimum_order": bool(res["value"].sum() >= sup.at[sid, "minimum_order_value"]),
                "below_minimum_fill": bool(res["below_minimum_fill"].iloc[0]),
            }
        )
    return (pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()), pd.DataFrame(summary)
