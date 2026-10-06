"""The simulated business around the real demand patterns.

Real: which SKU sold, when, how many units, to which customer (UCI Online Retail II).
Simulated here: the fictional U.S. importer's locations, product economics, suppliers, lead
times, events and FBA behaviour. Every draw is seeded; supplier luck is a function of
(seed, SKU or supplier, order week), so two policies placing the same order face the same outcome.

The `baseline` array (true demand per day, SKU and location) is the reference the evaluation layer
scores against. It is handed to the engine, never to a planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qstats_planner.utils.calendar import TradingCalendar
from qstats_planner.utils.rng import stable_uniform, stream

LOCATIONS = ("EAST_DC", "WEST_DC", "AMAZON_FBA")
EAST, WEST, FBA = 0, 1, 2
DCS = (EAST, WEST)


@dataclass
class Environment:
    seed: int
    days: pd.DatetimeIndex  # every calendar day of the simulation
    trading: np.ndarray  # (n_days,) bool
    calendar: TradingCalendar
    products: pd.DataFrame  # index = position in the arrays
    suppliers: pd.DataFrame
    baseline: np.ndarray  # (n_days, n_sku, n_loc) HIDDEN reference demand
    pre_uplift: np.ndarray  # baseline before simulated event uplift (HIDDEN)
    launch_day: np.ndarray  # first day the SKU can sell (index into days)
    end_day: np.ndarray  # last day of its life (index), inclusive
    launch_stock: np.ndarray  # (n_sku,) units delivered at launch for post-warm-up launches
    events: pd.DataFrame  # PROMOTION / LIQUIDATION calendar (simulated)
    lead_time_history: pd.DataFrame  # receipts before the simulation starts
    unknown_mask: np.ndarray  # (n_days, n_sku, n_loc) inventory feed missing
    fba_unavailable_share: np.ndarray  # (n_days, n_sku) share of FBA stock not sellable
    fba_transit: np.ndarray  # (n_weeks, n_sku) transit days for a transfer shipped that week
    _lt: dict = field(default_factory=dict)
    scenario: str = "base"

    @property
    def n_sku(self) -> int:
        return len(self.products)

    @property
    def n_days(self) -> int:
        return len(self.days)

    def week_of(self, day: int) -> int:
        return day // 7

    def lead_time(self, sku: int, day: int) -> tuple[int, bool]:
        """(lead time in days, cancelled) for an order of `sku` placed at the end of `day`."""
        w = min(self.week_of(day), self._lt["z_idio"].shape[0] - 1)
        s = int(self.products.at[sku, "supplier_idx"])
        sup = self.suppliers.iloc[s]
        if self.scenario == "null":  # sensitivity world: suppliers deliver exactly on the quote
            return int(sup["quoted_lead_time_days"]), False
        base = sup["base_median_days"] * np.exp(
            self._lt["shock"][w, s] * sup["shock_sd"] + self._lt["z_idio"][w, sku] * sup["idio_sd"]
        )
        delay = self._lt["delay_days"][w, sku] * sup["delay_mean_days"] if self._lt["delay_u"][w, sku] < sup["delay_p"] else 0.0
        extra = self._lt["disruption"][w, s]
        return int(max(7, round(base + delay + extra))), bool(self._lt["cancel_u"][w, sku] < self._lt["cancel_p"])


# --------------------------------------------------------------------------- product master


def categorise(description: str, categories: dict[str, list[str]], default: str) -> str:
    text = str(description).upper()
    for name, words in categories.items():
        if any(w in text for w in words):
            return name
    return default


def _case_pack(quantities: pd.Series, candidates: list[int]) -> int:
    """The most common order quantity among plausible case sizes (the source sells in packs)."""
    q = quantities[quantities.isin(candidates)]
    return int(q.mode().iloc[0]) if len(q) else 12


def build_products(
    pop: pd.DataFrame, lines: pd.DataFrame, sel_end: pd.Timestamp, n_suppliers: int, cfg: dict, seed: int
) -> pd.DataFrame:
    biz, prod = cfg["business"], cfg["products"]
    rng = stream(seed, "products")
    hist = lines[lines["date"] <= sel_end]
    rows = []
    cats = list(biz["categories"]) + [biz["default_category"]]
    # Each category is served by one or two suppliers.
    cat_suppliers = {c: sorted({i % n_suppliers, (i * 3 + 1) % n_suppliers}) for i, c in enumerate(cats)}
    for sku, desc in pop[["sku", "description"]].itertuples(index=False):
        h = hist[hist["sku"] == sku]
        src = lines[lines["sku"] == sku] if h.empty else h
        ref = float(src["unit_price"].median()) * biz["usd_per_source_price"]
        price = max(2.99, np.floor(ref * rng.uniform(*biz["retail_markup"])) + 0.99)
        cost = round(price * rng.uniform(*biz["landed_cost_ratio"]), 2)
        cat = categorise(desc, biz["categories"], biz["default_category"])
        sup_choices = cat_suppliers[cat]
        supplier_idx = sup_choices[int(rng.integers(len(sup_choices)))]
        pack = _case_pack(src["units"], prod["case_pack_candidates"])
        weeks_alive = max(1.0, (sel_end - h["date"].min()).days / 7) if len(h) else 1.0
        weekly = h["units"].sum() / weeks_alive if len(h) else 0.0
        moq_cases = max(2, int(np.ceil(rng.uniform(*prod["moq_weeks_of_demand"]) * weekly / pack))) if weekly else 4
        rows.append(
            {
                "sku": sku,
                "description": desc,
                "category": cat,
                "supplier_idx": supplier_idx,
                "selling_price": price,
                "unit_cost": cost,
                "case_pack": pack,
                "moq": moq_cases * pack,
                "cube_per_case": round(rng.uniform(*prod["cube_per_case_m3"]), 4),
                "advertising_pct": round(rng.uniform(*biz["advertising_cost_pct"]), 3),
                "_revenue": float((h["units"] * h["unit_price"]).sum()),
            }
        )
    p = pd.DataFrame(rows)
    rank = p["_revenue"].rank(ascending=False, pct=True)
    p["abc_class"] = np.where(rank <= 0.2, "A", np.where(rank <= 0.5, "B", "C"))
    p["target_service_level"] = p["abc_class"].map(prod["service_level_by_class"])
    u = rng.uniform(size=len(p)) * np.where(p["abc_class"] == "C", 1.4, 1.0)
    p["fba_enabled"] = u < biz["fba_enabled_share"]
    p["preferred_source_dc"] = np.where(rng.uniform(size=len(p)) < 0.7, "EAST_DC", "WEST_DC")
    fc = biz["fulfillment_cost_pct"]
    fx = biz["fulfillment_cost_fixed"]
    p["fulfillment_cost_dc"] = (p["selling_price"] * fc["DC"] + fx["DC"]).round(2)
    p["fulfillment_cost_fba"] = (p["selling_price"] * fc["FBA"] + fx["FBA"]).round(2)
    p["advertising_cost"] = (p["selling_price"] * p["advertising_pct"]).round(2)
    p["contribution_dc"] = p["selling_price"] - p["unit_cost"] - p["fulfillment_cost_dc"] - p["advertising_cost"]
    p["contribution_fba"] = p["selling_price"] - p["unit_cost"] - p["fulfillment_cost_fba"] - p["advertising_cost"]
    share = np.where(p["fba_enabled"], biz["fba_channel_share"], 0.0)
    p["contribution_margin"] = ((1 - share) * p["contribution_dc"] + share * p["contribution_fba"]).round(2)
    p["contribution_margin_pct"] = (p["contribution_margin"] / p["selling_price"]).round(3)
    return p.drop(columns="_revenue")


def build_suppliers(cfg: dict, seed: int, quote_quantile: float = 0.5) -> pd.DataFrame:
    s = cfg["suppliers"]
    rng = stream(seed, "suppliers")
    rows = []
    for i in range(s["count"]):
        rows.append(
            {
                "supplier_id": f"SUP-{i + 1:02d}",
                "supplier_name": f"Supplier {i + 1:02d}",
                "country": s["countries"][i % len(s["countries"])],
                "base_median_days": float(rng.uniform(*s["lead_time_median_days"])),
                "idio_sd": float(rng.uniform(*s["lead_time_log_sd"])),
                "shock_sd": float(rng.uniform(*s["supplier_week_shock_log_sd"])),
                "delay_p": float(rng.uniform(*s["delay_probability"])),
                "delay_mean_days": float(rng.uniform(*s["delay_mean_days"])),
                "minimum_order_value": float(round(rng.uniform(*s["minimum_order_value_usd"]), -2)),
                "minimum_container_fill": s["minimum_container_fill"],
                "container_capacity_m3": s["container"]["capacity_m3"],
            }
        )
    sup = pd.DataFrame(rows)
    # Quoted lead time = median of the supplier's true distribution (disruptions excluded):
    # the quote is honest on average, and the tail is what a planner has to learn. The
    # "optimistic quotes" scenario quotes the 25th percentile instead.
    draw = stream(seed, "suppliers", "quote")
    quotes = []
    for r in sup.itertuples():
        n = 20000
        base = r.base_median_days * np.exp(draw.normal(size=n) * r.shock_sd + draw.normal(size=n) * r.idio_sd)
        late = draw.uniform(size=n) < r.delay_p
        base = base + late * draw.exponential(size=n) * r.delay_mean_days
        quotes.append(float(np.round(np.quantile(np.maximum(7, np.round(base)), quote_quantile))))
    sup["quoted_lead_time_days"] = quotes
    return sup


# --------------------------------------------------------------------------- demand by location


def assign_locations(lines: pd.DataFrame, fba_enabled: pd.Series, cfg: dict) -> np.ndarray:
    """Each customer is assigned once, by a stable hash, to a region and (for FBA SKUs) a channel.

    Keeping whole customers together keeps the real order lumpiness at every location.
    Only small-basket customers (median line at most `fba_max_median_line` units) can be Amazon
    shoppers: wholesale-sized buyers stay with the company's own channels, as they would in practice.
    """
    biz = cfg["business"]
    east_share = biz["locations"][0]["region_share"]
    key = lines["customer_key"]
    u_region = stable_uniform(key, "region")
    u_channel = stable_uniform(key, "channel")
    loc = np.where(u_region < east_share, EAST, WEST)
    basket = key.map(lines.groupby("customer_key")["units"].median()).to_numpy()
    small = basket <= biz["fba_max_median_line"]
    fba = lines["sku"].map(fba_enabled).fillna(False).to_numpy(bool) & small & (u_channel < biz["fba_channel_share"])
    return np.where(fba, FBA, loc)


def baseline_array(lines: pd.DataFrame, loc: np.ndarray, days: pd.DatetimeIndex, sku_pos: dict) -> np.ndarray:
    arr = np.zeros((len(days), len(sku_pos), len(LOCATIONS)))
    d_idx = days.get_indexer(lines["date"])
    s_idx = lines["sku"].map(sku_pos).to_numpy()
    ok = (d_idx >= 0) & ~pd.isna(s_idx)
    np.add.at(arr, (d_idx[ok], s_idx[ok].astype(int), loc[ok]), lines["units"].to_numpy()[ok])
    return arr


# --------------------------------------------------------------------------- events


def build_events(
    products: pd.DataFrame,
    launch_day: np.ndarray,
    end_day: np.ndarray,
    disappeared: np.ndarray,
    days: pd.DatetimeIndex,
    cfg: dict,
    seed: int,
) -> pd.DataFrame:
    """Simulated promotions and liquidations. Real data does not say which spikes were promotions,
    so none of these labels claims to describe the source retailer."""
    ev, sim = cfg["events"], cfg["simulation"]
    rng = stream(seed, "events")
    n = len(products)
    start_day = sim["warmup_weeks"] * 7
    rows = []
    eligible = np.where((launch_day <= start_day + 70) & (end_day >= len(days) - 1))[0]
    promo_skus = rng.choice(eligible, size=min(len(eligible), int(round(n * ev["promotion_sku_share"]))), replace=False)
    for i in promo_skus:
        k = int(rng.integers(ev["promotions_per_sku"][0], ev["promotions_per_sku"][1] + 1))
        starts = np.sort(rng.choice(np.arange(start_day + 21, len(days) - 28, 7), size=k, replace=False))
        last_end = -1
        for s in starts:
            if s <= last_end + 14:
                continue
            length = 7 * int(rng.integers(ev["promotion_weeks"][0], ev["promotion_weeks"][1] + 1))
            uplift = float(np.exp(rng.normal(ev["promotion_uplift_log_mean"], ev["promotion_uplift_log_sd"])))
            rows.append(
                {
                    "sku_idx": int(i),
                    "kind": "PROMOTION",
                    "start_day": int(s),
                    "end_day": int(s + length - 1),
                    "announce_day": int(s - 7 * ev["promotion_notice_weeks"]),
                    "uplift": uplift,
                    "price_change_pct": -0.15,
                }
            )
            last_end = s + length - 1
    dying = np.where(disappeared & (end_day > start_day + 8 * 7))[0]
    dying = np.setdiff1d(dying, promo_skus)
    liq = rng.choice(dying, size=min(len(dying), int(round(n * ev["liquidation_sku_share"]))), replace=False)
    for i in liq:
        e = int(end_day[i])
        s = e - 7 * ev["liquidation_weeks"] + 1
        rows.append(
            {
                "sku_idx": int(i),
                "kind": "LIQUIDATION",
                "start_day": s,
                "end_day": e,
                "announce_day": s,
                "uplift": float(ev["liquidation_uplift"]),
                "price_change_pct": -0.30,
            }
        )
    out = pd.DataFrame(rows, columns=["sku_idx", "kind", "start_day", "end_day", "announce_day", "uplift", "price_change_pct"])
    if not out.empty:
        out["sku"] = products.loc[out["sku_idx"], "sku"].to_numpy()
        out["start_date"] = days[out["start_day"].to_numpy()]
        out["end_date"] = days[out["end_day"].to_numpy()]
    return out


def apply_uplift(baseline: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    out = baseline.copy()
    for e in events.itertuples():
        out[e.start_day : e.end_day + 1, e.sku_idx, :] *= e.uplift
    return np.round(out)


# --------------------------------------------------------------------------- build


def build_environment(
    pop: pd.DataFrame,
    lines: pd.DataFrame,
    descriptions: pd.Series,
    trading_days: pd.DatetimeIndex,
    cfg: dict,
    seed: int,
    population: str = "demo",
    supply_seed: int | None = None,
    scenario: str = "base",
) -> Environment:
    """`seed` fixes the business (products, economics, suppliers, events). `supply_seed` (default:
    the same) drives supplier luck, FBA behaviour and feed gaps; varying it alone gives replicate
    worlds with identical products and identical demand."""
    ss = seed if supply_seed is None else supply_seed
    sim, sup_cfg = cfg["simulation"], cfg["suppliers"]
    data_start = trading_days.min()
    data_end = trading_days.max()
    # Week 1 starts on the Monday of the first trading day; day 0 is that Monday. The simulation
    # ends on the last complete week, so every weekly plan sees whole weeks.
    origin = data_start - pd.Timedelta(days=int(data_start.dayofweek))
    last_sunday = data_end - pd.Timedelta(days=(int(data_end.dayofweek) + 1) % 7)
    days = pd.date_range(origin, last_sunday, freq="D")
    calendar = TradingCalendar(trading_days, data_end, cfg)
    trading = calendar.is_trading(days)

    pop = pop[pop["population"] == population].reset_index(drop=True)
    sku_pos = {s: i for i, s in enumerate(pop["sku"])}
    plines = lines[lines["sku"].isin(sku_pos)]
    sel_end = origin + pd.Timedelta(weeks=cfg["population"]["selection_weeks"]) - pd.Timedelta(days=1)
    suppliers = build_suppliers(cfg, seed, 0.25 if scenario == "optimistic_quotes" else 0.5)
    pop = pop.assign(description=pop["sku"].map(descriptions).fillna("ITEM " + pop["sku"]))
    products = build_products(pop, plines, sel_end, len(suppliers), cfg, seed)
    products["supplier_id"] = suppliers.loc[products["supplier_idx"], "supplier_id"].to_numpy()

    loc = assign_locations(plines, products.set_index("sku")["fba_enabled"], cfg)
    raw = baseline_array(plines, loc, days, sku_pos)

    launch_day = days.get_indexer(pd.DatetimeIndex(pop["first_seen_date"]))
    last_seen = days.get_indexer(pd.DatetimeIndex(pop["last_seen_date"]))
    disappeared = pop["disappeared"].to_numpy(bool)
    end_day = np.where(disappeared, last_seen, len(days) - 1)

    events = build_events(products, launch_day, end_day, disappeared, days, cfg, seed)
    if scenario == "null":
        events = events.iloc[0:0]
    baseline = apply_uplift(raw, events)

    # Launch stock for SKUs launched after the warm-up, sized from the first eight weeks of
    # comparable launches in the full catalogue before the fork (identical in every world).
    warm_end = sim["warmup_weeks"] * 7
    first8 = lines.merge(lines.groupby("sku")["date"].min().rename("first").reset_index(), on="sku")
    first8 = first8[
        (first8["date"] < first8["first"] + pd.Timedelta(weeks=8)) & (first8["first"] < sel_end) & ~first8["sku"].isin(sku_pos)
    ]
    launch_units = first8.groupby("sku")["units"].sum()
    # twelve weeks of a typical launch's first-eight-week rate
    typical = 1.5 * float(launch_units.median()) if len(launch_units) else 150.0
    packs = products["case_pack"].to_numpy()
    launch_stock = np.where(launch_day > warm_end, np.ceil(typical / packs) * packs, 0.0)

    n_weeks = len(days) // 7 + 30
    rng = stream(ss, "lead_times")
    shock = rng.normal(size=(n_weeks, len(suppliers)))
    disruption = np.zeros((n_weeks, len(suppliers)))
    for s in range(len(suppliers)):
        k = rng.poisson(sup_cfg["disruptions_per_supplier"])
        for _ in range(k):
            w0 = int(rng.integers(sim["warmup_weeks"], len(days) // 7))
            disruption[w0 : w0 + 3, s] = rng.uniform(*sup_cfg["disruption_extra_days"])
    lt = {
        "shock": shock,
        "z_idio": stream(ss, "lead_times", "idio").normal(size=(n_weeks, len(products))),
        "delay_u": stream(ss, "lead_times", "delay_u").uniform(size=(n_weeks, len(products))),
        "delay_days": stream(ss, "lead_times", "delay_d").exponential(size=(n_weeks, len(products))),
        "cancel_u": stream(ss, "lead_times", "cancel").uniform(size=(n_weeks, len(products))),
        "cancel_p": sup_cfg["cancel_probability"],
        "disruption": disruption,
    }

    # Receipts before the simulation starts: the lead-time history a planner inherits.
    hist_rows = []
    hrng = stream(ss, "lead_times", "history")
    for r in suppliers.itertuples():
        for k in range(sup_cfg["history_pos_per_supplier"]):
            base = r.base_median_days * np.exp(hrng.normal() * r.shock_sd + hrng.normal() * r.idio_sd)
            if hrng.uniform() < r.delay_p:
                base += hrng.exponential() * r.delay_mean_days
            ltd = int(max(7, round(base))) if scenario != "null" else int(r.quoted_lead_time_days)
            order = origin - pd.Timedelta(days=int(ltd + hrng.integers(0, 300)))
            hist_rows.append(
                {
                    "po_id": f"HIST-{r.supplier_id}-{k:02d}",
                    "supplier_id": r.supplier_id,
                    "order_date": order,
                    "receipt_date": order + pd.Timedelta(days=ltd),
                    "lead_time_days": float(ltd),
                }
            )
    lead_hist = pd.DataFrame(hist_rows)

    urng = stream(ss, "unknown_availability")
    unknown = np.zeros_like(baseline, dtype=bool)
    for i in range(len(products)):
        for l_ in range(len(LOCATIONS)):
            for _ in range(urng.poisson(sim["unknown_availability_blocks_per_sku"] / len(LOCATIONS))):
                s0 = int(urng.integers(warm_end, len(days) - 20))
                unknown[s0 : s0 + int(urng.integers(*sim["unknown_availability_block_days"])), i, l_] = True

    frng = stream(ss, "fba")
    lo, hi = sim["fba_unavailable_share"]
    walk = np.cumsum(frng.normal(scale=0.15, size=(len(days), len(products))), axis=0)
    fba_share = lo + (hi - lo) * (0.5 + 0.5 * np.tanh(walk / 3))
    fba_transit = frng.integers(sim["fba_transit_days"][0], sim["fba_transit_days"][1] + 1, size=(n_weeks, len(products)))

    env = Environment(
        seed=ss,
        days=days,
        trading=trading,
        calendar=calendar,
        products=products,
        suppliers=suppliers,
        baseline=baseline,
        pre_uplift=raw,
        launch_day=launch_day,
        end_day=end_day,
        launch_stock=launch_stock,
        events=events,
        lead_time_history=lead_hist,
        unknown_mask=unknown,
        fba_unavailable_share=fba_share,
        fba_transit=fba_transit,
        _lt=lt,
        scenario=scenario,
    )
    return env
