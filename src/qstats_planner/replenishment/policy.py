"""The QStats planning cycle, as of a plan date.

    sales + snapshots ─▶ censored-demand reconstruction ─▶ weekly demand history
        ─▶ segment + champion forecast (rolling-origin selection) ─▶ daily forecast path
        ─▶ supplier lead-time distribution (Kaplan-Meier) ─▶ demand over lead time + review
        ─▶ order-up-to level S = Q_alpha ─▶ inventory position ─▶ PO quantity (MOQ, case pack)
        ─▶ DC split ─▶ Amazon FBA replenishment from the source DC

Ablation switches (each one off reproduces the Legacy choice for that ingredient):
    uncensor       off: sales are taken as demand
    seasonal       off: no seasonal prior in any candidate model
    probabilistic  off: quoted lead time, safety stock = N days of forecast demand
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qstats_planner.demand.reconstruction import channel_data, reconstruct
from qstats_planner.forecasting import pipeline
from qstats_planner.forecasting.pipeline import ForecastState, History
from qstats_planner.forecasting.uncertainty import recent_level
from qstats_planner.inventory import lead_time as ltmod
from qstats_planner.inventory.lead_time_demand import LTDResult, lead_time_demand
from qstats_planner.replenishment.order_quantity import round_orders, round_up_to_pack, split_by_share
from qstats_planner.simulation.environment import EAST, FBA, WEST
from qstats_planner.simulation.policies.base import dc_east_share, network_position, weekly


@dataclass
class PlanSettings:
    uncensor: bool = True
    seasonal: bool = True
    probabilistic: bool = True
    service_level: float | None = None  # None: each product's own target
    demand_multiplier: float = 1.0  # scenario: demand growth
    lead_time_multiplier: float = 1.0  # scenario
    lead_time_spread: float = 1.0  # scenario: supplier variability
    safety_days: float = 30.0  # used when probabilistic is off


@dataclass
class PlanState:
    t: int
    history: History
    recon: object
    forecast_state: ForecastState
    weekly_fc: np.ndarray  # (H, n)
    daily_fc: np.ndarray  # (D, n)
    future_days: pd.DatetimeIndex
    lead_times: dict
    sku_lead_time: list
    ltd: LTDResult
    alpha: np.ndarray
    position: np.ndarray  # network inventory position (planning-weighted)
    target: np.ndarray  # order-up-to level
    raw_requirement: np.ndarray
    order_qty: np.ndarray
    east_qty: np.ndarray
    west_qty: np.ndarray
    fba: pd.DataFrame
    fba_share: np.ndarray


def build_history(view, recon, prior, settings: PlanSettings) -> History:
    t = view.t
    sales_net = view.sales.sum(axis=2)
    if settings.uncensor:
        adj = recon.adjusted.sum(axis=2)
        imp = np.where(
            recon.imputed, recon.adjusted - np.stack([view.sales[:, :, 0] + view.sales[:, :, 1], view.sales[:, :, 2]], axis=2), 0
        ).sum(axis=2)
    else:
        adj, imp = sales_net, np.zeros_like(sales_net)
    Y, imputed_w, observed_w = weekly(adj, t), weekly(imp, t), weekly(sales_net, t)
    W = Y.shape[0]
    exposure = view.trading[: W * 7].reshape(W, 7).sum(axis=1).astype(float)
    launch_week = np.where(view.launch_day >= 0, view.launch_day // 7, W + 1)
    event_week = np.zeros((W, view.n_sku), bool)
    ev = view.events()
    for e in ev.itertuples() if len(ev) else []:
        event_week[e.start_day // 7 : min(e.end_day, t) // 7 + 1, e.sku_idx] = True
    valid = (exposure[:, None] > 0) & (np.arange(W)[:, None] >= launch_week[None, :]) & ~event_week
    pr = prior if settings.seasonal else prior.neutral()
    season = pr.weekly_from_daily(view.all_days, W)
    return History(Y, imputed_w, observed_w, valid, exposure, season, launch_week, pr.sku_group)


def run_cycle(
    view,
    cfg: dict,
    prior,
    settings: PlanSettings,
    previous: ForecastState | None = None,
    refit: bool = True,
    keep_backtest: bool = False,
) -> PlanState:
    rc, fc, inv = cfg["reconstruction"], cfg["forecasting"], cfg["inventory"]
    prod = view.products
    n = view.n_sku
    R = view.review_period_days

    # 1. reconstruction
    pr = prior if settings.seasonal else prior.neutral()
    season_days = pr.daily_factors(view.days)
    cd = channel_data(view, season_days)
    recon = reconstruct(
        cd,
        rc["planner_method"] if settings.uncensor else "no_adjustment",
        view.days.dayofweek.to_numpy(),
        two_sided=True,
        window=rc["window_trading_days"],
        min_clean=rc["min_clean_days"],
        unknown_zero_run_p=rc["unknown_zero_run_probability"],
        gamma_min_shape=rc["gamma_min_shape"],
    )
    hist = build_history(view, recon, prior, settings)

    # 2. lead times
    quoted = prod["supplier_idx"].map(view.suppliers["quoted_lead_time_days"]).to_numpy(float)
    if settings.probabilistic:
        lts = ltmod.estimate_all(view.lead_time_observations(), view.suppliers, inv["lead_time_shrink_k"])
    else:
        lts = {r.supplier_id: ltmod.point(float(r.quoted_lead_time_days)) for r in view.suppliers.itertuples()}
    if settings.lead_time_multiplier != 1.0 or settings.lead_time_spread != 1.0:
        lts = {k: v.scaled(settings.lead_time_multiplier, settings.lead_time_spread) for k, v in lts.items()}
    sku_lt = [lts[s] for s in prod["supplier_id"]]
    lt_p50 = np.array([d.quantile(0.5) for d in sku_lt])
    protection_weeks = (lt_p50 + R) / 7.0

    # 3. forecast
    if refit or previous is None:
        fs = pipeline.fit(hist, protection_weeks, cfg, settings.seasonal, previous, keep_backtest)
    else:
        fs = previous
    H = fc["forecast_horizon_weeks"]
    D_weeks = max(H, int(np.ceil((cfg["forecasting"]["planning_horizon_days"] + 30) / 7)))
    future_days = pd.date_range(view.date + pd.Timedelta(days=1), periods=D_weeks * 7, freq="D")
    trading_future = view.calendar.is_trading(future_days)
    fut_exp = trading_future.reshape(D_weeks, 7).sum(axis=1).astype(float)
    season_future_daily = pr.daily_factors(future_days)
    fut_season = season_future_daily.reshape(D_weeks, 7, n).mean(axis=1)
    weekly_fc = pipeline.forecast(hist, fs, cfg, settings.seasonal, fut_exp, fut_season, D_weeks)
    weekly_fc = weekly_fc * settings.demand_multiplier
    daily_fc = pipeline.to_daily(weekly_fc, future_days, trading_future, season_future_daily)

    # 4. demand over lead time + review, and the order-up-to level
    alpha = np.full(n, settings.service_level) if settings.service_level else prod["target_service_level"].to_numpy(float)
    level = np.nan_to_num(recent_level(hist.Y, hist.valid)[-1]) * settings.demand_multiplier
    ltd = lead_time_demand(daily_fc, sku_lt, R, fs.segments, fs.errors, alpha, level)
    if settings.probabilistic:
        target = ltd.target
    else:
        mean_daily = daily_fc[: int(quoted.max()) + R].mean(axis=0)
        target = ltd.mean + settings.safety_days * mean_daily

    # 5. position and order
    position = network_position(view, reserved_weight=inv["reserved_planning_weight"] if settings.probabilistic else 1.0)
    need = (position < target) & view.launched & ~view.discontinued()
    raw = np.where(need, target - position, 0.0)
    qty = round_orders(raw, prod["case_pack"].to_numpy(), prod["moq"].to_numpy())
    loc_sales = weekly(view.sales, view.t)
    pref_east = (prod["preferred_source_dc"] == "EAST_DC").to_numpy()
    east, west = split_by_share(qty, dc_east_share(loc_sales, pref_east), prod["case_pack"].to_numpy(), pref_east)

    # 6. Amazon FBA
    fba_share, fba = _fba_plan(view, cfg, recon, daily_fc, fs, alpha, settings, pref_east, level)
    return PlanState(
        view.t,
        hist,
        recon,
        fs,
        weekly_fc,
        daily_fc,
        future_days,
        lts,
        sku_lt,
        ltd,
        alpha,
        position,
        target,
        raw,
        qty,
        east,
        west,
        fba,
        fba_share,
    )


def _fba_plan(view, cfg, recon, daily_fc, fs, alpha, settings: PlanSettings, pref_east, level) -> tuple[np.ndarray, pd.DataFrame]:
    sim, lg = cfg["simulation"], cfg["legacy"]
    prod = view.products
    n = view.n_sku
    R = view.review_period_days
    fba_on = prod["fba_enabled"].to_numpy(bool)
    adj = (
        recon.adjusted
        if settings.uncensor
        else np.stack([view.sales[:, :, 0] + view.sales[:, :, 1], view.sales[:, :, 2]], axis=2)
    )
    recent = adj[-91:]
    tot = recent.sum(axis=(0, 2))
    share = np.where(tot > 0, recent[:, :, 1].sum(axis=0) / np.maximum(tot, 1e-9), cfg["business"]["fba_channel_share"] * 0.5)
    share = np.where(fba_on, share, 0.0)
    fba_daily = daily_fc * share[None, :]

    # transit-time distribution from completed transfers (shrunk to the configured range)
    tr = view.transfers()
    done = tr.dropna(subset=["arrive_day"])
    lo, hi = sim["fba_transit_days"]
    obs = (done["arrive_day"] - done["ship_day"]).to_numpy(float) if len(done) else np.array([])
    prior = np.linspace(lo, hi, 8)
    times = np.concatenate([obs[-200:], prior])
    vals, counts = np.unique(np.round(times + sim["transfer_pick_days"]), return_counts=True)
    transit = ltmod.LeadTimeDistribution("FBA", vals, counts / counts.sum(), len(obs), 0, float(np.median(times)))

    p = view.position
    fba_pos = (
        p.available[:, FBA]
        + p.fba_transfer
        + p.fba_inbound
        + p.dc_committed.sum(axis=1)
        + cfg["inventory"]["reserved_planning_weight"] * p.fba_reserved
    )
    if settings.probabilistic:
        ltd = lead_time_demand(fba_daily, [transit] * n, R, fs.segments, fs.errors, alpha, level * share)
        target = ltd.target
        mean = ltd.mean
    else:
        repl = sim["transfer_pick_days"] + float(np.mean(sim["fba_transit_days"])) + R
        rate = fba_daily[:28].mean(axis=0)
        mean = rate * repl
        target = np.where(
            p.on_hand[:, FBA] + p.fba_inbound < rate * (repl + lg["fba_trigger_extra_days"]),
            rate * (repl + lg["fba_cover_days"]),
            0.0,
        )
        fba_pos = p.on_hand[:, FBA] + p.fba_inbound + p.dc_committed.sum(axis=1)
    send = np.where(fba_on & view.launched, np.maximum(target - fba_pos, 0), 0)
    send = round_up_to_pack(send, prod["case_pack"].to_numpy())
    # source: preferred DC, keeping a week of its own expected direct demand; then the other DC
    direct_week = daily_fc[:7].sum(axis=0) * (1 - share)
    east_share = dc_east_share(weekly(view.sales, view.t), pref_east)
    keep = np.stack([direct_week * east_share, direct_week * (1 - east_share)], axis=1)
    avail = np.maximum(p.available[:, [EAST, WEST]] - keep, 0)
    rows = []
    for i in np.where(send > 0)[0]:
        first = EAST if pref_east[i] else WEST
        pack = int(prod.at[i, "case_pack"])
        q1 = min(send[i], int(avail[i, first] // pack * pack))
        rest = send[i] - q1
        q2 = min(rest, int(avail[i, 1 - first] // pack * pack))
        for src, q in ((first, q1), (1 - first, q2)):
            if q > 0:
                rows.append({"sku_idx": int(i), "qty": int(q), "source": int(src)})
    out = pd.DataFrame(
        {
            "sku_idx": np.arange(n),
            "fba_enabled": fba_on,
            "fba_share": share,
            "fba_position": fba_pos,
            "fba_target": np.where(fba_on, target, 0),
            "fba_mean_demand": np.where(fba_on, mean, 0),
            "recommended": send,
        }
    )
    out.attrs["transfers"] = rows
    out.attrs["transit_p90"] = transit.quantile(0.9)
    return share, out
