"""Run the QStats planning cycle on the business as it stands today and store the plan.

"Today" is the end of the last complete week of data (Sunday 7 Dec 2025 on the shifted calendar),
in the world the Legacy process ran: the plan is what QStats would tell this business to do now.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from qstats_planner.demand.reconstruction import STATUS_NAMES  # noqa: E402
from qstats_planner.domain import tables  # noqa: E402
from qstats_planner.economics.impact import portfolio_kpis  # noqa: E402
from qstats_planner.forecasting import pipeline  # noqa: E402
from qstats_planner.forecasting.uncertainty import recent_level  # noqa: E402
from qstats_planner.optimization.containers import plan_containers  # noqa: E402
from qstats_planner.replenishment import recommendations  # noqa: E402
from qstats_planner.replenishment.policy import PlanSettings, run_cycle  # noqa: E402
from qstats_planner.simulation import runner  # noqa: E402
from qstats_planner.utils.config import database_url, load_config, resolve  # noqa: E402


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    inp = runner.load_inputs(cfg)
    env, prior, _ = runner.build_world(inp, cfg, "demo", cfg["random_seed"])
    sim = resolve(cfg["paths"]["simulation_dir"]) / "demo"
    with open(sim / f"world_{runner.LEGACY_REF}_seed{cfg['random_seed']}.pkl", "rb") as fh:
        world = pickle.load(fh)["engine"]
    world.env = env
    view = world.view(env.n_days - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st = run_cycle(view, cfg, prior, PlanSettings(), keep_backtest=True)
        created = pd.Timestamp.now().floor("s")
        plan = recommendations.build(view, st, cfg, created)
        diag = pipeline.diagnostics(st.history, st.forecast_state)
    prod = view.products
    n = view.n_sku
    sku = prod["sku"].to_numpy()
    plan_date = plan["plan_date"]
    recs = plan["recommendations"]
    sp = plan["sku_plan"].assign(sku=sku)

    # containers from the purchase lines (BUY and CRITICAL with a quantity)
    buy = recs[(recs["action"] == "BUY") & (recs["recommended_quantity"] > 0)]
    buy = buy.drop_duplicates("sku_idx")[["sku_idx", "recommended_quantity"]]
    cont, cont_sum = plan_containers(
        buy,
        prod,
        view.suppliers,
        sp["weeks_of_cover_after"].to_numpy(),
        sp["weekly_demand"].to_numpy(),
        ~sp["low_margin"].to_numpy(),
        sp["discontinued"].to_numpy(),
        cfg=cfg,
    )

    # forecasts with intervals (single-week errors for each week ahead)
    H = cfg["forecasting"]["forecast_horizon_weeks"]
    level = np.nan_to_num(recent_level(st.history.Y, st.history.valid)[-1])
    fc_rows = []
    weeks = pd.date_range(plan_date, periods=H, freq="7D")
    for h in range(H):
        for i in range(n):
            e = st.forecast_state.errors.errors(st.forecast_state.segments[i], 1)
            q = np.quantile(np.maximum(st.weekly_fc[h, i] + level[i] * e, 0), [0.1, 0.5, 0.8, 0.9, 0.95])
            fc_rows.append((sku[i], weeks[h], h + 1, st.weekly_fc[h, i], *q))
    fcast = pd.DataFrame(fc_rows, columns=["sku", "week", "horizon", "expected", "p10", "p50", "p80", "p90", "p95"])

    # weekly history (planner side: observed, reconstructed, imputed; events and stockout days)
    W = st.history.Y.shape[0]
    wk_days = env.days[: W * 7 : 7]
    rec = st.recon
    cens = rec.imputed.reshape(-1, rec.imputed.shape[1], 2)[: W * 7].reshape(W, 7, n, 2).any(axis=3).sum(axis=1)
    status = rec.status[: W * 7].reshape(W, 7, n, 2)
    promo = ((status == 5) | (status == 6)).any(axis=(1, 3))
    bt = st.forecast_state.backtest
    champ = st.forecast_state.selection.champion
    w1 = bt.week1_forecast[champ, :, np.arange(n)].T  # made at w for week w+1
    backtest_fc = np.vstack([np.full((1, n), np.nan), w1[:-1]])  # aligned to the week forecast
    hist = pd.DataFrame(
        {
            "sku": np.tile(sku, W),
            "week": np.repeat(wk_days, n),
            "observed": st.history.observed.ravel(),
            "backtest_forecast": backtest_fc.ravel(),
            "reconstructed": st.history.Y.ravel(),
            "imputed": st.history.imputed.ravel(),
            "stockout_days": cens.ravel(),
            "event_week": promo.ravel(),
            "valid": st.history.valid.ravel(),
        }
    )
    hist = hist[hist["valid"] | (hist["observed"] > 0)]

    # daily channel-level demand observations with the planner's status and reconstruction
    T = rec.status.shape[0]
    start = world.warm_end
    idx = pd.MultiIndex.from_product([range(start, T), range(n), range(2)], names=["day", "sku_idx", "ch"])
    dch = np.stack([view.sales[:, :, 0] + view.sales[:, :, 1], view.sales[:, :, 2]], axis=2)
    dob = pd.DataFrame(
        {"units": dch[start:].ravel(), "reconstructed": rec.adjusted[start:].ravel(), "status": rec.status[start:].ravel()},
        index=idx,
    ).reset_index()
    dob = dob[dob["status"] > 0]
    dob["date"] = env.days[dob["day"].to_numpy()].date
    dob["sku"] = sku[dob["sku_idx"].to_numpy()]
    dob["channel"] = np.where(dob["ch"] == 0, "DIRECT", "AMAZON")
    dob["availability_status"] = dob["status"].map(STATUS_NAMES)

    # projection, 180 days
    D = plan["path"].shape[0]
    pdays = pd.date_range(plan_date, periods=D, freq="D")
    proj = pd.DataFrame(
        {
            "sku": np.tile(sku, D),
            "date": np.repeat(pdays, n),
            "projected": plan["path"].ravel(),
            "projected_with_order": plan["path_with_order"].ravel(),
            "receipts": plan["receipts"].ravel(),
            "receipts_with_order": plan["receipts_with_order"].ravel(),
            "expected_demand": st.daily_fc[:D].ravel(),
        }
    )
    proj["safety_stock"] = np.tile(sp["safety_stock"].to_numpy(), D)

    # FBA
    p = view.position
    fba = st.fba.assign(
        sku=sku,
        fba_available=p.available[:, 2],
        fba_inbound=p.fba_inbound,
        fba_reserved=p.fba_reserved,
        fba_transfer=p.fba_transfer,
        east_available=p.available[:, 0],
        west_available=p.available[:, 1],
    )
    fba["fba_daily_demand"] = st.daily_fc[:28].mean(axis=0) * st.fba_share
    fba["fba_days_of_supply"] = (fba["fba_available"] + fba["fba_transfer"]) / fba["fba_daily_demand"].replace(0, np.nan)
    tr = pd.DataFrame(st.fba.attrs["transfers"], columns=["sku_idx", "qty", "source"])
    fba["send_qty"] = fba["sku_idx"].map(tr.groupby("sku_idx")["qty"].sum()).fillna(0)
    fba["source_dc"] = fba["sku_idx"].map(tr.groupby("sku_idx")["source"].first().map({0: "EAST_DC", 1: "WEST_DC"}))
    fba["remaining_source"] = (
        np.where(fba["source_dc"] == "WEST_DC", fba["west_available"], fba["east_available"]) - fba["send_qty"]
    )
    fba = fba[fba["fba_enabled"]]

    kpis = portfolio_kpis(view, st, plan, cfg)
    kpis.update(
        plan_date=str(plan_date.date()),
        created_at=str(created),
        runtime_seconds=round(time.time() - t0, 1),
        transit_p90=float(st.fba.attrs["transit_p90"]),
    )
    seg_scores = st.forecast_state.selection.segment_scores.reset_index().rename(columns={"index": "segment"})
    lt_rows = []
    for sid, dist in st.lead_times.items():
        lt_rows += [
            {"supplier_id": sid, "days": float(d), "prob": float(pr_)} for d, pr_ in zip(dist.days, dist.prob, strict=True)
        ]

    # write
    url = database_url(cfg)
    db = tables.engine_for(url)
    with db.begin() as conn:
        conn.execute(text("DELETE FROM plan_recommendations"))
        conn.execute(text("DELETE FROM demand_observations"))
    recs.assign(status="OPEN", stockout_date=pd.to_datetime(recs["stockout_date"]).dt.date)[
        [c.name for c in tables.recommendations.columns]
    ].to_sql("plan_recommendations", db, if_exists="append", index=False)
    dob[["date", "sku", "channel", "units", "reconstructed", "availability_status"]].to_sql(
        "demand_observations", db, if_exists="append", index=False, chunksize=50000
    )
    for name, df in (
        ("plan_sku", sp),
        ("plan_forecast", fcast),
        ("plan_history", hist),
        ("plan_projection", proj),
        ("plan_fba", fba),
        ("plan_containers", cont),
        ("plan_container_summary", cont_sum),
        ("plan_forecast_performance", diag.assign(sku=sku[diag["sku_idx"].to_numpy()])),
        ("plan_segment_scores", seg_scores),
        ("plan_lead_time_distribution", pd.DataFrame(lt_rows)),
        ("plan_kpis", pd.DataFrame([{"key": k, "value": json.dumps(v, default=float)} for k, v in kpis.items()])),
    ):
        df.to_sql(name, db, if_exists="replace", index=False, chunksize=50000)

    # the scenario simulator reuses the fitted champions and error tables, not the backtest itself
    light = dataclasses.replace(
        st.forecast_state, backtest=None, extra={"selection_horizon": st.forecast_state.extra["selection_horizon"]}
    )
    with open(sim / "plan_state.pkl", "wb") as fh:
        pickle.dump({"forecast_state": light, "kpis": kpis}, fh)
    counts = recs["action"].value_counts().to_dict()
    print(f"plan for {plan_date:%a %d %b %Y}: {len(recs)} recommendations {counts}")
    print(f"containers: {len(cont_sum)} suppliers; kpis: {json.dumps({k: kpis[k] for k in list(kpis)[:8]}, default=float)}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
