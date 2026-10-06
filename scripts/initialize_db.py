"""Create the database and load master data, the operational history and the evaluation results.

The operational history is World L at the reference seed: the business as the Legacy process ran
it, which is the state the live planning cycle starts from.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qstats_planner.domain import tables  # noqa: E402
from qstats_planner.domain.locations import LOCATIONS  # noqa: E402
from qstats_planner.inventory import lead_time as ltmod  # noqa: E402
from qstats_planner.simulation import runner  # noqa: E402
from qstats_planner.utils.config import database_url, load_config, resolve  # noqa: E402


def main() -> int:
    cfg = load_config()
    url = database_url(cfg)
    eng_db = tables.engine_for(url)
    tables.create_schema(eng_db)

    inp = runner.load_inputs(cfg)
    env, _, _ = runner.build_world(inp, cfg, "demo", cfg["random_seed"])
    sim = resolve(cfg["paths"]["simulation_dir"]) / "demo"
    with open(sim / f"world_{runner.LEGACY_REF}_seed{cfg['random_seed']}.pkl", "rb") as fh:
        world = pickle.load(fh)["engine"]
    world.env = env
    view = world.view(env.n_days - 1)
    days = env.days

    pop = inp.pop[inp.pop["population"] == "demo"].set_index("sku")["profile"]
    prod = env.products.assign(sku_idx=np.arange(env.n_sku), profile=lambda d: d["sku"].map(pop))
    cols = [c.name for c in tables.products.columns]
    prod[cols].to_sql("products", eng_db, if_exists="append", index=False)

    obs = view.lead_time_observations()
    lts = ltmod.estimate_all(obs, view.suppliers, cfg["inventory"]["lead_time_shrink_k"])
    sup = view.suppliers.copy()
    sup["lead_time_mean"] = [lts[s].mean for s in sup["supplier_id"]]
    sup["lead_time_std"] = [float(np.sqrt(((lts[s].days - lts[s].mean) ** 2 * lts[s].prob).sum())) for s in sup["supplier_id"]]
    sup["lead_time_p50"] = [lts[s].quantile(0.5) for s in sup["supplier_id"]]
    sup["lead_time_p90"] = [lts[s].quantile(0.9) for s in sup["supplier_id"]]
    sup["receipts"] = [lts[s].n_received for s in sup["supplier_id"]]
    sup["open_orders"] = [lts[s].n_open for s in sup["supplier_id"]]
    sup[[c.name for c in tables.suppliers.columns]].to_sql("suppliers", eng_db, if_exists="append", index=False)
    pd.DataFrame({"location_id": LOCATIONS, "kind": ["DC", "DC", "FBA"]}).to_sql(
        "locations", eng_db, if_exists="append", index=False
    )

    po = view.purchase_orders()
    d = lambda s: pd.to_datetime(days[0]) + pd.to_timedelta(s, unit="D")  # noqa: E731
    po_out = pd.DataFrame(
        {
            "po_id": po["po_id"],
            "sku": env.products.loc[po["sku_idx"], "sku"].to_numpy(),
            "supplier_id": po["supplier_id"],
            "quantity": po["quantity"],
            "qty_east": po["qty_east"],
            "qty_west": po["qty_west"],
            "order_date": d(po["order_day"]).dt.date,
            "expected_arrival": d(po["expected_day"]).dt.date,
            "actual_arrival": d(po["actual_day"]).dt.date,
            "status": po["status"],
            "policy": po["policy"],
        }
    )
    po_out.to_sql("purchase_orders", eng_db, if_exists="append", index=False)
    obs.rename(columns={"event": "received"})[["po_id", "supplier_id", "lead_time_days", "received"]].to_sql(
        "lead_time_observations", eng_db, if_exists="append", index=False
    )

    start = world.warm_end
    T, n, L = world.sales.shape
    idx = pd.MultiIndex.from_product([range(start, T), range(n), range(L)], names=["day", "sku_idx", "loc"])
    snap = pd.DataFrame(
        {
            "on_hand": world.closing_on_hand[start:].ravel(),
            "available": world.closing_available[start:].ravel(),
            "sales": world.sales[start:].ravel(),
            "receipts": world.receipts[start:].ravel(),
        },
        index=idx,
    ).reset_index()
    snap["date"] = days[snap["day"].to_numpy()].date
    snap["sku"] = env.products["sku"].to_numpy()[snap["sku_idx"].to_numpy()]
    snap["location"] = np.array(LOCATIONS)[snap["loc"].to_numpy()]
    fba_off = ~env.products["fba_enabled"].to_numpy()[snap["sku_idx"].to_numpy()] & (snap["loc"] == 2)
    snap[~fba_off][["date", "sku", "location", "on_hand", "available", "sales", "receipts"]].to_sql(
        "inventory_snapshots", eng_db, if_exists="append", index=False, chunksize=50000
    )

    # evaluation layer (read by the comparison and benchmark pages only)
    for name in ("summary", "per_sku", "bootstrap", "calibration"):
        f = sim / f"{name}.parquet"
        if f.exists():
            pd.read_parquet(f).to_sql(f"eval_{name}", eng_db, if_exists="replace", index=False, chunksize=50000)
    for sc in ("null", "optimistic_quotes"):
        f = sim / f"scenario_{sc}" / "summary.parquet"
        if f.exists():
            pd.read_parquet(f).assign(scenario=sc).to_sql(f"eval_scenario_{sc}", eng_db, if_exists="replace", index=False)
    for name in ("scores", "episodes", "weekly"):
        f = sim / "benchmark" / f"{name}.parquet"
        if f.exists():
            pd.read_parquet(f).to_sql(f"eval_benchmark_{name}", eng_db, if_exists="replace", index=False, chunksize=50000)
    meta = {
        k: json.loads((sim / f).read_text()) for k, f in (("matched", "matched.json"), ("run", "run.json")) if (sim / f).exists()
    }
    pd.DataFrame([{"key": k, "value": json.dumps(v)} for k, v in meta.items()]).to_sql(
        "eval_meta", eng_db, if_exists="replace", index=False
    )

    # both worlds, weekly network totals, for the comparison charts
    rows = []
    for name in (runner.LEGACY_REF, runner.QSTATS_REF):
        with open(sim / f"world_{name}_seed{cfg['random_seed']}.pkl", "rb") as fh:
            w = pickle.load(fh)["engine"]
        W = T // 7

        def r(a, W=W):
            return a[: W * 7].reshape(W, 7, *a.shape[1:]).sum(axis=1)

        inv = np.nan_to_num(w.closing_on_hand[: W * 7]).sum(axis=2) * env.products["unit_cost"].to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "world": name,
                    "week": days[: W * 7 : 7],
                    "demand": r(env.baseline).sum(axis=(1, 2)),
                    "sold": r(w.sales).sum(axis=(1, 2)),
                    "lost": r(w.lost).sum(axis=(1, 2)),
                    "inventory_value": inv.reshape(W, 7, -1).sum(axis=2).mean(axis=1),
                }
            )
        )
    pd.concat(rows).to_sql("eval_world_weekly", eng_db, if_exists="replace", index=False)
    print(f"database ready: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
