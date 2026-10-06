"""Simulate the supply chain around the real demand, run the policy comparison and the
constrained-demand benchmark.

    python scripts/simulate_supply_chain.py            # demo population, all seeds + scenarios
    python scripts/simulate_supply_chain.py --quick    # reference seed only (faster)

The policy comparison itself lives in scripts/evaluate_policies.py (see docs/EVAL_PLAN.md).
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qstats_planner.demand import reconstruction as R  # noqa: E402
from qstats_planner.evaluation.benchmark import episode_table, score  # noqa: E402
from qstats_planner.simulation import runner  # noqa: E402
from qstats_planner.utils.config import load_config, resolve  # noqa: E402


def benchmark(cfg: dict) -> None:
    """Score every reconstruction method on the demo SKUs' World L history (evaluation layer)."""
    inp = runner.load_inputs(cfg)
    env, prior, _ = runner.build_world(inp, cfg, "demo", cfg["random_seed"])
    out = resolve(cfg["paths"]["simulation_dir"]) / "demo"
    with open(out / f"world_{runner.LEGACY_REF}_seed{cfg['random_seed']}.pkl", "rb") as fh:
        eng = pickle.load(fh)["engine"]
    eng.env = env
    view = eng.view(env.n_days - 1)
    cd = R.channel_data(view, prior.daily_factors(env.days))
    B = env.baseline
    truth = np.stack([B[:, :, 0] + B[:, :, 1], B[:, :, 2]], axis=2)
    rc = cfg["reconstruction"]
    status, cen = R.classify(cd, rc["unknown_zero_run_probability"])
    # channels stocked at least once after the warm-up: a channel never stocked looks exactly like a
    # channel with no demand, and no method can recover it from sales alone
    warm = eng.warm_end
    stocked = (np.nan_to_num(cd.opening[warm:]) > 0).any(axis=0) & cd.active[warm:].any(axis=0)
    rows, recs = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for two in (False, True):
            for m in R.METHODS:
                rec = R.reconstruct(
                    cd,
                    m,
                    env.days.dayofweek.to_numpy(),
                    two_sided=two,
                    window=rc["window_trading_days"],
                    min_clean=rc["min_clean_days"],
                    unknown_zero_run_p=rc["unknown_zero_run_probability"],
                    gamma_min_shape=rc["gamma_min_shape"],
                )
                mode = "retrospective" if two else "real_time"
                rows.append(
                    {"method": m, "mode": mode, "channels": "all", **score(cd.sales, rec.adjusted, truth, cen, cd.active)}
                )
                rows.append(
                    {
                        "method": m,
                        "mode": mode,
                        "channels": "stocked at least once",
                        **score(cd.sales, rec.adjusted, truth, cen & stocked[None], cd.active),
                    }
                )
                if two:
                    recs[m] = rec.adjusted
    bdir = out / "benchmark"
    bdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(bdir / "scores.parquet", index=False)
    ep = episode_table(cd.sales, recs[rc["planner_method"]], truth, cen, cd.active)
    ep["sku"] = env.products.loc[ep["sku_idx"], "sku"].to_numpy()
    ep["stocked"] = stocked[ep["sku_idx"].to_numpy(), ep["channel"].to_numpy()]
    ep["channel"] = np.where(ep["channel"] == 0, "DIRECT", "AMAZON")
    ep["start_date"] = env.days[ep["start_day"].to_numpy()]
    ep["end_date"] = env.days[ep["end_day"].to_numpy()]
    ep.to_parquet(bdir / "episodes.parquet", index=False)
    # weekly series per SKU and channel: baseline, observed, each method (for the benchmark page)
    W = env.n_days // 7
    wk = lambda a: a[: W * 7].reshape(W, 7, *a.shape[1:]).sum(axis=1)  # noqa: E731
    frames = []
    for c, name in ((0, "DIRECT"), (1, "AMAZON")):
        base = {
            "baseline": wk(truth[:, :, c]),
            "observed": wk(cd.sales[:, :, c]),
            "censored_days": wk(cen[:, :, c].astype(float)),
        }
        base.update({m: wk(recs[m][:, :, c]) for m in recs if m != "no_adjustment"})
        idx = pd.MultiIndex.from_product([env.days[: W * 7 : 7], env.products["sku"]], names=["week", "sku"])
        df = pd.DataFrame({k: v.ravel() for k, v in base.items()}, index=idx).reset_index()
        df["channel"] = name
        df["stocked"] = df["sku"].map(dict(zip(env.products["sku"], stocked[:, c], strict=True)))
        frames.append(df[(df["baseline"] > 0) | (df["observed"] > 0)])
    pd.concat(frames, ignore_index=True).to_parquet(bdir / "weekly.parquet", index=False)
    print(
        "benchmark:",
        pd.DataFrame(rows)
        .query("mode == 'retrospective'")[["method", "channels", "episode_mae", "episode_bias", "recovery_pct"]]
        .round(3)
        .to_string(index=False),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    py = [sys.executable, str(ROOT / "scripts" / "evaluate_policies.py"), "--population", "demo"]
    seeds = ["--seeds", str(cfg["random_seed"])] if args.quick else []
    subprocess.run(py + seeds, check=True)
    for sc in ("null", "optimistic_quotes"):
        subprocess.run(py + ["--seeds", str(cfg["random_seed"]), "--scenario", sc], check=True)
    benchmark(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
