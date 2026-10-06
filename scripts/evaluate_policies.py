"""Legacy vs QStats on the same simulated business: frontier, ablation and replicate worlds.

    python scripts/evaluate_policies.py --population dev     # development (tuning allowed)
    python scripts/evaluate_policies.py --population demo    # the reported comparison

Every variant forks from the same week-52 state of the same world. Replicate worlds change
supplier luck only (identical products and identical demand). Results go to
data/simulation/<population>/: per-SKU outcomes per variant and seed, summaries, the bootstrap,
calibration rows, and (for the reference seed) both worlds' daily records for the app.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qstats_planner.evaluation import calibration, comparison  # noqa: E402
from qstats_planner.simulation import runner  # noqa: E402
from qstats_planner.utils.config import load_config, resolve  # noqa: E402
from qstats_planner.utils.parallel import default_workers  # noqa: E402

_INP = None
_CFG = None
_BASES: dict = {}


def _init():
    global _INP, _CFG
    _CFG = load_config()
    _INP = runner.load_inputs(_CFG)


def _task(args):
    population, seed, names, out_dir, keep, scenario = args
    cfg = _CFG
    t0 = time.time()
    env, prior, base = runner.build_world(_INP, cfg, population, seed, scenario)
    start, end = comparison.scoring_window(env, cfg)
    event_skus = set(env.event_skus)  # same SKUs in every world, so headlines stay comparable
    results = []
    for name in names:
        eng, pol = runner.run_variant(base, name, cfg, prior)
        df = comparison.per_sku(env, eng, pol.history, cfg, start, end)
        df["variant"], df["seed"] = name, seed
        df["event_sku"] = df.index.isin(event_skus)
        cal = None
        if name == runner.QSTATS_REF:
            cal = calibration.calibration_rows(env, pol.history, start, cfg["simulation"]["review_period_days"])
            cal["seed"] = seed
        if keep and name in (runner.LEGACY_REF, runner.QSTATS_REF):
            eng.env = None
            with open(Path(out_dir) / f"world_{name}_seed{seed}.pkl", "wb") as fh:
                pickle.dump({"engine": eng, "history": pol.history}, fh)
        results.append((name, df, cal))
    return population, seed, results, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="demo")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=default_workers(), help="default: min(8, CPUs, 75% of memory / 1.2 GB)")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--scenario", default="base", choices=["base", "null", "optimistic_quotes"])
    args = ap.parse_args()
    cfg = load_config()
    seeds = args.seeds or cfg["simulation"]["seeds"]
    if cfg["random_seed"] not in seeds:
        # the outputs below replace the reference artefacts that the app and the README tests read
        ap.error(f"--seeds must include the reference seed {cfg['random_seed']}")
    names = args.variants or runner.variant_names()
    out = resolve(cfg["paths"]["simulation_dir"]) / args.population
    if args.scenario != "base":
        out = out / f"scenario_{args.scenario}"
    out.mkdir(parents=True, exist_ok=True)

    # one task per (seed, chunk of variants); QStats variants are the slow ones, so spread them
    legacy = [n for n in names if n.startswith("legacy")]
    others = [n for n in names if not n.startswith("legacy")]
    tasks = []
    for s in seeds:
        keep = s == cfg["random_seed"] and args.scenario == "base"
        tasks.append((args.population, s, legacy, str(out), keep, args.scenario))
        for n in others:
            tasks.append((args.population, s, [n], str(out), keep, args.scenario))
    t0 = time.time()
    frames, cals = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for _pop, seed, res, secs in ex.map(_task, tasks):
            for _name, df, cal in res:
                frames.append(df)
                if cal is not None:
                    cals.append(cal)
            print(f"  seed {seed}: {', '.join(r[0] for r in res)} ({secs:.0f}s)", flush=True)
    per_sku = pd.concat(frames, ignore_index=True)
    per_sku.to_parquet(out / "per_sku.parquet", index=False)
    if cals:
        pd.concat(cals, ignore_index=True).to_parquet(out / "calibration.parquet", index=False)

    rows = []
    for (variant, seed), g in per_sku.groupby(["variant", "seed"]):
        g = g.reset_index(drop=True)
        for subset, m in (("headline", ~g["event_sku"]), ("all", g["event_sku"] | True), ("event_skus", g["event_sku"])):
            if m.any():
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "subset": subset,
                        **comparison.summarise(g[m.to_numpy()].reset_index(drop=True), cfg),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_parquet(out / "summary.parquet", index=False)
    head = per_sku[~per_sku["event_sku"]]

    ref_seed = cfg["random_seed"]
    boots = []
    for seed in [s_ for s_ in seeds if (head["seed"] == s_).any()]:
        pv = {v: g.reset_index(drop=True) for v, g in head[head["seed"] == seed].groupby("variant")}
        if not set(runner.LEGACY_FAMILY + runner.QSTATS_FAMILY) <= set(pv):
            continue
        b = comparison.bootstrap(
            pv, cfg, runner.LEGACY_REF, runner.LEGACY_FAMILY, runner.QSTATS_FAMILY, n_boot=1000 if seed == ref_seed else 200
        )
        b["seed"] = seed
        boots.append(b)
    if boots:
        pd.concat(boots, ignore_index=True).to_parquet(out / "bootstrap.parquet", index=False)

    meta = {
        "population": args.population,
        "seeds": seeds,
        "variants": names,
        "runtime_seconds": round(time.time() - t0, 1),
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    show = summary[(summary["seed"] == ref_seed) & (summary["subset"] == "headline")].set_index("variant")
    cols = [
        "fill_rate",
        "in_stock_rate",
        "average_inventory_value",
        "lost_contribution",
        "inventory_turns",
        "excess_inventory_value",
        "wape",
        "forecast_bias",
    ]
    print(show[cols].round(3).to_string())
    if boots:
        b = pd.concat(boots)
        points = {}
        for seed in seeds:
            sm = {
                v: comparison.summarise(g.reset_index(drop=True), cfg) for v, g in head[head["seed"] == seed].groupby("variant")
            }
            points[int(seed)] = comparison.matched_comparison(sm, runner.LEGACY_REF, runner.LEGACY_FAMILY, runner.QSTATS_FAMILY)
        b0 = b[b["seed"] == ref_seed]
        lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
        out_of_range = float(b0["inventory_saving_pct"].isna().mean())
        for s_ in points:
            bs = b[b["seed"] == s_]
            below = bs["legacy_fill"] < bs["qstats_lowest_fill"]
            points[s_]["bootstrap_out_of_range_primary"] = float(bs["inventory_saving_pct"].isna().mean())
            # out of range because even QStats's lowest setting delivered more fill than Legacy-30 ...
            points[s_]["bootstrap_out_of_range_primary_below"] = float(below.mean())
            # ... and did so with no more inventory (QStats better on both counts)
            points[s_]["bootstrap_out_of_range_primary_dominated"] = float(
                (below & (bs["qstats_lowest_inventory"] <= bs["legacy_inventory"])).mean()
            )
            points[s_]["bootstrap_out_of_range_secondary"] = float(bs["legacy_extra_inventory_pct"].isna().mean())
            # out of range because QStats's fill is above the highest legacy setting's
            points[s_]["bootstrap_out_of_range_secondary_above"] = float((bs["qstats_fill"] > bs["legacy_highest_fill"]).mean())
        (out / "matched.json").write_text(json.dumps(points, indent=2))
        print(
            f"\nprimary metric: inventory saving at Legacy-30's fill rate, seed {ref_seed}: "
            f"{points[ref_seed]['inventory_saving_pct']:.3f} (90% bootstrap interval {lo:.3f} to {hi:.3f}, "
            f"{out_of_range:.1%} of resamples outside the frontier)"
        )
        print("point estimate by seed:", {s_: round(float(v["inventory_saving_pct"]), 3) for s_, v in points.items()})
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
