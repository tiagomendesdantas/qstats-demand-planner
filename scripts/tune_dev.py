"""Development decisions, made on the dev SKUs only (never on the demo SKUs).

1. Legacy smoothing constant: the alpha a legacy team would pick, by one-week-ahead MAE on the
   sales it can see (weeks 13-52 of the shared history, dev SKUs).
2. Reconstruction method for the planner: lowest two-sided episode MAE on the dev SKUs' censored
   history (World L, full period).

Writes data/simulation/dev/tuning.json. The chosen values are then set in config/demo.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from qstats_planner.demand import reconstruction as R  # noqa: E402
from qstats_planner.evaluation.benchmark import score  # noqa: E402
from qstats_planner.replenishment.common import ses_levels, weekly  # noqa: E402
from qstats_planner.simulation import runner  # noqa: E402
from qstats_planner.utils.config import load_config, resolve  # noqa: E402


def main() -> int:
    cfg = load_config()
    inp = runner.load_inputs(cfg)
    env, prior, base = runner.build_world(inp, cfg, "dev", cfg["random_seed"])
    out: dict = {}

    # 1. Legacy alpha on the observed (censored) sales before the fork
    view = base.view(base.day - 1)
    Y = weekly(view.sales, view.t).sum(axis=2)
    E = view.trading[: Y.shape[0] * 7].reshape(-1, 7).sum(axis=1)
    launch = np.where(view.launch_day >= 0, view.launch_day // 7, 10_000)
    valid = (E > 0)[:, None] & (np.arange(len(E))[:, None] >= launch[None, :])
    weeks = np.arange(len(E))
    score_mask = (weeks >= cfg["simulation"]["warmup_weeks"])[:, None] & valid
    res = {}
    for a in cfg["forecasting"]["ses_alphas"]:
        lv = ses_levels(Y, a, valid)
        fc = np.vstack([np.full((1, Y.shape[1]), np.nan), lv[:-1]])
        m = score_mask & np.isfinite(fc)
        res[a] = float(np.abs(fc[m] - Y[m]).mean())
    best = min(res, key=res.get)
    out["legacy_alpha"] = {"mae_by_alpha": res, "chosen": best}

    # 2. reconstruction method on the dev SKUs' full World L history
    eng, _ = runner.run_variant(base, runner.LEGACY_REF, cfg, prior)
    v = eng.view(env.n_days - 1)
    cd = R.channel_data(v, prior.daily_factors(env.days))
    B = env.baseline
    truth = np.stack([B[:, :, 0] + B[:, :, 1], B[:, :, 2]], axis=2)
    rc = cfg["reconstruction"]
    cen = R.classify(cd, rc["unknown_zero_run_probability"])[1]
    table = {}
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
            table[f"{'two' if two else 'one'}_sided/{m}"] = score(cd.sales, rec.adjusted, truth, cen, cd.active)
    two_sided = {
        k.split("/")[1]: v["episode_mae"] for k, v in table.items() if k.startswith("two") and not k.endswith("no_adjustment")
    }
    out["reconstruction"] = {"scores": table, "chosen": min(two_sided, key=two_sided.get)}

    path = resolve(cfg["paths"]["simulation_dir"]) / "dev"
    path.mkdir(parents=True, exist_ok=True)
    (path / "tuning.json").write_text(json.dumps(out, indent=2, default=float))
    print("legacy alpha:", res, "-> chosen", best)
    for k, v in table.items():
        print(f"  {k:32s} episode MAE {v['episode_mae']:7.1f}  bias {v['episode_bias']:7.1f}  recovery {v['recovery_pct']:.1%}")
    print("reconstruction chosen:", out["reconstruction"]["chosen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
