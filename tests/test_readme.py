"""The README's numbers must match the evaluation artefacts they come from."""

import json
import re

import pandas as pd
import pytest

from qstats_planner.utils.config import ROOT

SIM = ROOT / "data" / "simulation" / "demo"
README = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())


def p1(v):
    return f"{abs(v) * 100:.1f}%"


def signed(v):
    return ("−" if v < 0 else "+") + p1(v)


def more_or_less(v):
    """A saving share as the README words it: positive saving = less inventory."""
    return f"{p1(v)} {'less' if v > 0 else 'more'}"


@pytest.mark.skipif(not (SIM / "matched.json").exists(), reason="run `make demo` first")
def test_readme_numbers_match_artefacts():
    m = json.loads((SIM / "matched.json").read_text())
    ref = m["42"]
    summ = pd.read_parquet(SIM / "summary.parquet")
    h = summ[(summ["subset"] == "headline") & (summ["seed"] == 42)].set_index("variant")
    boot = pd.read_parquet(SIM / "bootstrap.parquet")
    b0 = boot[boot["seed"] == 42]
    lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
    slo, shi = b0["legacy_extra_inventory_pct"].quantile([0.05, 0.95])
    cal = pd.read_parquet(SIM / "calibration.parquet")
    c = cal[cal["seed"] == 42]
    null = json.loads((SIM / "scenario_null" / "matched.json").read_text())["42"]
    nb = pd.read_parquet(SIM / "scenario_null" / "bootstrap.parquet")
    nlo, nhi = nb["inventory_saving_pct"].quantile([0.05, 0.95])
    L, Q = h.loc["legacy_30d"], h.loc["qstats"]
    expected = [
        f"| Fill rate | {p1(L['fill_rate'])} | {p1(Q['fill_rate'])} |",
        f"| Average inventory | ${L['average_inventory_value'] / 1e3:.1f}k | ${Q['average_inventory_value'] / 1e3:.1f}k |",
        f"| Forecast bias | {signed(L['forecast_bias'])} | {signed(Q['forecast_bias'])} |",
        f"QStats needed {p1(ref['inventory_saving_pct'])} *more* inventory",
        f"(90% interval: from {more_or_less(hi)} to {more_or_less(lo)};",
        f"{more_or_less(m['7']['inventory_saving_pct'])} and {more_or_less(m['2026']['inventory_saving_pct'])} in two replicate",
        f"In {ref['bootstrap_out_of_range_primary'] * 100:.1f}% of the bootstrap resamples",
        f"needs {p1(ref['legacy_extra_inventory_pct'])} more inventory than QStats (90% interval {signed(slo)} to {signed(shi)})",
        f"{p1(m['7']['legacy_extra_inventory_pct'])} and {p1(m['2026']['legacy_extra_inventory_pct'])} in the replicate worlds",
        f"held {p1((c['realised'] <= c['q90']).mean())} of outcomes and the P95 {p1((c['realised'] <= c['q95']).mean())}",
        f"saved {p1(null['inventory_saving_pct'])} at the legacy fill rate (interval {p1(nlo)} to {p1(nhi)})",
        f"recovered ${(L['lost_contribution'] - Q['lost_contribution']) / 1e3:.1f}k of contribution",
        f"at ${(Q['holding_cost'] - L['holding_cost']) / 1e3:.1f}k of extra carrying cost",
    ]
    for e in expected:
        assert re.sub(r"\s+", " ", e) in README, e


@pytest.mark.skipif(not (SIM / "benchmark" / "scores.parquet").exists(), reason="run `make demo` first")
def test_readme_benchmark_table_matches():
    s = pd.read_parquet(SIM / "benchmark" / "scores.parquet")
    s = s[s["channels"] == "all"]
    two = s[s["mode"] == "retrospective"].set_index("method")
    rt = s[s["mode"] == "real_time"].set_index("method")
    for method in ("pre_post_velocity", "local_profile", "model_expectation", "censored_gamma"):
        r = two.loc[method]
        row = (f"| {r['episode_mae']:.0f} | −{abs(r['episode_bias']):.0f} | {r['recovery_pct'] * 100:.0f}% | "
               f"{rt.at[method, 'recovery_pct'] * 100:.0f}% |")
        assert row in README, (method, row)
