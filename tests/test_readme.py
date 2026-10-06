"""The README's numbers must match the evaluation artefacts they come from."""

import json

import pandas as pd
import pytest

from qstats_planner.utils.config import ROOT

SIM = ROOT / "data" / "simulation" / "demo"
README = (ROOT / "README.md").read_text()


def pct(v, signed=False):
    s = f"{abs(v) * 100:.1f}%"
    return (("−" if v < 0 else "+") + s) if signed else s


@pytest.mark.skipif(not (SIM / "matched.json").exists(), reason="run `make demo` first")
def test_readme_numbers_match_artefacts():
    m = json.loads((SIM / "matched.json").read_text())
    ref = m["42"]
    summ = pd.read_parquet(SIM / "summary.parquet")
    h = summ[(summ["subset"] == "headline") & (summ["seed"] == 42)].set_index("variant")
    boot = pd.read_parquet(SIM / "bootstrap.parquet")
    b0 = boot[boot["seed"] == 42]
    lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
    cal = pd.read_parquet(SIM / "calibration.parquet")
    c = cal[cal["seed"] == 42]
    expected = [
        f"| Fill rate | {pct(h.at['legacy_30d', 'fill_rate'])} | {pct(h.at['qstats', 'fill_rate'])} |",
        f"| Average inventory | ${h.at['legacy_30d', 'average_inventory_value'] / 1e3:.1f}k | "
        f"${h.at['qstats', 'average_inventory_value'] / 1e3:.1f}k |",
        f"gave {pct(ref['inventory_saving_pct'], True).replace('+', '')}",
        f"(90% interval {pct(lo, True)} to {pct(hi, True)}",
        f"{pct(m['7']['inventory_saving_pct'], True)} and {pct(m['2026']['inventory_saving_pct'], True)} in two replicate",
        f"needs {pct(ref['legacy_extra_inventory_pct'])} more inventory",
        f"({pct(m['7']['legacy_extra_inventory_pct'])} and {pct(m['2026']['legacy_extra_inventory_pct'])} in the replicates)",
        f"held {pct((c['realised'] <= c['q90']).mean())} of outcomes and the P95 {pct((c['realised'] <= c['q95']).mean())}",
        f"| Forecast bias | {pct(h.at['legacy_30d', 'forecast_bias'], True)} | {pct(h.at['qstats', 'forecast_bias'], True)} |",
    ]
    for e in expected:
        assert e in README, e


@pytest.mark.skipif(not (SIM / "benchmark" / "scores.parquet").exists(), reason="run `make demo` first")
def test_readme_benchmark_table_matches():
    s = pd.read_parquet(SIM / "benchmark" / "scores.parquet")
    r = s[s["mode"] == "retrospective"].set_index("method")
    for method in ("local_profile", "censored_gamma"):
        row = r.loc[method]
        assert f"| {row['episode_mae']:.0f} | −{abs(row['episode_bias']):.0f} | {row['recovery_pct'] * 100:.0f}% |" in README, (
            method
        )
