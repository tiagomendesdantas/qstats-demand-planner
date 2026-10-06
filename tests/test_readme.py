"""The README's numbers must match the evaluation artefacts they come from, including the direction
words ("more", "less", the sign of a bias), which are derived from the values here, not hard-coded."""

import json
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from qstats_planner.utils.config import ROOT

SIM = ROOT / "data" / "simulation" / "demo"
DB = ROOT / "data" / "planner.sqlite"
README = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())


def p1(v):
    return f"{abs(v) * 100:.1f}%"


def signed(v):
    return ("−" if v < 0 else "+") + p1(v)


def more_or_less(v):
    """A saving share as the README words it: positive saving = less inventory."""
    return f"{p1(v)} {'less' if v > 0 else 'more'}"


def more_or_less_extra(v):
    """Inventory the legacy rule needs beyond QStats's: positive = more."""
    return f"{p1(v)} {'more' if v > 0 else 'less'}"


def k(v):
    return f"${v / 1e3:.1f}k"


def check(expected):
    for e in expected:
        assert re.sub(r"\s+", " ", e) in README, e


def _matched():
    if not (SIM / "matched.json").exists():
        pytest.skip("run `make demo` first")
    m = json.loads((SIM / "matched.json").read_text())
    if not {"42", "7", "2026"} <= set(m):
        pytest.skip("replicate worlds not run (simulate_supply_chain.py --quick)")
    return m


def test_readme_comparison_numbers_match_artefacts():
    m = _matched()
    ref = m["42"]
    summ = pd.read_parquet(SIM / "summary.parquet")
    head = summ[summ["subset"] == "headline"]
    h = head[head["seed"] == 42].set_index("variant")
    boot = pd.read_parquet(SIM / "bootstrap.parquet")
    b0 = boot[boot["seed"] == 42]
    lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
    L, Q, L120 = h.loc["legacy_30d"], h.loc["qstats"], h.loc["legacy_120d"]
    sav = ref["inventory_saving_pct"]
    amount, word = p1(sav), ("less" if sav > 0 else "more")
    check(
        [
            f"| Fill rate | {p1(L['fill_rate'])} | {p1(Q['fill_rate'])} |",
            f"| Average inventory | {k(L['average_inventory_value'])} | {k(Q['average_inventory_value'])} |",
            f"| Lost contribution | {k(L['lost_contribution'])} | {k(Q['lost_contribution'])} |",
            f"| Forecast bias | {signed(L['forecast_bias'])} | {signed(Q['forecast_bias'])} |",
            f"QStats needed {amount} *{word}* inventory (90% interval: from {more_or_less(hi)} to {more_or_less(lo)};",
            f"{more_or_less(m['7']['inventory_saving_pct'])} and "
            f"{more_or_less(m['2026']['inventory_saving_pct'])} in two replicate",
            f"In {ref['bootstrap_out_of_range_primary'] * 100:.1f}% of the bootstrap resamples",
            f"in {round(ref['bootstrap_out_of_range_primary_dominated'] * len(b0))} of the {len(b0):,} it did so "
            "with no more inventory",
        ]
    )
    # every excluded resample lies below QStats's lowest setting, as the README says
    assert ref["bootstrap_out_of_range_primary_below"] == ref["bootstrap_out_of_range_primary"]

    # secondary: out of range in the reference world because QStats's fill is above the legacy frontier
    assert np.isnan(ref["legacy_extra_inventory_pct"]) and Q["fill_rate"] > L120["fill_rate"]
    assert ref["bootstrap_out_of_range_secondary_above"] == ref["bootstrap_out_of_range_secondary"]
    check(
        [
            f"QStats delivered {p1(Q['fill_rate'])} fill",
            f"120 days of safety stock, reached {p1(L120['fill_rate'])} and held "
            f"{more_or_less_extra(L120['average_inventory_value'] / Q['average_inventory_value'] - 1)} inventory than QStats",
            f"as it is in {ref['bootstrap_out_of_range_secondary'] * 100:.1f}% of its resamples",
            f"the legacy rule needed {more_or_less_extra(m['7']['legacy_extra_inventory_pct'])} and "
            f"{more_or_less_extra(m['2026']['legacy_extra_inventory_pct'])} inventory than QStats",
        ]
    )

    def net(x):
        return (
            (L["lost_contribution"] - x["lost_contribution"])
            + (L["cross_dc_cost"] - x["cross_dc_cost"])
            - (x["holding_cost"] - L["holding_cost"])
        )

    check(
        [
            f"on {p1(1 - Q['average_inventory_value'] / L120['average_inventory_value'])} less inventory than that",
            f"recovered {k(L['lost_contribution'] - Q['lost_contribution'])} of contribution and saved "
            f"{k(L['cross_dc_cost'] - Q['cross_dc_cost'])} of cross-DC shipping, at "
            f"{k(Q['holding_cost'] - L['holding_cost'])} of extra carrying cost: about {'+' if net(Q) > 0 else '−'}"
            f"${abs(net(Q)) / 1e3:.0f}k",
            f"ending with {k(Q['excess_inventory_value'] - L['excess_inventory_value'])} more stock",
            f"would net about {'+' if net(L120) > 0 else '−'}${abs(net(L120)) / 1e3:.0f}k against its 30-day self",
        ]
    )

    seeds = (42, 7, 2026)
    w = {s: head[head["seed"] == s].set_index("variant") for s in seeds}
    def three(variant):
        v = [f"{w[s].at[variant, 'wape']:.3f}" for s in seeds]
        return f"{v[0]}, {v[1]} and {v[2]}"

    wq, wl = three("qstats"), three("legacy_30d")
    assert all(w[s].at["qstats", "wape"] > w[s].at["legacy_30d", "wape"] for s in seeds)  # "got slightly worse"
    check([f"One-week WAPE {wq} against the legacy process's {wl}"])

    cal = pd.read_parquet(SIM / "calibration.parquet")
    c = cal[cal["seed"] == 42]
    check(
        [
            f"held {p1((c['realised'] <= c['q90']).mean())} of outcomes and the P95 {p1((c['realised'] <= c['q95']).mean())}",
        ]
    )

    null = json.loads((SIM / "scenario_null" / "matched.json").read_text())["42"]
    nb = pd.read_parquet(SIM / "scenario_null" / "bootstrap.parquet")
    nlo, nhi = nb["inventory_saving_pct"].quantile([0.05, 0.95])
    xlo, xhi = nb["legacy_extra_inventory_pct"].quantile([0.05, 0.95])
    assert null["inventory_saving_pct"] > 0  # the README says "saved"
    check(
        [
            f"QStats saved {p1(null['inventory_saving_pct'])} at the legacy fill rate (interval from {more_or_less(nlo)} to "
            f"{more_or_less(nhi)})",
            f"the legacy rule needed {more_or_less_extra(null['legacy_extra_inventory_pct'])} inventory to reach QStats's fill "
            f"(interval {signed(xlo)} to {signed(xhi)})",
        ]
    )


def test_readme_ablation_and_targets_match():
    _matched()
    from qstats_planner.evaluation.comparison import efficiency_vs_legacy
    from qstats_planner.simulation.runner import LEGACY_FAMILY

    summ = pd.read_parquet(SIM / "summary.parquet")
    head = summ[summ["subset"] == "headline"]
    eff = {
        s: efficiency_vs_legacy({v: dict(r) for v, r in g.set_index("variant").iterrows()}, LEGACY_FAMILY)
        for s, g in head.groupby("seed")
    }
    pr = [eff[s]["legacy_30d_prior"] for s in (42, 7, 2026)]
    check([f"{signed(pr[0])}, {signed(pr[1])} and {signed(pr[2])} in the three worlds"])
    readable = [eff[s]["qstats"] for s in (42, 7, 2026) if np.isfinite(eff[s]["qstats"])]
    assert len(readable) == 2  # "Full QStats can be read in two of them"
    check([f"Full QStats can be read in two of them ({signed(readable[0])} and {signed(readable[1])})"])
    wins = sum(int(eff[s]["legacy_30d_prior"] >= eff[s]["qstats"]) for s in (42, 7, 2026) if np.isfinite(eff[s]["qstats"]))
    assert wins == 2  # "the prior-only change beats it in both"
    for s, g in head.groupby("seed"):
        a, b = g.set_index("variant").loc["qstats_sl95"], g.set_index("variant").loc["qstats"]
        assert a["fill_rate"] >= b["fill_rate"] and a["average_inventory_value"] <= b["average_inventory_value"], s


@pytest.mark.skipif(not (SIM / "benchmark" / "scores.parquet").exists(), reason="run `make demo` first")
def test_readme_benchmark_table_matches():
    s = pd.read_parquet(SIM / "benchmark" / "scores.parquet")
    s = s[s["channels"] == "all"]
    two = s[s["mode"] == "retrospective"].set_index("method")
    rt = s[s["mode"] == "real_time"].set_index("method")
    for method in ("no_adjustment", "pre_post_velocity", "local_profile", "model_expectation", "censored_gamma"):
        r = two.loc[method]
        bias = f"{'−' if r['episode_bias'] < 0 else '+'}{abs(r['episode_bias']):.0f}"
        row = (
            f"| {r['episode_mae']:.0f} | {bias} | {r['recovery_pct'] * 100:.0f}% | "
            f"{rt.at[method, 'recovery_pct'] * 100:.0f}% |"
        )
        assert row in README, (method, row)
    r0 = two.loc["no_adjustment"]
    days, episodes, lost = int(r0["censored_days"]), int(r0["episodes"]), int(r0["lost_units"])
    check([f"{days:,} censored channel-days in {episodes:,} episodes, {lost:,} units"])
    assert (two.drop("no_adjustment")["episode_bias"] < 0).all()  # "Every method still under-estimates"


@pytest.mark.skipif(not DB.exists(), reason="run `make demo` first: no database")
def test_readme_live_plan_numbers_match_the_database():
    with sqlite3.connect(DB) as conn:
        kp = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT key, value FROM plan_kpis")}
        sp = pd.read_sql("SELECT selection_reason FROM plan_sku", conn)
        ets = pd.read_sql("SELECT * FROM plan_ets_challenger WHERE windows > 0", conn)
        meta = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT key, value FROM eval_meta")}
    check(
        [
            f"On that date {kp['skus_at_stockout_risk']} of {kp['skus_monitored']} SKUs are projected to run out",
            f"The plan has {kp['purchase_lines']} purchase lines worth {k(kp['purchase_value'])} "
            f"({kp['purchase_lines_for_review']} routed to a person",
            f"stock and open orders can serve {kp['projected_fill_13w_now'] * 100:.0f}% of forecast demand, and "
            f"{kp['projected_fill_13w_after_plan'] * 100:.0f}% if this week's orders are approved",
            f"container top-ups add {k(kp['container_top_up_value'])}",
        ]
    )
    own = int(sp["selection_reason"].str.startswith("SKU's own best").sum())
    e = (ets["ets_error"] * ets["windows"]).sum() / ets["windows"].sum()
    c = (ets["champion_error"] * ets["windows"]).sum() / ets["windows"].sum()
    check(
        [
            f"{own} of {len(sp)} SKUs run on their own SKU-level pick",
            f"scaled error of {e:.3f} against the champions' {c:.3f} ({int(ets['windows'].sum()):,} windows, {len(ets)} SKUs)",
        ]
    )
    amz = meta["amazon_never_sold"]
    q, lg = amz["qstats"], amz["legacy_30d"]
    check(
        [
            f"{q['never_sold']} of {q['fba_enabled']} Amazon-enabled SKUs never sold on Amazon after the fork in the QStats "
            f"world ({lg['never_sold']} in the legacy world); {q['never_sold_with_demand']} of them had Amazon demand, "
            f"{q['unserved_units']:,.0f} units",
        ]
    )
