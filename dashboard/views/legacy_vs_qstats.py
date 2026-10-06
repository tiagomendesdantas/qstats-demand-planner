import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import theme

import data
from qstats_planner.evaluation.comparison import efficiency_vs_legacy, matched_comparison
from qstats_planner.simulation.runner import LEGACY_FAMILY, LEGACY_REF, QSTATS_FAMILY, QSTATS_REF

pal = theme.palette()
cfg = data.cfg()
seed = cfg["random_seed"]
summ = data.eval_table("summary")
head = summ[summ["subset"] == "headline"]
ref = head[head["seed"] == seed].set_index("variant")
L, Q = ref.loc[LEGACY_REF], ref.loc[QSTATS_REF]
meta = data.eval_meta()["matched"]
m = meta[str(seed)]
boot = data.eval_table("bootstrap")
b0 = boot[boot["seed"] == seed]

man = json.loads((Path(data.ROOT) / cfg["paths"]["processed_dir"] / "manifest.json").read_text())
first = pd.Timestamp(man["shifted_date_range"][0])
fork = first - pd.Timedelta(days=first.dayofweek) + pd.Timedelta(weeks=cfg["simulation"]["fork_week"] - 1)
review = cfg["simulation"]["review_period_days"]
win0 = fork + pd.Timedelta(days=review + float(data.table("suppliers")["quoted_lead_time_days"].max()))
win1 = pd.Timestamp(data.kpis()["plan_date"]) - pd.Timedelta(days=1)
scored_days = (win1 - win0).days + 1
theme.title(
    "Legacy vs QStats",
    f"The same simulated business replayed twice from the same day ({fork:%d %b %Y}): once with the current planning "
    "process, once with QStats. Same products, same real demand pattern, same supplier delays. Scored on the same "
    f"{int(Q['skus'])} SKUs (those without a simulated promotion or liquidation) over {win0:%d %b} – {win1:%d %b %Y}, "
    "after QStats's first orders had time to arrive.",
)

lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
slo, shi = b0["legacy_extra_inventory_pct"].quantile([0.05, 0.95])


def more_less(v: float) -> str:
    """A saving share in words: positive saving = less inventory."""
    return f"{abs(v) * 100:.1f}% {'less' if v > 0 else 'more'}" if np.isfinite(v) else "out of range"


def more_less_extra(v: float) -> str:
    """The inventory the current rule needs beyond QStats's: positive = more."""
    return f"{abs(v) * 100:.1f}% {'more' if v > 0 else 'less'}" if np.isfinite(v) else "out of range"


sav, extra = m["inventory_saving_pct"], m["legacy_extra_inventory_pct"]
vd = data.verdicts()
v1, v2 = vd["primary"], vd["secondary"]
others = [s_ for s_ in meta if s_ != str(seed)]
oor = m.get("bootstrap_out_of_range_primary", float("nan"))
oor2 = m.get("bootstrap_out_of_range_secondary", float("nan"))
other_sec = ", ".join(more_less_extra(meta[s_]["legacy_extra_inventory_pct"]) for s_ in others)
other_prim = ", ".join(more_less(meta[s_]["inventory_saving_pct"]) for s_ in others)
below = m.get("bootstrap_out_of_range_primary_below", float("nan"))
dominated = m.get("bootstrap_out_of_range_primary_dominated", float("nan"))
excluded = (
    f"{oor * 100:.1f}% of resamples fell outside QStats's frontier and are left out"
    + (
        f"; in all of them QStats's lowest setting already delivered more fill, and in {dominated * 100:.1f}% of all "
        "resamples it did so with no more inventory"
        if np.isfinite(below) and oor > 0 and abs(below - oor) < 1e-9
        else ""
    )
)
top = vd["legacy_top"]
if np.isfinite(extra):
    second = (
        f"QStats runs at {theme.pct(m['qstats_fill'])} fill, where the current rule would need {more_less_extra(extra)} "
        f"inventory than QStats (90% interval {theme.signed_pct(slo)} to {theme.signed_pct(shi)}; {oor2 * 100:.1f}% of "
        f"resamples out of range; {other_sec} in the other worlds)."
    )
else:
    second = (
        f"QStats runs at {theme.pct(m['qstats_fill'])} fill, above the {theme.pct(top['fill'])} the current rule "
        f"reached at its highest setting ({top['days']} days of safety stock), on "
        f"{(1 - vd['qstats_inventory'] / top['inventory']) * 100:.1f}% less inventory than that setting. The "
        f"pre-registered reading does not extrapolate, so the secondary metric is out of range here (and in "
        f"{oor2 * 100:.1f}% of resamples). Other replicate worlds: {other_sec}."
    )
theme.callout(
    f"<b>{v1}</b> To deliver the current process's fill rate ({theme.pct(m['legacy_fill'])}), QStats needs "
    f"{more_less(sav)} inventory (90% interval: from {more_less(hi)} to {more_less(lo)}; {excluded}). Other "
    f"replicate worlds: {other_prim}. <b>{v2}</b> {second} The test was fixed in advance; every correction since is "
    "logged with the numbers before and after (docs/EVAL_PLAN.md)."
)


def net_vs_legacy(x) -> float:
    """Contribution recovered + cross-DC shipping saved - extra carrying cost, against Legacy-30."""
    return (
        (L["lost_contribution"] - x["lost_contribution"])
        + (L["cross_dc_cost"] - x["cross_dc_cost"])
        - (x["holding_cost"] - L["holding_cost"])
    )


money = net_vs_legacy(Q)
hold = cfg["business"]["holding_cost_annual_pct"]
theme.strip(
    [
        ("Contribution recovered", theme.money(L["lost_contribution"] - Q["lost_contribution"]), "fewer lost sales"),
        ("Cross-DC shipping saved", theme.money(L["cross_dc_cost"] - Q["cross_dc_cost"]), "fewer split shipments"),
        (
            "Extra carrying cost",
            theme.money(Q["holding_cost"] - L["holding_cost"]),
            f"{hold:.0%} a year, over {scored_days} days",
        ),
        ("Net over the scored days", theme.money(money), "before end-of-period stock"),
        (
            "Extra excess at the end",
            theme.money(Q["excess_inventory_value"] - L["excess_inventory_value"]),
            f"beyond {cfg['inventory']['excess_weeks_of_cover']} weeks of demand",
        ),
    ]
)
lfam = ref.loc[[v for v in LEGACY_FAMILY if v in ref.index]]
near = (lfam["fill_rate"] - Q["fill_rate"]).abs().idxmin()
LN = lfam.loc[near]
theme.note(
    f"In plain terms: QStats delivered {theme.pct(Q['fill_rate'])} fill with {theme.money(Q['average_inventory_value'])} "
    f"of average inventory. The current process's nearest setting, {near.removeprefix('legacy_').removesuffix('d')} days "
    f"of safety stock, delivered {theme.pct(LN['fill_rate'])} with {theme.money(LN['average_inventory_value'])}, and "
    f"would net {theme.money(net_vs_legacy(LN))} against Legacy-30 on the same terms. Part of any net figure comes "
    "from running at a different service level; the frontier below, not this strip, separates the method from that."
)

rows = [
    ("Forecast WAPE (1 week)", "wape", "pct", -1),
    ("Forecast bias", "forecast_bias", "spct", 0),
    ("Fill rate (service level)", "fill_rate", "pct", 1),
    ("In-stock rate", "in_stock_rate", "pct", 1),
    ("Stockout channel-days", "stockout_days", "int", -1),
    ("Lost units", "lost_units", "int", -1),
    ("Lost contribution", "lost_contribution", "money", -1),
    ("Average inventory", "average_inventory_value", "money", -1),
    ("Inventory turns", "inventory_turns", "x", 1),
    ("Excess inventory at the end", "excess_inventory_value", "money", -1),
    ("Purchase value (from the fork)", "purchase_value", "money", 0),
    ("Working capital", "working_capital", "money", -1),
    ("Ending position (on hand + on order)", "ending_position_value", "money", 0),
    ("Cross-DC shipping cost", "cross_dc_cost", "money", -1),
]
fmt = {"pct": theme.pct, "spct": theme.signed_pct, "int": theme.units, "money": theme.money, "x": lambda v: f"{v:.2f}"}
out = []
for label, col, f, better in rows:
    lv, qv = L[col], Q[col]
    if col == "forecast_bias":
        verdict = "✓" if abs(qv) < abs(lv) else "✗"
    elif better == 0:
        verdict = ""
    else:
        verdict = "✓" if (qv - lv) * better > 0 else "✗"
    diff = (
        f"{(qv - lv) * 100:+.1f} pts"
        if f in ("pct", "spct")
        else theme.money(qv - lv)
        if f == "money"
        else f"{qv - lv:+,.0f}"
        if f == "int"
        else f"{qv - lv:+.2f}"
    )
    out.append({"Metric": label, "Legacy": fmt[f](lv), "QStats": fmt[f](qv), "Difference": diff, "QStats better": verdict})
theme.section("Outcomes, reference world")
st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch", height="content")
holds = "more" if Q["average_inventory_value"] > L["average_inventory_value"] else "less"
loses = "fewer" if Q["lost_units"] < L["lost_units"] else "more"
theme.note(
    "✓ / ✗ mark which planner did better on each line; blank where better or worse depends on the business "
    f"(purchases, ending position). Here QStats holds {holds} inventory and loses {loses} sales; whether that trade "
    "is right depends on margins, which is why the frontier below matters more than any single row."
)

theme.section("Service vs inventory: each planner at several settings")
pts = head[head["seed"] == seed].set_index("variant")
fig = theme.figure(380, y_title="fill rate", x_title="average inventory at cost")
lf = pts.loc[LEGACY_FAMILY].sort_values("average_inventory_value")
qf = pts.loc[QSTATS_FAMILY].sort_values("average_inventory_value")
lab = lambda v: (
    v.replace("legacy_", "").replace("qstats_sl", "").replace("qstats", "targets") + ("%" if v.startswith("qstats_sl") else "")
)  # noqa: E731
fig.add_scatter(
    x=lf["average_inventory_value"],
    y=lf["fill_rate"],
    name="Legacy, days of safety stock",
    mode="lines+markers+text",
    line=dict(color=pal["s2"], width=2),
    marker=dict(size=9, color=pal["s2"], line=dict(width=2, color=pal["surface"])),
    text=[lab(v) for v in lf.index],
    textposition="bottom right",
    textfont=dict(size=11, color=pal["ink2"]),
    hovertemplate="$%{x:,.0f} · %{y:.1%}",
)
fig.add_scatter(
    x=qf["average_inventory_value"],
    y=qf["fill_rate"],
    name="QStats, service target",
    mode="lines+markers+text",
    line=dict(color=pal["s1"], width=2),
    marker=dict(size=9, color=pal["s1"], line=dict(width=2, color=pal["surface"])),
    text=[lab(v) for v in qf.index],
    # the class-target point sits next to the uniform 95% one: label it below so the two stay legible
    textposition=["bottom right" if v == QSTATS_REF else "top left" for v in qf.index],
    textfont=dict(size=11, color=pal["ink2"]),
    hovertemplate="$%{x:,.0f} · %{y:.1%}",
)
lp = pts.loc["legacy_30d_prior"]
fig.add_scatter(
    x=[lp["average_inventory_value"]],
    y=[lp["fill_rate"]],
    name="Legacy 30d + seasonal prior",
    mode="markers",
    marker=dict(size=11, color=pal["s3"], symbol="diamond", line=dict(width=2, color=pal["surface"])),
    hovertemplate="$%{x:,.0f} · %{y:.1%}",
)
fig.update_xaxes(tickprefix="$", showgrid=True, gridcolor=pal["grid"])
fig.update_yaxes(tickformat=".0%")
fig.update_layout(hovermode="closest")
theme.show(fig)
theme.note(
    "Up and to the left is better: more fill for the same inventory. Read the vertical gap between the curves at a "
    "given inventory, or the horizontal gap at a given fill rate. The amber point is the current process with only "
    "the seasonal prior added."
)

theme.section("Which ingredient does the work (2 × 2 × 2 ablation)")
eff_rows = []
for s_, g in head.groupby("seed"):
    sm = {v: r for v, r in g.set_index("variant").iterrows()}
    eff = efficiency_vs_legacy({k: dict(v) for k, v in sm.items()}, LEGACY_FAMILY)
    for v, e in eff.items():
        eff_rows.append({"variant": v, "seed": s_, "eff": e})
eff = pd.DataFrame(eff_rows).pivot(index="variant", columns="seed", values="eff")
arms = {"legacy_30d_prior": "Legacy + seasonal prior"}
for u in (0, 1):
    for sv in (0, 1):
        for p in (0, 1):
            name = "qstats" if (u, sv, p) == (1, 1, 1) else f"ablation_u{u}s{sv}p{p}"
            arms[name] = (
                f"{'reconstruction' if u else '—'} · {'prior' if sv else '—'} · "
                f"{'probabilistic SS, KM lead times, QStats FBA rule' if p else '30-day SS, quoted lead time, legacy FBA rule'}"
            )
ab = eff.reindex(list(arms)).rename(index=arms).astype(float) * 100
ab = ab[[seed] + [c for c in ab.columns if c != seed]]
ab.columns = [f"world {c}" + (" (reference)" if c == seed else "") for c in ab.columns]
st.dataframe(
    ab.reset_index().rename(
        columns={"variant": "QStats ingredients (reconstruction · seasonal prior · safety stock, lead times, FBA rule)"}
    ),
    hide_index=True,
    width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%+.1f%%") for c in ab.columns},
)
theme.note(
    "Each cell: inventory the Legacy frontier needs at that arm's fill rate ÷ the arm's inventory − 1. Positive = less "
    "stock for the same service. None = the arm's fill rate is outside the Legacy frontier. The arm with no QStats "
    "ingredient is QStats's forecaster with the 30-day rule, on raw sales, with 13 candidate models (the prior "
    "variants drop out with the prior). The third switch changes more than safety stock: it also moves from quoted "
    "to Kaplan–Meier lead times, from the legacy FBA rule to QStats's, and counts Amazon RESERVED units at 0.5 "
    "instead of 1.0."
)

sm95 = {s_: (g.set_index("variant").loc["qstats_sl95"], g.set_index("variant").loc["qstats"]) for s_, g in head.groupby("seed")}
wins95 = sum(
    int(a["fill_rate"] >= b["fill_rate"] and a["average_inventory_value"] <= b["average_inventory_value"])
    for a, b in sm95.values()
)
prior_wins, prior_comparable = vd["prior_wins"], vd["prior_comparable"]


def span(arm: str) -> str:
    """An arm's efficiency across the worlds, in words."""
    v = eff.loc[arm].dropna() * 100
    if v.empty:
        return "below the Legacy frontier in every world"
    rng = f"{v.iloc[0]:+.1f}%" if len(v) == 1 else f"{v.min():+.1f}% to {v.max():+.1f}%"
    gone = eff.shape[1] - len(v)
    return rng + (f" (out of range in {gone} of {eff.shape[1]} worlds)" if gone else "")


alone = {
    "reconstruction alone scores": "ablation_u1s0p0",
    "the prior alone": "ablation_u0s1p0",
    "probabilistic safety stock (with its lead times and FBA rule) alone": "ablation_u0s0p1",
    "the forecaster with none of them": "ablation_u0s0p0",
}
classes = ", ".join(f"{c} {v:.0%}" for c, v in cfg["products"]["service_level_by_class"].items())
theme.callout(
    f"<b>Read the ablation plainly.</b> The current process with only the seasonal prior added matches or beats full "
    f"QStats in {prior_wins} of the {prior_comparable} worlds where both can be read off the Legacy frontier. On "
    "the same efficiency scale, "
    + "; ".join(f"{k} {span(v)}" for k, v in alone.items())
    + f". QStats's own class targets ({classes}) against a uniform 95%: the uniform target gave more fill with less "
    f"stock in {wins95} of {len(sm95)} worlds. The class targets were fixed before the replay and are kept."
)

a, b = st.columns(2, gap="large")
with a:
    theme.section("Replicate worlds (different supplier luck)")
    rw = pd.DataFrame(
        [
            {
                "world": s_,
                "Legacy fill": head[(head.seed == int(s_)) & (head.variant == LEGACY_REF)]["fill_rate"].iloc[0],
                "QStats fill": v["qstats_fill"],
                "saving at Legacy's fill": v["inventory_saving_pct"],
                "Legacy extra at QStats's fill": v["legacy_extra_inventory_pct"],
            }
            for s_, v in meta.items()
        ]
    )
    st.dataframe(
        rw,
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in rw.columns[1:]},
    )
with b:
    theme.section("Sensitivity worlds (reference seed)")
    srows = []
    for sc, label in (
        ("scenario_null", "Lead time = quote, no events"),
        ("scenario_optimistic_quotes", "Quotes at P25 of true lead time"),
    ):
        try:
            t = data.eval_table(sc)
            t = t[t["subset"] == "headline"].set_index("variant")
            srows.append(
                {
                    "world": label,
                    "Legacy fill": t.at[LEGACY_REF, "fill_rate"],
                    "QStats fill": t.at[QSTATS_REF, "fill_rate"],
                    "Legacy inventory": t.at[LEGACY_REF, "average_inventory_value"],
                    "QStats inventory": t.at[QSTATS_REF, "average_inventory_value"],
                }
            )
        except Exception:
            pass
    if srows:
        st.dataframe(
            pd.DataFrame(srows),
            hide_index=True,
            width="stretch",
            column_config={
                "Legacy fill": st.column_config.NumberColumn(format="percent"),
                "QStats fill": st.column_config.NumberColumn(format="percent"),
                "Legacy inventory": st.column_config.NumberColumn(format="dollar"),
                "QStats inventory": st.column_config.NumberColumn(format="dollar"),
            },
        )
    for sc, label in (
        ("scenario_null", "Lead time = quote, no events"),
        ("scenario_optimistic_quotes", "Quotes at P25 of true lead time"),
    ):
        try:
            t = data.eval_table(sc)
            t = t[t["subset"] == "headline"].set_index("variant")
            mm = matched_comparison({v: dict(r) for v, r in t.iterrows()}, LEGACY_REF, LEGACY_FAMILY, QSTATS_FAMILY)
            s1, s2 = mm["inventory_saving_pct"], mm["legacy_extra_inventory_pct"]
            q_low = t.loc[QSTATS_FAMILY, "fill_rate"].min()
            l_high = t.loc[LEGACY_FAMILY, "fill_rate"].max()
            first = (
                f"QStats needs {more_less(s1)} inventory"
                if np.isfinite(s1)
                else f"no reading: Legacy-30's fill ({theme.pct(mm['legacy_fill'])}) is below QStats's lowest setting "
                f"({theme.pct(q_low)})"
            )
            second = (
                f"the current rule needs {more_less_extra(s2)} inventory than QStats"
                if np.isfinite(s2)
                else f"no reading: QStats's fill ({theme.pct(mm['qstats_fill'])}) is above the current rule's highest "
                f"setting ({theme.pct(l_high)})"
            )
            theme.note(f"{label}: at the current process's fill rate, {first}; at QStats's fill, {second}.")
        except Exception:
            pass

theme.section("Week by week, both worlds (headline SKUs)")
ww = data.eval_table("world_weekly")
ww["week"] = pd.to_datetime(ww["week"])
ww = ww[ww["week"] >= fork]
fig = theme.figure(280, y_title="fill rate")
for w, color, name in ((LEGACY_REF, pal["s2"], "Legacy"), (QSTATS_REF, pal["s1"], "QStats")):
    s_ = ww[ww["world"] == w]
    fig.add_scatter(
        x=s_["week"],
        y=s_["sold"] / s_["demand"].replace(0, np.nan),
        name=name,
        line=dict(color=color, width=2),
        hovertemplate="%{y:.1%}",
    )
fig.add_vrect(
    x0=fork,
    x1=win0,
    fillcolor=pal["shade"],
    line_width=0,
    layer="below",
    annotation_text="transition (not scored)",
    annotation_position="top left",
    annotation_font=dict(size=11, color=pal["muted"]),
)
fig.update_yaxes(tickformat=".0%")
theme.show(fig)

theme.section("How the two planners are built")
METHOD_TEXT = {
    "pre_post_velocity": "the sales rate on clean days around the stockout",
    "local_profile": "local level × weekday × season",
    "model_expectation": "the level a smoothing model expected at the stockout",
    "censored_gamma": "local level × weekday × season, with sold-out days treated as a lower bound on demand",
}
fc = cfg["forecasting"]
own = data.table("sku_plan")["selection_reason"].str.startswith("SKU's own best")
planner_rows = [
    (
        "Demand history",
        "Sales as recorded",
        f"Sales, with stockout days reconstructed ({METHOD_TEXT[cfg['reconstruction']['planner_method']]})",
    ),
    (
        "Forecast",
        f"Simple exponential smoothing, α = {cfg['legacy']['ses_alpha']} (tuned on dev SKUs), closed weeks skipped",
        "Champion per segment among 25 candidates by rolling-origin error of lead-time demand; a SKU keeps its own "
        f"pick when it beats its segment's by {fc['sku_override_margin']:.0%} on {fc['sku_override_min_blocks']}+ "
        f"non-overlapping windows ({int(own.sum())} of {len(own)} SKUs in this week's plan)",
    ),
    ("Seasonality", "None", "Pooled monthly prior from other products (two groups), fixed before the replay"),
    ("Lead time", "Supplier's quote, as if certain", "Kaplan–Meier from receipts, open POs censored"),
    (
        "Safety stock",
        f"{cfg['legacy']['safety_days']} days of forecast demand",
        "Service-level quantile of demand over lead time + review (mixture of lead-time and forecast-error distributions)",
    ),
    ("Review, MOQ, case pack", "Weekly; same rounding", "Weekly; same rounding"),
    (
        "Amazon FBA",
        "Days of cover: trigger at pick + transit + review + 14 days, send to + 30 days",
        "Service-level quantile over the replenishment window, by inventory state",
    ),
]
st.markdown(
    "| | Legacy (the current process) | QStats |\n|---|---|---|\n"
    + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in planner_rows)
    + "\n\nThe legacy process is not a straw man: same review cadence, same rounding, same visibility of open POs, "
    "its smoothing constant tuned the same way, and its FBA rule counts pick, transit and review time."
)
