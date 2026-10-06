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
win0 = fork + pd.Timedelta(days=float(data.table("suppliers")["quoted_lead_time_days"].max()))
win1 = pd.Timestamp(data.kpis()["plan_date"]) - pd.Timedelta(days=1)
theme.title(
    "Legacy vs QStats",
    f"The same simulated business replayed twice from the same day ({fork:%d %b %Y}): once with the current planning "
    "process, once with QStats. Same products, same real demand pattern, same supplier delays. Scored on the same "
    f"{int(Q['skus'])} SKUs (those without simulated promotions) over {win0:%d %b} – {win1:%d %b %Y}, after QStats's "
    "first orders had time to arrive.",
)

lo, hi = b0["inventory_saving_pct"].quantile([0.05, 0.95])
theme.callout(
    f"<b>At today's service level, no inventory saving.</b> To deliver the current process's fill rate "
    f"({theme.pct(m['legacy_fill'])}), QStats needs {theme.signed_pct(-m['inventory_saving_pct'])} inventory "
    f"(90% interval {theme.signed_pct(-hi)} to {theme.signed_pct(-lo)}; other replicate worlds "
    f"{', '.join(theme.signed_pct(-meta[s]['inventory_saving_pct']) for s in meta if s != str(seed))}). "
    "<b>Where it differs is what more service costs:</b> "
    f"QStats runs at {theme.pct(m['qstats_fill'])} fill, and the current rule "
    f"would need {theme.signed_pct(m['legacy_extra_inventory_pct'])} more inventory than QStats to get there. "
    "This was the pre-registered test, run once (docs/EVAL_PLAN.md)."
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
st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch", height=36 * (len(out) + 1) + 4)
theme.note(
    "✓ / ✗ mark which planner did better on each line; blank where better or worse depends on the business "
    "(purchases, ending position). Legacy holds less inventory and loses more sales; whether that trade is right "
    "depends on margins, which is why the frontier below matters more than any single row."
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
    textposition="top left",
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
                f"{'reconstruction' if u else '—'} · {'prior' if sv else '—'} · {'probabilistic SS' if p else '30-day SS'}"
            )
ab = eff.reindex(list(arms)).rename(index=arms).astype(float) * 100
ab = ab[[seed] + [c for c in ab.columns if c != seed]]
ab.columns = [f"world {c}" + (" (reference)" if c == seed else "") for c in ab.columns]
st.dataframe(
    ab.reset_index().rename(columns={"variant": "QStats ingredients (reconstruction · seasonal prior · safety stock)"}),
    hide_index=True,
    width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%+.1f%%") for c in ab.columns},
)
theme.note(
    "Each cell: inventory the Legacy frontier needs at that arm's fill rate ÷ the arm's inventory − 1. Positive = less "
    "stock for the same service. Empty = the arm's fill rate is outside the Legacy frontier. Probabilistic safety "
    "stock learned from censored sales is worse than the 30-day rule: stockout correction has to come first."
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
    for sc, label in (("scenario_null", "honest quotes, no events"), ("scenario_optimistic_quotes", "optimistic quotes")):
        try:
            t = data.eval_table(sc)
            t = t[t["subset"] == "headline"].set_index("variant")
            mm = matched_comparison({v: dict(r) for v, r in t.iterrows()}, LEGACY_REF, LEGACY_FAMILY, QSTATS_FAMILY)
            sav = mm["inventory_saving_pct"]
            theme.note(
                f"{label.capitalize()}: at the current process's fill rate QStats needs "
                f"{theme.signed_pct(-sav) if np.isfinite(sav) else 'n/a (outside its frontier)'} inventory; "
                f"to reach QStats's fill the current rule needs {theme.signed_pct(mm['legacy_extra_inventory_pct'])} more."
            )
        except Exception:
            pass

theme.section("Week by week, both worlds")
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
planner_rows = [
    ("Demand history", "Sales as recorded", "Sales, with stockout days reconstructed (local level × weekday × season)"),
    (
        "Forecast",
        f"Simple exponential smoothing, α = {cfg['legacy']['ses_alpha']} (tuned on dev SKUs), closed weeks skipped",
        "Champion per segment among 25 candidates, by rolling-origin error of lead-time demand",
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
