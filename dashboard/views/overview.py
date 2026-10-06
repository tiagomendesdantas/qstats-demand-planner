import numpy as np
import pandas as pd
import streamlit as st
import theme

import data

pal = theme.palette()
k = data.kpis()
recs = data.recommendations()
sp = data.table("sku_plan")
prods = data.table("products")
sp = sp.merge(prods[["sku_idx", "category", "unit_cost"]], on="sku_idx")

cfg = data.cfg()
seed = cfg["random_seed"]
risk = k["skus_at_stockout_risk"]
risk_days = k["stockout_risk_weeks"] * 7
so_new = sp["stockout_day_with_order"].to_numpy()
helped = int(risk - ((so_new >= 0) & (so_new < risk_days)).sum())
cal = data.eval_table("calibration").query("seed == @seed")
p90 = float((cal["realised"] <= cal["q90"]).mean()) if len(cal) else float("nan")
narrow = f" (optimistic: the model's P90 covered {theme.pct(p90, 0)} of outcomes in the replay)" if p90 < 0.88 else ""
lede = (
    "QStats's plan for the business as the current process left it. "
    f"{risk} of {k['skus_monitored']} SKUs are projected to run out within {k['stockout_risk_weeks']} weeks; "
    + (
        f"this week's orders land in time for {helped} of them; for the rest, expediting or transfers are the levers. "
        if helped
        else "orders placed today land after all of them, so expediting and transfers are the levers there. "
    )
    + f"The plan has {k['purchase_lines']} purchase lines worth {theme.money(k['purchase_value'])}, "
    f"{k['purchase_lines_for_review']} of them awaiting review. If all are approved, the plan's own estimate of the "
    "demand-weighted chance of covering demand over lead time + review goes from "
    f"{theme.pct(k['service_level_now'], 0)} to {theme.pct(k['service_level_after_plan'], 0)}{narrow}. "
    f"{theme.money(k['excess_inventory_value'])} of stock on hand sits beyond {k['excess_weeks_of_cover']} weeks of "
    "forecast demand."
)
theme.title("Executive overview", lede)

theme.strip(
    [
        ("SKUs monitored", f"{k['skus_monitored']}", "2 DCs + Amazon FBA"),
        ("Inventory value", theme.money(k["inventory_value"]), f"{theme.units(k['inventory_units'])} units at cost"),
        (
            "Purchase recommendations",
            f"{k['purchase_lines']}",
            f"{theme.money(k['purchase_value'])} at cost · {k['purchase_lines_for_review']} to review",
        ),
        ("SKUs at stockout risk", f"{risk}", f"out within {k['stockout_risk_weeks']} weeks, no new order"),
        ("Contribution at risk", theme.money(k["contribution_at_risk"]), "expected shortfall, lead time + review"),
        ("Excess inventory", theme.money(k["excess_inventory_value"]), f"on hand beyond {k['excess_weeks_of_cover']} weeks"),
        (
            "Modelled service",
            f"{theme.pct(k['service_level_now'], 0)} → {theme.pct(k['service_level_after_plan'], 0)}",
            "plan's own estimate, now → after plan",
        ),
        (
            "Forecast bias",
            theme.signed_pct(k["forecast_bias_26w"]),
            f"WAPE {theme.pct(k['forecast_wape_26w'], 0)}, champions' backtest, 26 weeks",
        ),
    ]
)

left, right = st.columns(2, gap="large")

with left:
    theme.section("Recommendations by action")
    c = recs[recs["action"] != "NO_ACTION"]["action"].value_counts().sort_values()
    fig = theme.figure(300)
    fig.add_bar(
        y=c.index,
        x=c.values,
        orientation="h",
        marker_color=pal["s1"],
        text=c.values,
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]),
        hovertemplate="%{y}: %{x}<extra></extra>",
        showlegend=False,
    )
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], tickformat=",d", range=[0, c.max() * 1.12])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    theme.show(fig)

with right:
    theme.section("SKUs projected to run out, next 13 weeks")
    weeks = np.arange(0, 13 * 7, 7)
    so0 = sp["stockout_day"].to_numpy()
    so1 = sp["stockout_day_with_order"].to_numpy()
    plan_date = pd.Timestamp(k["plan_date"])
    x = [plan_date + pd.Timedelta(days=int(d)) for d in weeks]
    n0 = [int(((so0 >= 0) & (so0 <= d + 6)).sum()) for d in weeks]
    n1 = [int(((so1 >= 0) & (so1 <= d + 6)).sum()) for d in weeks]
    fig = theme.figure(300, y_title="SKUs projected out")
    fig.add_scatter(x=x, y=n0, name="Current trajectory", line=dict(color=pal["s2"], width=2), mode="lines")
    fig.add_scatter(x=x, y=n1, name="With this week's plan", line=dict(color=pal["s1"], width=2), mode="lines")
    fig.update_yaxes(tickformat=",d", rangemode="tozero")
    theme.show(fig)
    lt50 = data.table("suppliers")["lead_time_p50"] / 7
    theme.note(
        "Cumulative count of SKUs whose expected stock reaches zero by each week, if nothing is ordered after this "
        f"week's plan. Orders placed today arrive after the supplier's median lead time ({lt50.min():.0f} to "
        f"{lt50.max():.0f} weeks), so earlier stockouts can only be helped by expediting or transfers."
    )

left, right = st.columns(2, gap="large")
with left:
    theme.section("Stockout risk by category")
    cat = sp.assign(risk=(sp["stockout_day"] >= 0) & (sp["stockout_day"] < k["stockout_risk_weeks"] * 7))
    g = cat.groupby("category").agg(at_risk=("risk", "sum"), skus=("risk", "size")).sort_values("at_risk")
    fig = theme.figure(300)
    fig.add_bar(
        y=g.index,
        x=g["at_risk"],
        orientation="h",
        marker_color=pal["s2"],
        showlegend=False,
        text=[f"{a} of {b}" for a, b in zip(g["at_risk"], g["skus"], strict=True)],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]),
        hovertemplate="%{y}: %{x} SKUs at risk<extra></extra>",
    )
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], tickformat=",d", range=[0, max(g["at_risk"].max(), 1) * 1.3])
    theme.show(fig)

with right:
    theme.section("Inventory investment by category")
    v = pd.Series(k["inventory_value_by_category"]).sort_values()
    fig = theme.figure(300)
    fig.add_bar(
        y=v.index,
        x=v.values,
        orientation="h",
        marker_color=pal["s1"],
        showlegend=False,
        text=[theme.money(x) for x in v.values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]),
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
    )
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], tickprefix="$", range=[0, v.max() * 1.2])
    theme.show(fig)

theme.section("How this plan compares with the current process")
vd = data.verdicts()
m = vd["m"]
sav, extra = m["inventory_saving_pct"], m["legacy_extra_inventory_pct"]


def more_less(v: float, positive: str, negative: str) -> str:
    return f"{abs(v) * 100:.1f}% {positive if v > 0 else negative}"


at_legacy = (
    f"QStats needed {more_less(sav, 'less', 'more')} inventory there"
    if np.isfinite(sav)
    else "QStats's settings did not reach that fill rate"
)
top = vd["legacy_top"]
at_qstats = (
    f"QStats ran the business at {theme.pct(m['qstats_fill'], 1)} fill, where the current rule would need "
    f"{more_less(extra, 'more', 'less')} inventory than QStats"
    if np.isfinite(extra)
    else f"QStats ran the business at {theme.pct(m['qstats_fill'], 1)} fill, above the "
    f"{theme.pct(top['fill'], 1)} the current rule reached at its highest setting ({top['days']} days of safety "
    f"stock), on {(1 - vd['qstats_inventory'] / top['inventory']) * 100:.1f}% less inventory than that setting"
)
others = vd["secondary_other_worlds"]
replicates = (
    " In the replicate worlds where it can be read, the current rule needed "
    + " and ".join(more_less(v, "more", "less") for v in others)
    + " inventory than QStats to reach QStats's fill."
    if others
    else ""
)
theme.callout(
    f"In a controlled replay of the same business over the past year: <b>{vd['primary']}</b> At the current process's "
    f"fill ({theme.pct(m['legacy_fill'], 1)}), {at_legacy}. <b>{vd['secondary']}</b> {at_qstats}.{replicates} "
    "The current process with only a seasonal prior added did as well or better than full QStats in "
    f"{vd['prior_wins']} of the {vd['prior_comparable']} worlds where both could be read. Details on "
    "<b>Legacy vs QStats</b>."
)
