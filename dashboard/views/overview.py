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

risk = k["skus_at_stockout_risk"]
lede = (
    f"{risk} of {k['skus_monitored']} SKUs run out within {k['stockout_risk_weeks']} weeks unless something changes. "
    f"This week's plan buys {theme.money(k['purchase_value'])} across {k['purchase_lines']} purchase lines and lifts the "
    f"demand-weighted chance of covering lead-time demand from {theme.pct(k['service_level_now'], 0)} to "
    f"{theme.pct(k['service_level_after_plan'], 0)}. {theme.money(k['excess_inventory_value'])} sits beyond "
    f"{k['excess_weeks_of_cover']} weeks of cover."
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
        ("Contribution at risk", theme.money(k["contribution_at_risk"]), "expected shortfall over lead time"),
        ("Excess inventory", theme.money(k["excess_inventory_value"]), f"beyond {k['excess_weeks_of_cover']} weeks of cover"),
        (
            "Service level",
            f"{theme.pct(k['service_level_now'], 0)} → {theme.pct(k['service_level_after_plan'], 0)}",
            "cycle service, now → after plan",
        ),
        (
            "Forecast bias",
            theme.signed_pct(k["forecast_bias_26w"]),
            f"WAPE {theme.pct(k['forecast_wape_26w'], 0)}, last 26 weeks",
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
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], tickformat=",d")
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
    theme.note(
        "Cumulative count of SKUs whose expected stock reaches zero by each week, if nothing is ordered after this "
        "week's plan. Orders placed today arrive after the supplier's median lead time (6 to 10 weeks), so earlier "
        "stockouts can only be helped by expediting or transfers."
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
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], tickformat=",d")
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
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], tickprefix="$")
    theme.show(fig)

theme.section("How this plan compares with the current process")
meta = data.eval_meta().get("matched", {})
m = meta.get("42") or meta.get(42)
if m:
    theme.callout(
        "In a controlled replay of the same business over the past year, QStats did <b>not</b> hold less inventory at the "
        f"service level the current process delivers ({theme.pct(m['legacy_fill'], 1)} fill): it needed "
        f"{theme.signed_pct(-m['inventory_saving_pct'])} inventory there, within noise. What it changed is the operating point: "
        f"{theme.pct(m['qstats_fill'], 1)} fill, where the current rule would need "
        f"{theme.signed_pct(m['legacy_extra_inventory_pct'])} more inventory than QStats. Details and intervals on "
        "<b>Legacy vs QStats</b>."
    )
