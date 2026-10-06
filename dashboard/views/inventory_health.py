import numpy as np
import pandas as pd
import streamlit as st
import theme

import data
from qstats_planner.replenishment.recommendations import NO_DEMAND_DAILY

pal = theme.palette()
k = data.kpis()
sp = data.table("sku_plan")
prods = data.table("products")
sups = data.table("suppliers")
d = sp.merge(prods[["sku_idx", "description", "category", "supplier_id", "unit_cost"]], on="sku_idx")
d["value"] = d["on_hand"] * d["unit_cost"]  # the Inventory value tile, SKU by SKU
cover = d["weeks_of_cover"].replace(np.inf, np.nan)

theme.title("Inventory health", "Where the money sits, how long it lasts, and which suppliers the plan depends on.")
theme.strip(
    [
        ("Inventory value", theme.money(k["inventory_value"]), "on hand, all locations, at cost"),
        (
            "Median cover",
            f"{np.median(d['weeks_of_cover']):.1f} wk",
            "on hand + on order / forecast; no-forecast SKUs count as unlimited",
        ),
        ("Projected stockouts", f"{k['skus_at_stockout_risk']}", f"within {k['stockout_risk_weeks']} weeks, no new order"),
        ("Excess", theme.money(k["excess_inventory_value"]), f"on hand beyond {k['excess_weeks_of_cover']} weeks of forecast"),
        ("Inventory turns", f"{k['inventory_turns']:.1f}", "trailing 13 weeks, annualised"),
        (
            "Open POs",
            theme.money(
                (
                    data.table("purchase_orders", None, True)
                    .merge(prods[["sku", "unit_cost"]], on="sku")
                    .eval("quantity * unit_cost")
                ).sum()
            ),
            "at cost",
        ),
    ]
)

a, b = st.columns(2, gap="large")
with a:
    theme.section("Weeks of cover by SKU")
    bins = [0, 2, 4, 8, 13, 26, 52, np.inf]
    labels = ["<2", "2–4", "4–8", "8–13", "13–26", "26–52", "52+"]
    c = pd.cut(cover.fillna(999), bins, labels=labels, right=False).value_counts().reindex(labels)
    fig = theme.figure(280, y_title="SKUs")
    fig.add_bar(
        x=labels,
        y=c.values,
        marker_color=[pal["s2"]] * 4 + [pal["s1"]] * 1 + [pal["s3"]] * 2,
        showlegend=False,
        text=c.values,
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]),
        hovertemplate="%{x} weeks: %{y} SKUs<extra></extra>",
    )
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_xaxes(title=dict(text="weeks of cover (on hand + on order)"))
    fig.update_yaxes(tickformat=",d", range=[0, max(c.max(), 1) * 1.15])
    theme.show(fig)
    q = sups["quoted_lead_time_days"]
    theme.note(
        f"Blue: under 13 weeks. Green: 13–26. Amber: over {k['excess_weeks_of_cover']} weeks, the excess ceiling. "
        f"Quoted supplier lead times run {q.min():.0f}–{q.max():.0f} days. SKUs with no forecast demand count as 52+."
    )
with b:
    theme.section("Inventory value on hand, by weeks of cover")
    v = (
        d.assign(band=pd.cut(cover.fillna(999), bins, labels=labels, right=False))
        .groupby("band", observed=False)["value"]
        .sum()
        .reindex(labels)
    )
    fig = theme.figure(280)
    fig.add_bar(
        x=labels,
        y=v.values,
        marker_color=[pal["s2"]] * 4 + [pal["s1"]] + [pal["s3"]] * 2,
        showlegend=False,
        text=[theme.money(x) for x in v.values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]),
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_yaxes(tickprefix="$", range=[0, max(v.max(), 1) * 1.15])
    theme.show(fig)

theme.section("Supplier exposure")
po = data.table("purchase_orders", None, True).merge(prods[["sku", "unit_cost"]], on="sku")
po["value"] = po["quantity"] * po["unit_cost"]
g = po.groupby("supplier_id").agg(
    open_pos=("po_id", "size"), open_value=("value", "sum"), delayed=("status", lambda s: int((s == "DELAYED").sum()))
)
skus = prods.groupby("supplier_id").size().rename("skus")
risk = d.assign(r=(d["stockout_day"] >= 0) & (d["stockout_day"] < 56)).groupby("supplier_id")["r"].sum().rename("skus_at_risk")
ex = (
    sups.set_index("supplier_id")[
        ["supplier_name", "country", "quoted_lead_time_days", "lead_time_p50", "lead_time_p90", "receipts"]
    ]
    .join([skus, g, risk])
    .fillna(0)
)
st.dataframe(
    ex.reset_index(),
    hide_index=True,
    width="stretch",
    column_config={
        "quoted_lead_time_days": st.column_config.NumberColumn("quoted LT (d)", format="%.0f"),
        "lead_time_p50": st.column_config.NumberColumn("LT P50 (d)", format="%.0f"),
        "lead_time_p90": st.column_config.NumberColumn("LT P90 (d)", format="%.0f"),
        "receipts": st.column_config.NumberColumn("receipts observed", format="%d"),
        "open_value": st.column_config.NumberColumn("open PO value", format="dollar"),
        "skus_at_risk": st.column_config.NumberColumn("SKUs at risk (8 wk)", format="%d"),
    },
)
theme.note(
    "Lead-time P50/P90: Kaplan–Meier estimate from received POs, with still-open POs counted as censored "
    "(they have taken at least their age), shrunk toward the quote while receipts are few."
)

theme.section("Largest excess positions")
ex2 = d[d["excess_value"] > 0].sort_values("excess_value", ascending=False).head(15)
st.dataframe(
    pd.DataFrame(
        {
            "SKU": ex2["sku"],
            "Product": ex2["description"].str.title(),
            "Category": ex2["category"],
            "On hand": ex2["on_hand"],
            "Weekly demand": ex2["weekly_demand"],
            "Weeks on hand": ex2["on_hand"] / ex2["weekly_demand"].where(ex2["weekly_demand"] >= 7 * NO_DEMAND_DAILY),
            "Excess value": ex2["excess_value"],
            "Discontinued": ex2["discontinued"],
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "On hand": st.column_config.NumberColumn(format="localized"),
        "Weekly demand": st.column_config.NumberColumn("Weekly forecast", format="%.1f"),
        "Weeks on hand": st.column_config.NumberColumn(format="%.0f", help="Blank: no forecast demand"),
        "Excess value": st.column_config.NumberColumn(format="dollar"),
    },
)
theme.note(
    f"Excess: units on hand beyond {k['excess_weeks_of_cover']} weeks of forecast demand, at cost; the Excess tile is "
    "this column summed over every SKU. Open orders are not counted here. The Action center's EXCESS lines do "
    "count them, because delaying or cancelling an order is one of the remedies."
)
