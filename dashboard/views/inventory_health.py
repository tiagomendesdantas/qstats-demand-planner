import numpy as np
import pandas as pd
import streamlit as st

import data
import theme

pal = theme.palette()
k = data.kpis()
sp = data.table("sku_plan")
prods = data.table("products")
sups = data.table("suppliers")
d = sp.merge(prods[["sku_idx", "description", "category", "supplier_id", "unit_cost"]], on="sku_idx")
d["value"] = d["stock_now"] * d["unit_cost"]
cover = d["weeks_of_cover"].replace(np.inf, np.nan)

theme.title("Inventory health", "Where the money sits, how long it lasts, and which suppliers the plan depends on.")
theme.strip([
    ("Inventory value", theme.money(k["inventory_value"]), "on hand, all locations, at cost"),
    ("Median cover", f"{np.nanmedian(cover):.1f} wk", "on hand + on order / forecast"),
    ("Projected stockouts", f"{k['skus_at_stockout_risk']}", f"within {k['stockout_risk_weeks']} weeks, no new order"),
    ("Excess", theme.money(k["excess_inventory_value"]), f"beyond {k['excess_weeks_of_cover']} weeks of cover"),
    ("Inventory turns", f"{k['inventory_turns']:.1f}", "trailing 13 weeks, annualised"),
    ("Open POs", theme.money((data.table("purchase_orders", None, True).merge(prods[["sku", "unit_cost"]], on="sku")
                              .eval("quantity * unit_cost")).sum()), "at cost"),
])

a, b = st.columns(2, gap="large")
with a:
    theme.section("Weeks of cover by SKU")
    bins = [0, 2, 4, 8, 13, 26, 52, np.inf]
    labels = ["<2", "2–4", "4–8", "8–13", "13–26", "26–52", "52+"]
    c = pd.cut(cover.fillna(999), bins, labels=labels, right=False).value_counts().reindex(labels)
    fig = theme.figure(280, y_title="SKUs")
    fig.add_bar(x=labels, y=c.values, marker_color=[pal["s2"]] * 4 + [pal["s1"]] * 1 + [pal["s3"]] * 2, showlegend=False,
                text=c.values, textposition="outside", textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]),
                hovertemplate="%{x} weeks: %{y} SKUs<extra></extra>")
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_xaxes(title=dict(text="weeks of cover (on hand + on order)"))
    fig.update_yaxes(tickformat=",d")
    theme.show(fig)
    theme.note("Blue: under 13 weeks (shorter than most supplier lead times). Green: 13–26. Amber: over 26 weeks, flagged as excess.")
with b:
    theme.section("Inventory value by weeks of cover")
    v = d.assign(band=pd.cut(cover.fillna(999), bins, labels=labels, right=False)).groupby("band", observed=False)["value"].sum().reindex(labels)
    fig = theme.figure(280)
    fig.add_bar(x=labels, y=v.values, marker_color=[pal["s2"]] * 4 + [pal["s1"]] + [pal["s3"]] * 2, showlegend=False,
                text=[theme.money(x) for x in v.values], textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=12, color=pal["ink2"]), hovertemplate="%{x}: $%{y:,.0f}<extra></extra>")
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_yaxes(tickprefix="$")
    theme.show(fig)

theme.section("Supplier exposure")
po = data.table("purchase_orders", None, True).merge(prods[["sku", "unit_cost"]], on="sku")
po["value"] = po["quantity"] * po["unit_cost"]
g = po.groupby("supplier_id").agg(open_pos=("po_id", "size"), open_value=("value", "sum"),
                                  delayed=("status", lambda s: int((s == "DELAYED").sum())))
skus = prods.groupby("supplier_id").size().rename("skus")
risk = d.assign(r=(d["stockout_day"] >= 0) & (d["stockout_day"] < 56)).groupby("supplier_id")["r"].sum().rename("skus_at_risk")
ex = sups.set_index("supplier_id")[["supplier_name", "country", "quoted_lead_time_days", "lead_time_p50", "lead_time_p90",
                                    "receipts"]].join([skus, g, risk]).fillna(0)
st.dataframe(ex.reset_index(), hide_index=True, width="stretch", column_config={
    "quoted_lead_time_days": st.column_config.NumberColumn("quoted LT (d)", format="%.0f"),
    "lead_time_p50": st.column_config.NumberColumn("LT P50 (d)", format="%.0f"),
    "lead_time_p90": st.column_config.NumberColumn("LT P90 (d)", format="%.0f"),
    "receipts": st.column_config.NumberColumn("receipts observed", format="%d"),
    "open_value": st.column_config.NumberColumn("open PO value", format="dollar"),
    "skus_at_risk": st.column_config.NumberColumn("SKUs at risk (8 wk)", format="%d"),
})
theme.note("Lead-time P50/P90: Kaplan–Meier estimate from received POs, with still-open POs counted as censored "
           "(they have taken at least their age), shrunk toward the quote while receipts are few.")

theme.section("Largest excess positions")
ex2 = d[np.isfinite(d["weeks_of_cover"]) & (d["weeks_of_cover"] > k["excess_weeks_of_cover"])].copy()
ex2["excess_value"] = (ex2["stock_now"] + ex2["on_order"] - k["excess_weeks_of_cover"] * ex2["weekly_demand"]) * ex2["unit_cost"]
ex2 = ex2.sort_values("excess_value", ascending=False).head(15)
st.dataframe(pd.DataFrame({"SKU": ex2["sku"], "Product": ex2["description"].str.title(), "Category": ex2["category"],
                           "Weeks of cover": ex2["weeks_of_cover"], "Weekly demand": ex2["weekly_demand"],
                           "Excess value": ex2["excess_value"], "Discontinued": ex2["discontinued"]}),
             hide_index=True, width="stretch", column_config={
                 "Weeks of cover": st.column_config.NumberColumn(format="%.0f"),
                 "Weekly demand": st.column_config.NumberColumn(format="%.1f"),
                 "Excess value": st.column_config.NumberColumn(format="dollar")})
