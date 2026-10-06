import json

import pandas as pd
import streamlit as st

import data
import theme

theme.title("Action center", "Exceptions first: each line says what to do, why, what it is worth and how sure the plan is. "
            "Accept, override or reject; every decision is stored with the system's original quantity.")

recs = data.recommendations()
prods = data.table("products")
recs = recs.merge(prods[["sku", "supplier_id"]].rename(columns={"supplier_id": "_s"}), on="sku", how="left")

f1, f2, f3, f4, f5, f6 = st.columns(6)
sev = f1.multiselect("Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
act = f2.multiselect("Action", sorted(recs["action"].unique()), placeholder="All actions")
sup = f3.multiselect("Supplier", sorted(recs["supplier_id"].unique()), placeholder="All suppliers")
loc = f4.multiselect("Location", sorted(recs["location"].unique()), placeholder="All locations")
cat = f5.multiselect("Category", sorted(recs["category"].unique()), placeholder="All categories")
seg = f6.multiselect("Segment", sorted(recs["segment"].unique()), placeholder="All segments")

view = recs[recs["severity"].isin(sev)] if sev else recs
for col, sel in (("action", act), ("supplier_id", sup), ("location", loc), ("category", cat), ("segment", seg)):
    if sel:
        view = view[view[col].isin(sel)]

counts = view["severity"].value_counts()
st.html(" &nbsp;&nbsp; ".join(f"{theme.chip(s)} <span class='qs-chip'>{int(counts.get(s, 0))}</span>"
                              for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]) +
        f" &nbsp;&nbsp; <span class='qs-note'>{len(view)} of {len(recs)} recommendations · "
        f"impact {theme.money(view['economic_impact'].clip(lower=0).sum())} protected</span>")

tbl = pd.DataFrame({
    "Priority": view["priority"],
    "Severity": view["severity"].map(theme.severity_label),
    "SKU": view["sku"],
    "Product": view["description"].str.title(),
    "Location": view["location"],
    "Action": view["action"],
    "Quantity": view["recommended_quantity"],
    "Projected stockout": pd.to_datetime(view["stockout_date"]),
    "Inventory position": view["inventory_position"],
    "Confidence": view["confidence"],
    "Economic impact": view["economic_impact"],
    "Status": view["status"],
})


def _color(v: str) -> str:
    for s, c in theme.STATUS.items():
        if isinstance(v, str) and v.endswith(s):
            return f"color: {c}; font-weight: 500"
    return ""


styled = tbl.style.map(_color, subset=["Severity"])
event = st.dataframe(
    styled, hide_index=True, width="stretch", height=430, on_select="rerun", selection_mode="single-row",
    key="rec_table",
    column_config={
        "Priority": st.column_config.NumberColumn(width="small"),
        "Quantity": st.column_config.NumberColumn(format="localized"),
        "Projected stockout": st.column_config.DateColumn(format="MMM D"),
        "Inventory position": st.column_config.NumberColumn(format="localized"),
        "Economic impact": st.column_config.NumberColumn(format="dollar", help="Expected contribution protected (or cost avoided); negative = carrying cost of excess"),
        "Product": st.column_config.TextColumn(width="large"),
    },
)
theme.note("Select a row to see the reasoning and record a decision. Economic impact: contribution expected to be protected "
           "by the action (units short avoided × contribution per unit), shipping cost avoided for transfers, or the yearly "
           "carrying cost of excess stock (negative).")

rows = event.selection.rows if event and event.selection else []
if rows:
    r = view.iloc[rows[0]]
    ev = json.loads(r["evidence"])
    theme.section(f"{r['recommendation_id']} · {r['sku']} · {r['description'].title()}")
    a, b = st.columns([3, 2], gap="large")
    with a:
        qty = f" {int(r['recommended_quantity']):,} units" if r["recommended_quantity"] else ""
        st.html(f"<div class='qs-explain'><div class='what'>{r['action'].replace('_', ' ')}{qty}</div>"
                f"<div class='why'>{r['reason']}</div></div>")
        st.html(theme.kv([
            ("Severity", theme.chip(r["severity"])),
            ("Expected effect", r["expected_effect"]),
            ("Economic impact", theme.money(r["economic_impact"])),
            ("Confidence", f"{r['confidence']} ({r['confidence_score']:.2f})"),
            ("Location", r["location"]),
            ("Status", r["status"]),
        ]))
        if st.button("Open SKU detail →", key="open_sku"):
            st.switch_page("views/sku_detail.py", query_params={"sku": r["sku"]})
    with b:
        st.html("<div class='qs-lab'>Evidence</div>" + theme.kv([
            (k, (f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 10 else
                 f"{v:.2f}" if isinstance(v, float) else str(v))) for k, v in ev.items()]))

    theme.section("Planner decision")
    c1, c2, c3 = st.columns([1, 1, 3])
    choice = c1.radio("Decision", ["ACCEPT", "OVERRIDE", "REJECT"], horizontal=False, key="dec")
    q = c2.number_input("Override quantity", min_value=0, step=int(prods.set_index("sku").at[r["sku"], "case_pack"]),
                        value=int(r["recommended_quantity"]), disabled=choice != "OVERRIDE")
    comment = c3.text_area("Comment", placeholder="Why (kept with the decision for the audit trail)", height=96)
    if st.button("Record decision", type="primary"):
        row = data.repo().record_decision(r["recommendation_id"], choice, int(q) if choice == "OVERRIDE" else None, comment)
        st.success(f"Recorded: {row['planner_action']} {r['recommendation_id']} "
                   f"(system {row['system_quantity']:,} → {row['override_quantity']:,}).")

dec = data.repo().decisions()
if len(dec):
    theme.section("Decision log")
    st.dataframe(dec.drop(columns=["id"]), hide_index=True, width="stretch",
                 column_config={"timestamp": st.column_config.DatetimeColumn(format="MMM D, HH:mm")})
