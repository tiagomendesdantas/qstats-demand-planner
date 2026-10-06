import pandas as pd
import streamlit as st

import data
import theme

pal = theme.palette()
lines, summary = data.table("containers")
prods = data.table("products").set_index("sku")

theme.title("Container planner", "This week's purchase lines grouped by supplier into 40ft high-cube containers (68 m³). "
            "An under-filled container is topped up with whole cases of that supplier's other SKUs, lowest cover first, "
            "never past 26 weeks of cover and never for a low-margin or discontinued SKU.")
if summary.empty:
    theme.note("No purchase lines this week.")
    st.stop()

theme.strip([
    ("Suppliers ordering", f"{len(summary)}", ""),
    ("Containers", f"{int(summary['containers'].sum())}", "40ft high cube"),
    ("Purchase value", theme.money(summary["purchase_value"].sum()), f"{theme.money(summary['top_up_value'].sum())} of it top-up"),
    ("Mean utilisation", theme.pct(summary["utilisation"].mean(), 0), "cube used / capacity"),
    ("Below minimum fill", f"{int(summary['below_minimum_fill'].sum())}", "consolidate or wait"),
])

sid = st.selectbox("Supplier", summary["supplier_id"],
                   format_func=lambda s: f"{s} · {summary.set_index('supplier_id').at[s, 'supplier_name']} "
                                         f"({summary.set_index('supplier_id').at[s, 'country']})")
s = summary.set_index("supplier_id").loc[sid]
ln = lines[lines["supplier_id"] == sid].copy()

a, b = st.columns([2, 3], gap="large")
with a:
    theme.section(f"{s['supplier_name']} · {int(s['containers'])} container{'s' if s['containers'] > 1 else ''}")
    fig = theme.figure(130)
    used = s["utilisation"]
    fig.add_bar(y=["fill"], x=[used], orientation="h", marker_color=pal["s1"], name="Used", showlegend=False,
                hovertemplate=f"used {used:.0%}<extra></extra>")
    fig.add_bar(y=["fill"], x=[1 - used], orientation="h", marker_color=pal["grid"], showlegend=False,
                hovertemplate=f"unused {1 - used:.0%}<extra></extra>")
    fig.update_layout(barmode="stack", hovermode="closest", margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(showticklabels=False)
    theme.show(fig)
    st.html(theme.kv([
        ("Container utilisation", theme.pct(s["utilisation"], 0)),
        ("Unused capacity", theme.pct(s["unused_capacity_pct"], 0)),
        ("Cube", f"{s['cube_m3']:.1f} m³"),
        ("Purchase value", theme.money(s["purchase_value"])),
        ("Recommended / top-up", f"{theme.money(s['recommended_value'])} / {theme.money(s['top_up_value'])}"),
        ("Minimum order value", f"{theme.money(s['minimum_order_value'])} · {'met' if s['meets_minimum_order'] else 'not met'}"),
    ]))
with b:
    theme.section("Container mix")
    tbl = pd.DataFrame({
        "SKU": ln["sku"], "Product": ln["sku"].map(prods["description"]).str.title(),
        "Recommended": ln["recommended_units"], "Top-up": ln["top_up_units"], "Units": ln["units"], "Cases": ln["cases"],
        "Cube m³": ln["cube_m3"], "Value": ln["value"], "Cover after (wk)": ln["weeks_of_cover_after"],
    }).sort_values("Units", ascending=False)
    tbl = tbl[tbl["Units"] > 0]
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "Value": st.column_config.NumberColumn(format="dollar"), "Cube m³": st.column_config.NumberColumn(format="%.2f"),
        "Cover after (wk)": st.column_config.NumberColumn(format="%.1f"),
        **{c: st.column_config.NumberColumn(format="localized") for c in ["Recommended", "Top-up", "Units", "Cases"]}})

theme.section("All suppliers")
st.dataframe(summary[["supplier_id", "supplier_name", "country", "containers", "utilisation", "unused_capacity_pct", "skus",
                      "purchase_value", "top_up_value", "meets_minimum_order", "below_minimum_fill"]],
             hide_index=True, width="stretch", column_config={
                 "utilisation": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                 "unused_capacity_pct": st.column_config.NumberColumn("unused", format="percent"),
                 "purchase_value": st.column_config.NumberColumn(format="dollar"),
                 "top_up_value": st.column_config.NumberColumn(format="dollar")})
theme.note("The greedy rule is deliberately transparent. The solver sits behind an interface "
           "(optimization/containers.py), so an OR-Tools or PuLP model can replace it without touching the rest of the plan.")
