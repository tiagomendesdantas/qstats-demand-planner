import pandas as pd
import streamlit as st

import data
import theme

cfg = data.cfg()
theme.title("Scenario simulator", "Rerun this week's plan under different assumptions. The fitted forecasts are reused; "
            "lead-time demand, safety stock, order quantities and containers are recomputed (about a second).")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    sl = st.selectbox("Target service level", ["Product targets (A 97 / B 95 / C 90%)", "90%", "95%", "97%", "99%"])
    growth = st.slider("Demand growth", -20, 30, 0, step=5, format="%d%%")
with c2:
    ltm = st.select_slider("Lead time", options=[0.8, 1.0, 1.2, 1.5], value=1.0, format_func=lambda v: f"× {v:.1f}")
    var = st.select_slider("Supplier variability", options=[0.5, 1.0, 1.5, 2.0], value=1.0, format_func=lambda v: f"× {v:.1f} spread")
with c3:
    cap = st.number_input("Container capacity (m³)", 20.0, 80.0, 68.0, step=1.0)
    pol = st.radio("Safety-stock policy", ["Probabilistic (QStats)", "Days of cover (legacy rule)"], horizontal=True)
    days = st.slider("Days of cover", 7, 90, 30, disabled=pol.startswith("Prob"))

svc = None if sl.startswith("Product") else int(sl.rstrip("%")) / 100
svc_obj = data.scenario_service()


@st.cache_data(show_spinner="Re-planning…")
def run(svc, growth, ltm, var, cap, prob, days):
    r = svc_obj.run(svc, growth, ltm, var, cap, prob, days)
    return r.summary, r.by_sku


base, _ = run(None, 0, 1.0, 1.0, 68.0, True, 30)
scen, by = run(svc, growth, ltm, var, cap, pol.startswith("Prob"), days)

lines = [("Purchase lines", "purchase_lines", "int"), ("Purchase units", "purchase_units", "int"),
         ("Purchase value", "purchase_value", "money"), ("SKUs at risk, before orders", "skus_at_risk_before_orders", "int"),
         ("SKUs at risk, after orders", "skus_at_risk_after_orders", "int"),
         ("Projected service level", "projected_service_level", "pct"),
         ("Projected average inventory, 13 weeks", "projected_average_inventory_13w", "money"),
         ("Safety stock value", "safety_stock_value", "money"), ("Working capital committed", "working_capital_committed", "money"),
         ("Containers", "containers", "int"), ("Mean container utilisation", "mean_container_utilisation", "pct")]
fmt = {"int": lambda v: f"{v:,.0f}", "money": theme.money, "pct": lambda v: theme.pct(v, 1)}
theme.strip([
    ("Purchase value", theme.money(scen["purchase_value"]), f"base {theme.money(base['purchase_value'])}"),
    ("Projected service", theme.pct(scen["projected_service_level"], 1), f"base {theme.pct(base['projected_service_level'], 1)}"),
    ("SKUs at risk after orders", f"{scen['skus_at_risk_after_orders']}", f"base {base['skus_at_risk_after_orders']}"),
    ("Avg inventory, 13 wk", theme.money(scen["projected_average_inventory_13w"]), f"base {theme.money(base['projected_average_inventory_13w'])}"),
    ("Working capital", theme.money(scen["working_capital_committed"]), f"base {theme.money(base['working_capital_committed'])}"),
])
tbl = pd.DataFrame([{"Measure": a, "This week's plan": fmt[f](base[k]), "Scenario": fmt[f](scen[k]),
                     "Change": (f"{(scen[k] - base[k]) * 100:+.1f} pts" if f == "pct" else
                                theme.money(scen[k] - base[k]) if f == "money" else f"{scen[k] - base[k]:+,.0f}")}
                    for a, k, f in lines])
theme.section("Scenario vs this week's plan")
st.dataframe(tbl, hide_index=True, width="stretch")
theme.note("Projected service level: demand-weighted probability of covering demand over lead time + review after the "
           "recommended orders. SKUs at risk: projected to run out within 8 weeks. Working capital committed: stock on hand "
           "plus the purchases this plan would commit to, at cost.")

theme.section("Largest purchase changes")
bb = run(None, 0, 1.0, 1.0, 68.0, True, 30)[1]
ch = by.merge(bb, on="sku", suffixes=("", "_base"))
ch["change"] = ch["purchase_value"] - ch["purchase_value_base"]
ch = ch[ch["change"].abs() > 0].sort_values("change", key=abs, ascending=False).head(15)
st.dataframe(ch[["sku", "order_qty_base", "order_qty", "purchase_value_base", "purchase_value", "change"]], hide_index=True, width="stretch",
             column_config={"order_qty_base": st.column_config.NumberColumn("units, plan", format="localized"),
                            "order_qty": st.column_config.NumberColumn("units, scenario", format="localized"),
                            "purchase_value_base": st.column_config.NumberColumn("value, plan", format="dollar"),
                            "purchase_value": st.column_config.NumberColumn("value, scenario", format="dollar"),
                            "change": st.column_config.NumberColumn(format="dollar")})
