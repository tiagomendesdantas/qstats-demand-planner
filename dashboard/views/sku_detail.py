import json

import numpy as np
import pandas as pd
import streamlit as st
import theme

import data

pal = theme.palette()
prods = data.table("products").set_index("sku")
sp = data.table("sku_plan").set_index("sku")
recs = data.recommendations()
sups = data.table("suppliers").set_index("supplier_id")
k = data.kpis()

order = recs[recs["action"] != "NO_ACTION"].drop_duplicates("sku")["sku"].tolist()
order += [s for s in prods.index if s not in order]
default = st.query_params.get("sku", order[0])
sku = st.selectbox(
    "SKU",
    order,
    index=order.index(default) if default in order else 0,
    format_func=lambda s: f"{s} · {prods.at[s, 'description'].title()}",
)
st.query_params["sku"] = sku
p, s = prods.loc[sku], sp.loc[sku]
sup = sups.loc[p["supplier_id"]]

theme.title(
    p["description"].title(),
    f"{sku} · {p['category']} · {sup['supplier_name']} ({sup['country']}) · "
    f"segment {s['segment'].replace('_', ' ').lower()} · forecast by {s['champion_model']} · "
    f"confidence {s['confidence'].lower()}",
)
locs = "EAST_DC, WEST_DC" + (", AMAZON_FBA" if p["fba_enabled"] else "")
theme.strip(
    [
        ("Locations", "3" if p["fba_enabled"] else "2", locs),
        ("Lead time", f"{s['lt_p50']:.0f} / {s['lt_p90']:.0f} d", f"P50 / P90 · quoted {sup['quoted_lead_time_days']:.0f} d"),
        ("MOQ · case pack", f"{p['moq']:,} · {p['case_pack']}", "units"),
        ("Margin", theme.pct(p["contribution_margin_pct"], 0), f"{theme.money(p['contribution_margin'], 2)} per unit"),
        ("Service target", theme.pct(p["target_service_level"], 0), f"class {p['abc_class']}"),
        ("Inventory position", theme.units(s["inventory_position"]), f"{theme.units(s['on_order'])} on order"),
        (
            "Weeks of cover",
            f"{s['weeks_of_cover']:.1f}" if np.isfinite(s["weeks_of_cover"]) else "—",
            f"{theme.units(s['weekly_demand'])} units / week",
        ),
    ]
)

r = recs[(recs["sku"] == sku) & (recs["action"] != "NO_ACTION")]
theme.section("Recommendation")
if r.empty:
    st.html(
        "<div class='qs-explain'><div class='what'>No action</div><div class='why'>The inventory position covers "
        "demand over lead time + review at the service target.</div></div>"
    )
else:
    top = r.iloc[0]
    ev = json.loads(top["evidence"])
    a, b = st.columns([3, 2], gap="large")
    with a:
        qty = f" {int(top['recommended_quantity']):,} units" if top["recommended_quantity"] else ""
        st.html(
            f"<div class='qs-explain'><div class='what'>{top['action'].replace('_', ' ')}{qty}</div>"
            f"<div class='why'>{top['reason']}</div></div>"
        )
        rows = [
            ("Severity", theme.chip(top["severity"])),
            ("Expected effect", top["expected_effect"]),
            (
                {
                    "CRITICAL_STOCKOUT": "Expected loss before relief",
                    "EXCESS": "Yearly carrying cost",
                    "TRANSFER": "Shipping cost avoided",
                }.get(top["action"], "Contribution protected"),
                theme.money(abs(top["economic_impact"])),
            ),
            ("Confidence", f"{top['confidence']} ({top['confidence_score']:.2f})"),
        ]
        if pd.notna(top["stockout_date"]):
            rows.insert(1, ("Projected stockout", pd.Timestamp(top["stockout_date"]).strftime("%b %d")))
        st.html(theme.kv(rows))
        if len(r) > 1:
            theme.note(
                "Also: "
                + " · ".join(f"{x.action.replace('_', ' ').lower()} ({x.severity.lower()})" for x in r.iloc[1:].itertuples())
            )
    with b:
        keys = [
            "Inventory position",
            "P50 demand over lead time + review",
            "P90 demand over lead time + review",
            "Safety stock",
            "Order-up-to level",
            "Lead time P50 (days)",
            "Lead time P90 (days)",
            "MOQ",
            "Case pack",
            "Raw requirement",
            "Recommended (rounded)",
            "To EAST_DC",
            "To WEST_DC",
            "Expected arrival",
            "Purchase value",
        ]
        st.html(
            "<div class='qs-lab'>Evidence</div>"
            + theme.kv([(kk, f"{ev[kk]:,.0f}" if isinstance(ev[kk], (int, float)) else str(ev[kk])) for kk in keys if kk in ev])
        )

# ---------------------------------------------------------------- demand
theme.section("Demand: sales, reconstructed demand and forecast")
h = data.table("history", sku)
fc = data.table("forecast", sku)
h["week"] = pd.to_datetime(h["week"])
fc["week"] = pd.to_datetime(fc["week"])
h = h[h["week"] >= h["week"].max() - pd.Timedelta(weeks=78)]
fig = theme.figure(340, y_title="units / week")
theme.shade_runs(fig, h["week"], h["stockout_days"] > 0, pal["shade"], "stockout")
theme.shade_runs(fig, h["week"], h["event_week"].astype(bool), pal["event"], "promotion / liquidation (simulated)")
fig.add_scatter(x=fc["week"], y=fc["p90"], line=dict(width=0), hoverinfo="skip", showlegend=False)
fig.add_scatter(
    x=fc["week"],
    y=fc["p10"],
    fill="tonexty",
    fillcolor=pal["band"],
    line=dict(width=0),
    name="80% interval (P10–P90)",
    hoverinfo="skip",
)
fig.add_scatter(
    x=fc["week"],
    y=fc["p95"],
    name="P95",
    mode="lines",
    line=dict(color=pal["s1"], width=1, dash="dot"),
    hovertemplate="%{y:,.0f}",
)
fig.add_scatter(
    x=h["week"],
    y=h["observed"],
    name="Observed sales",
    line=dict(color=pal["s2"], width=2),
    mode="lines",
    hovertemplate="%{y:,.0f}",
)
fig.add_scatter(
    x=h["week"],
    y=h["reconstructed"],
    name="Reconstructed demand",
    mode="lines",
    line=dict(color=pal["s1"], width=2, dash="dot"),
    hovertemplate="%{y:,.0f}",
)
fig.add_scatter(
    x=fc["week"],
    y=fc["expected"],
    name="Forecast",
    line=dict(color=pal["s1"], width=2),
    mode="lines",
    customdata=fc[["p10", "p90"]],
    hovertemplate="%{y:,.0f} (P10 %{customdata[0]:,.0f}, P90 %{customdata[1]:,.0f})",
)
fig.add_vline(x=pd.Timestamp(k["plan_date"]), line=dict(color=pal["rule"], width=1))
theme.show(fig)
theme.note(
    "Reconstructed demand differs from sales only in weeks with stockouts (shaded): there the planner estimates what "
    "would have sold. Weekly intervals come from backtest errors of a single week at that distance ahead, pooled "
    "over the SKU's segment. Week-to-week noise dominates this demand, so the band widens only a little with the "
    "horizon (and narrows where the forecast falls toward zero). Its coverage is not measured separately; the "
    "lead-time-demand quantiles that set orders were measured to run narrow (see Forecast performance)."
)

# ---------------------------------------------------------------- inventory
theme.section("Projected inventory position, next 180 days")
pr = data.table("projection", sku)
pr["date"] = pd.to_datetime(pr["date"])
fig = theme.figure(320, y_title="units in the network")
fig.add_scatter(
    x=pr["date"],
    y=pr["projected"].clip(lower=0),
    name="Without a new order",
    line=dict(color=pal["s2"], width=2),
    hovertemplate="%{y:,.0f}",
)
if s["recommended_quantity"] > 0:
    fig.add_scatter(
        x=pr["date"],
        y=pr["projected_with_order"].clip(lower=0),
        name="With the recommended order",
        line=dict(color=pal["s1"], width=2),
        hovertemplate="%{y:,.0f}",
    )
fig.add_scatter(
    x=pr["date"],
    y=pr["safety_stock"],
    name="Safety stock",
    line=dict(color=pal["ink"], width=1, dash="dash"),
    hovertemplate="%{y:,.0f}",
)
rec_days = pr[pr["receipts_with_order"] > 0]
if len(rec_days):
    yv = np.where(rec_days["receipts"] > 0, rec_days["projected"], rec_days["projected_with_order"]).clip(min=0)
    fig.add_scatter(
        x=rec_days["date"],
        y=yv,
        mode="markers",
        name="Inbound receipts",
        marker=dict(symbol="triangle-up", size=11, color=pal["ink"], line=dict(width=2, color=pal["surface"])),
        customdata=rec_days["receipts_with_order"],
        hovertemplate="receipt %{customdata:,.0f} units",
    )
if s["stockout_day"] >= 0:
    d0 = pd.Timestamp(k["plan_date"]) + pd.Timedelta(days=int(s["stockout_day"]))
    fig.add_vline(x=d0, line=dict(color=theme.STATUS["CRITICAL"], width=1.5))
    fig.add_annotation(
        x=d0,
        y=1,
        yref="paper",
        text=f"● projected stockout {d0:%b %d}",
        showarrow=False,
        xanchor="left",
        font=dict(size=12, color=theme.STATUS["CRITICAL"]),
        yanchor="bottom",
    )
fig.update_yaxes(rangemode="tozero")
theme.show(fig)
theme.note(
    "Expected network stock: today's stock plus expected receipts minus expected demand. Open POs land on their "
    "expected date; this week's recommended order lands after the supplier's median lead time. Orders from later weekly "
    "reviews are not drawn, so a line that reaches zero months out is expected: next week's plan orders again."
)

a, b = st.columns(2, gap="large")
with a:
    theme.section("Purchase orders")
    po = data.table("purchase_orders", sku)
    if len(po):
        po = po.sort_values("order_date", ascending=False).head(12)
        st.dataframe(
            po[["po_id", "status", "quantity", "order_date", "expected_arrival", "actual_arrival"]],
            hide_index=True,
            width="stretch",
            column_config={"quantity": st.column_config.NumberColumn(format="localized")},
        )
    else:
        theme.note("No purchase orders.")
with b:
    theme.section("Forecast candidates (rolling-origin backtest)")
    perf = data.table("forecast_performance", sku).sort_values("cum_scaled_error")
    perf["champion"] = perf["champion"].astype(bool)
    perf = pd.concat([perf[perf["champion"]], perf[~perf["champion"]].head(7)])  # the champion always shows
    st.dataframe(
        perf[["model", "champion", "cum_scaled_error", "wape", "bias", "mae"]],
        hide_index=True,
        width="stretch",
        column_config={
            "cum_scaled_error": st.column_config.NumberColumn("scaled error (lead time)", format="%.2f"),
            "wape": st.column_config.NumberColumn("WAPE 1-wk", format="%.2f"),
            "bias": st.column_config.NumberColumn(format="%.2f"),
            "mae": st.column_config.NumberColumn("MAE", format="%.1f"),
        },
    )
    theme.note(
        f"How this SKU's model was chosen: {s['selection_reason']}. Each segment gets a champion; a SKU keeps its "
        "own pick only when it beats the segment's on enough of its own non-overlapping windows. The error shown "
        "averages every scored window, overlapping ones included, so its ranking can differ from the choice."
    )
