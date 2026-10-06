import numpy as np
import pandas as pd
import streamlit as st
import theme

import data

pal = theme.palette()
k = data.kpis()
cfg = data.cfg()
f = data.table("fba")
f["fba_enabled"] = f["fba_enabled"].astype(bool)
prods = data.table("products")
f = f.merge(prods[["sku_idx", "description", "case_pack", "preferred_source_dc"]], on="sku_idx")

theme.title(
    "Amazon FBA",
    "Amazon stock is planned on its own: forecast Amazon demand and its uncertainty over the "
    "replenishment window (pick + transit + review), counted by inventory state, sent from the DC that can spare it.",
)
below = f[f["fba_position"] < f["fba_target"]]
theme.strip(
    [
        ("FBA SKUs", f"{len(f)}", "enabled for Amazon"),
        ("Below target", f"{len(below)}", "position under the service-level quantity"),
        ("Units to send", theme.units(f["send_qty"].sum()), f"{int((f['send_qty'] > 0).sum())} shipments"),
        (
            "Pick + transit (P90)",
            f"{k['transit_p90']:.0f} d",
            f"past transfers; the window adds the {cfg['simulation']['review_period_days']}-day review",
        ),
        (
            "Units at Amazon",
            theme.units(f["fba_available"].sum() + f["fba_transfer"].sum() + f["fba_reserved"].sum()),
            f"{theme.units(f['fba_inbound'].sum())} inbound",
        ),
    ]
)
theme.callout(
    "Planning value by state: AVAILABLE and FC TRANSFER count in full; INBOUND counts in full on its way; "
    "RESERVED counts at 0.5, because part of it never returns to sellable. The legacy rule counts every unit "
    "at Amazon as available and sends a fixed number of days of cover."
)

tbl = pd.DataFrame(
    {
        "SKU": f["sku"],
        "Product": f["description"].str.title(),
        "FBA available": f["fba_available"],
        "Inbound": f["fba_inbound"],
        "Reserved": f["fba_reserved"],
        "FC transfer": f["fba_transfer"],
        "Days of supply": f["fba_days_of_supply"].replace([np.inf, -np.inf], np.nan),
        "Forecast / day": f["fba_daily_demand"],
        "Target": f["fba_target"],
        "Position": f["fba_position"],
        "Recommended transfer": f["send_qty"],
        "Source DC": f["source_dc"].fillna("—"),
        "Remaining at source": np.where(f["send_qty"] > 0, f["remaining_source"], np.nan),
    }
).sort_values(["Recommended transfer", "Days of supply"], ascending=[False, True])
st.dataframe(
    tbl,
    hide_index=True,
    width="stretch",
    height=520,
    column_config={
        c: st.column_config.NumberColumn(format="localized")
        for c in [
            "FBA available",
            "Inbound",
            "Reserved",
            "FC transfer",
            "Target",
            "Position",
            "Recommended transfer",
            "Remaining at source",
        ]
    }
    | {
        "Days of supply": st.column_config.NumberColumn(format="%.0f", help="Blank: no Amazon demand forecast"),
        "Target": st.column_config.NumberColumn(format="%.0f"),
        "Position": st.column_config.NumberColumn(format="%.0f"),
        "Forecast / day": st.column_config.NumberColumn(format="%.2f"),
        "Product": st.column_config.TextColumn(width="large"),
    },
)
theme.note(
    "Target: the service-level quantile of Amazon demand over the replenishment window (pick + transit + review). "
    "Position: available + FC transfer + inbound + units being picked at the DC + 0.5 × reserved. The source DC "
    f"keeps {cfg['simulation']['dc_protection_days_for_fba']} days of its own expected demand before it sends. "
    "When no Amazon demand is forecast, the target is the forecast-error allowance alone: errors are scaled by the "
    "SKU's longer-run demand, so a SKU that sold on Amazon before keeps a small target. Sends that avoid less than "
    "one unit of expected shortfall, or whose Amazon sales lose money, are marked LOW in the Action center."
)
