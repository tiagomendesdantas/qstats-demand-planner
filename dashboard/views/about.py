import json
from pathlib import Path

import pandas as pd
import streamlit as st
import theme

import data

cfg = data.cfg()
man = json.loads((Path(data.ROOT) / cfg["paths"]["processed_dir"] / "manifest.json").read_text())
c = man["cleaning"]

theme.title("About the demo data", "What is real, what is transformed, what is simulated, and what the planner never sees.")

st.markdown(f"""
**Historical demand patterns.** Derived from real transactions in the UCI Online Retail II dataset: a UK-based online
retailer of giftware with many wholesale customers, {man["original_date_range"][0][:10]} to {man["original_date_range"][1][:10]}.

**Supply-chain environment.** Simulated. The company, its two U.S. distribution centres, Amazon FBA, inventory
availability, suppliers, lead times, purchase orders, MOQ, case packs, costs, margins, promotions and container
constraints are all generated for this demo (fixed seed, `config/demo.yaml`). None of it describes the original retailer.

**Stockout benchmark.** Historical sales patterns are treated as a baseline demand path and intentionally censored through
simulated stockouts, so reconstruction methods can be evaluated against a known reference within the simulation. This
says nothing about what the original retailer lost to stockouts.

**Dates.** Every date is shifted forward by exactly {cfg["calendar"]["shift_weeks"]} weeks (weekdays and seasons preserved),
so the history reads as {man["shifted_date_range"][0]} to {man["shifted_date_range"][1]}.
""")

theme.section("Layers")
st.dataframe(
    pd.DataFrame(
        [
            ("Real (UCI)", "Which product sold, on which day, how many units, at what price, to which customer ID"),
            (
                "Transformed",
                "Cleaning (below); daily series inside each product's active life; date shift; customers assigned to "
                "a region and channel by a stable hash, so each location keeps real order lumpiness",
            ),
            (
                "Simulated",
                "Warehouses, Amazon FBA and its stock states, inventory, suppliers and lead times, purchase orders, MOQ, "
                "case packs, cube, USD costs and margins, containers, promotions and liquidations, gaps in the inventory feed",
            ),
            (
                "Hidden from the planner",
                "Baseline demand, lost sales, true lead-time parameters, promotion uplifts: used only to score",
            ),
            (
                "Estimated by the planner",
                "Demand on stockout days, forecasts and intervals, lead-time distributions, every recommendation",
            ),
        ],
        columns=["Layer", "Content"],
    ),
    hide_index=True,
    width="stretch",
)

theme.section("Cleaning, with counts")
st.dataframe(
    pd.DataFrame(
        [
            ("Rows in the file (two sheets)", man["adapter"]["rows_in_file"]),
            ("Overlap between the sheets (1–9 Dec 2010), removed", man["adapter"]["sheet_overlap_rows"]),
            ("Exact duplicate lines, removed", c["exact_duplicates_removed"]),
            ("Adjustments and bad-debt lines, removed", c["adjustment_rows_removed"]),
            ("Postage, fees, vouchers, test and other non-product codes, removed", c["non_merchandise_rows_removed"]),
            ("Lines with a zero or negative price, removed", c["non_positive_price_rows_removed"]),
            ("Returns kept as returns (never subtracted blindly)", c["return_rows"]),
            ("Sales cancelled by an exactly matching return (left out of demand)", c["cancelled_sale_rows"]),
            ("Wholesale lots left out of demand (>10× the SKU's usual large line and ≥ 1,000 units)", c["bulk_rows"]),
            ("Lines without a customer ID (kept)", c["missing_customer_rows"]),
        ],
        columns=["Rule", "Lines"],
    ),
    hide_index=True,
    width="stretch",
    column_config={"Lines": st.column_config.NumberColumn(format="localized")},
)
st.markdown(f"""
Units: {c["gross_units_sold"]:,.0f} gross sold; {c["cancelled_units"]:,.0f} in cancelled sales; {c["bulk_units"]:,.0f} in
wholesale lots; {c["return_units"]:,.0f} returned. Demand = fulfilled sales that were neither cancelled nor wholesale lots.
Returns stay available as their own field. Details: `docs/data_methodology.md`.
""")

theme.section("Source and licence")
st.markdown("""
Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The file is downloaded unchanged and verified
by SHA-256; this project cleaned, aggregated, re-dated and re-channelled it as described above.

A dated portfolio demo by QStats on public data. It is not client work and has not been used in production.
""")
