"""QStats Demand & Inventory Planner: dashboard entry point.

    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="QStats Demand & Inventory Planner", page_icon="▦", layout="wide")

import data  # noqa: E402
import theme  # noqa: E402

pages = {
    "Plan": [
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/action_center.py", title="Action center"),
        st.Page("views/sku_detail.py", title="SKU detail"),
        st.Page("views/fba_planner.py", title="Amazon FBA"),
        st.Page("views/container_planner.py", title="Containers"),
        st.Page("views/scenarios.py", title="Scenarios"),
    ],
    "Analyze": [
        st.Page("views/inventory_health.py", title="Inventory health"),
        st.Page("views/forecast_performance.py", title="Forecast performance"),
    ],
    "Evidence": [
        st.Page("views/constrained_demand.py", title="Constrained demand"),
        st.Page("views/legacy_vs_qstats.py", title="Legacy vs QStats"),
    ],
    "About": [st.Page("views/about.py", title="About the data")],
}
nav = st.navigation(pages, position="top")
theme.setup_page()
theme.masthead(data.plan_date())
nav.run()
