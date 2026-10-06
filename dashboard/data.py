"""Cached access to the planner database for the dashboard (the same Repository the API uses)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from qstats_planner.services.repository import Repository  # noqa: E402
from qstats_planner.utils.config import database_url, load_config  # noqa: E402


@st.cache_resource
def cfg() -> dict:
    return load_config()


@st.cache_resource
def repo() -> Repository:
    return Repository(database_url(cfg()))


@st.cache_data(ttl=600)
def table(name: str, *args) -> pd.DataFrame:
    return getattr(repo(), name)(*args)


@st.cache_data(ttl=600)
def kpis() -> dict:
    return repo().kpis()


@st.cache_data(ttl=600)
def eval_table(name: str) -> pd.DataFrame:
    return repo().eval_table(name)


@st.cache_data(ttl=600)
def eval_meta() -> dict:
    return repo().eval_meta()


def recommendations() -> pd.DataFrame:
    """Not cached: planner decisions change the status column."""
    return repo().recommendations()


@st.cache_resource(show_spinner="Loading today's state for scenarios…")
def scenario_service():
    from qstats_planner.services.scenario import ScenarioService

    return ScenarioService(cfg())


def plan_date() -> str:
    return pd.Timestamp(kpis()["plan_date"]).strftime("%a %d %b %Y")
