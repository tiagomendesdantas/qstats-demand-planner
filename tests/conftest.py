"""Shared fixtures. Most tests run on a small synthetic environment built in memory, so the suite
does not need the UCI file; tests that need the real artefacts skip when they are absent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qstats_planner.simulation.environment import Environment
from qstats_planner.utils.calendar import TradingCalendar
from qstats_planner.utils.config import ROOT, load_config


@pytest.fixture
def cfg():
    return load_config(**{"simulation.warmup_weeks": 4})


def make_env(cfg, n_sku: int = 4, weeks: int = 30, rate: float = 6.0, seed: int = 1, fba: bool = True) -> Environment:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=weeks * 7, freq="D")  # a Monday
    trading = days.dayofweek != 5
    cal = TradingCalendar(days[trading], days[-1], cfg)
    baseline = np.zeros((len(days), n_sku, 3))
    rates = rate * (1 + np.arange(n_sku) / 2)
    for i in range(n_sku):
        baseline[:, i, 0] = rng.poisson(rates[i] * 0.6, len(days))
        baseline[:, i, 1] = rng.poisson(rates[i] * 0.4, len(days))
        baseline[:, i, 2] = rng.poisson(rates[i] * 0.3, len(days)) if (fba and i % 2 == 0) else 0
    baseline[~trading] = 0
    products = pd.DataFrame(
        {
            "sku": [f"S{i:03d}" for i in range(n_sku)],
            "description": [f"ITEM {i}" for i in range(n_sku)],
            "category": "Gifts & Accessories",
            "supplier_idx": [i % 2 for i in range(n_sku)],
            "supplier_id": [f"SUP-0{i % 2 + 1}" for i in range(n_sku)],
            "selling_price": 10.0,
            "unit_cost": 4.0,
            "case_pack": 12,
            "moq": 48,
            "cube_per_case": 0.05,
            "abc_class": "B",
            "target_service_level": 0.95,
            "fba_enabled": [fba and i % 2 == 0 for i in range(n_sku)],
            "preferred_source_dc": "EAST_DC",
            "fulfillment_cost_dc": 1.45,
            "fulfillment_cost_fba": 2.65,
            "advertising_cost": 0.8,
            "contribution_dc": 3.75,
            "contribution_fba": 2.55,
            "contribution_margin": 3.5,
            "contribution_margin_pct": 0.35,
        }
    )
    suppliers = pd.DataFrame(
        {
            "supplier_id": ["SUP-01", "SUP-02"],
            "supplier_name": ["Supplier 01", "Supplier 02"],
            "country": ["China", "Vietnam"],
            "base_median_days": [21.0, 28.0],
            "idio_sd": [0.1, 0.1],
            "shock_sd": [0.05, 0.05],
            "delay_p": [0.05, 0.05],
            "delay_mean_days": [7.0, 7.0],
            "minimum_order_value": [1000.0, 1000.0],
            "minimum_container_fill": [0.55, 0.55],
            "container_capacity_m3": [68.0, 68.0],
            "quoted_lead_time_days": [21.0, 28.0],
        }
    )
    nw = weeks + 30
    lt = {
        "shock": rng.normal(size=(nw, 2)),
        "z_idio": rng.normal(size=(nw, n_sku)),
        "delay_u": rng.uniform(size=(nw, n_sku)),
        "delay_days": rng.exponential(size=(nw, n_sku)),
        "cancel_u": np.ones((nw, n_sku)),
        "cancel_p": 0.0,
        "disruption": np.zeros((nw, 2)),
    }
    hist = pd.DataFrame(
        {
            "po_id": [f"H{k}" for k in range(8)],
            "supplier_id": ["SUP-01", "SUP-02"] * 4,
            "order_date": days[0] - pd.Timedelta(days=60),
            "receipt_date": days[0] - pd.Timedelta(days=30),
            "lead_time_days": [20.0, 27.0, 22.0, 29.0, 21.0, 28.0, 23.0, 30.0],
        }
    )
    events = pd.DataFrame(columns=["sku_idx", "kind", "start_day", "end_day", "announce_day", "uplift", "price_change_pct"])
    return Environment(
        seed=seed,
        days=days,
        trading=trading,
        calendar=cal,
        products=products,
        suppliers=suppliers,
        baseline=baseline,
        pre_uplift=baseline.copy(),
        launch_day=np.zeros(n_sku, int),
        end_day=np.full(n_sku, len(days) - 1),
        launch_stock=np.zeros(n_sku),
        events=events,
        lead_time_history=hist,
        unknown_mask=np.zeros_like(baseline, bool),
        fba_unavailable_share=np.full((len(days), n_sku), 0.05),
        fba_transit=np.full((nw, n_sku), 7),
        _lt=lt,
    )


@pytest.fixture
def env(cfg):
    return make_env(cfg)


@pytest.fixture
def neutral_prior():
    from qstats_planner.forecasting.seasonal_prior import SeasonalPrior

    return SeasonalPrior({0: np.ones(12), 1: np.ones(12)}, np.zeros(4, int))


def processed_available() -> bool:
    return (ROOT / "data" / "processed" / "daily_demand.parquet").exists()


def database_available() -> bool:
    return (ROOT / "data" / "planner.sqlite").exists()
