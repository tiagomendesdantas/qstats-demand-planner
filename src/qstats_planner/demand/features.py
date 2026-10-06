"""Weekly series and interpretable SKU metrics.

All metrics take an explicit window, so the same code serves the population selection (first 52
weeks only), the as-of segmentation inside the planner (history up to the plan date) and the
full-period description shown in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.utils.calendar import week_start
from qstats_planner.utils.text import has_keyword


def week_index(dates: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Week number, 1 for the week that contains `origin`."""
    ws = week_start(pd.Series(pd.DatetimeIndex(dates)))
    return ((ws - week_start(pd.Series([pd.Timestamp(origin)]))[0]).dt.days // 7 + 1).to_numpy()


def weekly_units(daily: pd.DataFrame, value: str = "units") -> pd.DataFrame:
    """Sum a daily column to Monday-based weeks; keeps the trading days counted per week."""
    g = daily.assign(week=week_start(daily["date"]))
    out = g.groupby(["sku", "week"]).agg(units=(value, "sum"), trading_days=("date", "size"))
    return out.reset_index()


def syntetos_boylan(units: np.ndarray, adi_cut: float, cv2_cut: float) -> tuple[float, float, str]:
    """ADI and CV^2 of non-zero demand, and the Syntetos-Boylan class."""
    nz = units[units > 0]
    if len(nz) == 0:
        return np.inf, np.nan, "LUMPY"
    adi = len(units) / len(nz)
    cv2 = float(np.var(nz) / np.mean(nz) ** 2) if len(nz) > 1 else 0.0
    if adi < adi_cut:
        return adi, cv2, "SMOOTH" if cv2 < cv2_cut else "ERRATIC"
    return adi, cv2, "INTERMITTENT" if cv2 < cv2_cut else "LUMPY"


def trend_strength(rate: np.ndarray) -> float:
    """Relative change across the window implied by a least-squares line (slope x n / mean)."""
    n = len(rate)
    if n < 8 or rate.mean() <= 0:
        return 0.0
    x = np.arange(n) - (n - 1) / 2
    slope = float((x * (rate - rate.mean())).sum() / (x**2).sum())
    return float(np.clip(slope * n / rate.mean(), -3, 3))


def peak_ratio(weeks: pd.Series, rate: np.ndarray) -> float:
    """Demand rate in September-November over the rate in the other months (1 = flat)."""
    months = pd.DatetimeIndex(weeks).month
    peak = np.isin(months, [9, 10, 11])
    if peak.sum() < 4 or (~peak).sum() < 4 or rate[~peak].mean() <= 0:
        return np.nan
    return float(rate[peak].mean() / rate[~peak].mean())


def sku_metrics(daily: pd.DataFrame, cfg: dict, start=None, end=None) -> pd.DataFrame:
    """The interpretable metrics the population and the segmentation are built from."""
    seg = cfg["segmentation"]
    d = daily
    if start is not None:
        d = d[d["date"] >= pd.Timestamp(start)]
    if end is not None:
        d = d[d["date"] <= pd.Timestamp(end)]
    weekly = weekly_units(d)
    keywords = seg["seasonal_keywords"]
    desc = d.drop_duplicates("sku").set_index("sku")["description"]
    rows = []
    for sku, w in weekly.groupby("sku", sort=False):
        units = w["units"].to_numpy(float)
        rate = units / np.maximum(w["trading_days"].to_numpy(float), 1)
        # Trend is measured outside the September-December season, so a seasonal ramp at either
        # end of a window is not read as growth or decline.
        off_peak = ~np.isin(pd.DatetimeIndex(w["week"]).month, [9, 10, 11, 12])
        adi, cv2, sb = syntetos_boylan(units, seg["adi_threshold"], seg["cv2_threshold"])
        rows.append(
            {
                "sku": sku,
                "total_units": units.sum(),
                "history_weeks": len(units),
                "mean_weekly_units": units.mean(),
                "zero_week_ratio": float((units == 0).mean()),
                "adi": adi,
                "cv2": cv2,
                "sb_class": sb,
                "coefficient_of_variation": float(units.std() / units.mean()) if units.mean() > 0 else np.nan,
                "trend_strength": trend_strength(rate[off_peak]),
                "peak_ratio": peak_ratio(w["week"], rate),
                "seasonal_keyword": has_keyword(desc.get(sku, ""), keywords),
            }
        )
    out = pd.DataFrame(rows)
    daily_stats = d.groupby("sku").agg(
        sales_days=("units", lambda s: int((s > 0).sum())),
        trading_days=("units", "size"),
        mean_daily_demand=("units", "mean"),
    )
    daily_stats["zero_demand_ratio"] = 1 - daily_stats["sales_days"] / daily_stats["trading_days"]
    out = out.merge(daily_stats.reset_index(), on="sku", how="left")
    # Seasonality strength, 0-1: share of the deviation of the peak ratio from flat (capped).
    out["seasonality_strength"] = (np.log(out["peak_ratio"].clip(lower=0.2)).abs() / np.log(3)).clip(0, 1)
    out["description"] = out["sku"].map(desc)
    return out
