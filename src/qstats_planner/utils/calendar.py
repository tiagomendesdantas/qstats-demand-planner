"""Trading calendar and week indexing.

The source retailer did not trade on Saturdays and closed for about twelve days over the year end.
A closed day is not a zero-demand day, so demand is modelled as a rate per trading day and weekly
forecasts are multiplied by the number of trading days in each week (the "exposure").
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def shift_dates(values: pd.Series | pd.DatetimeIndex, weeks: int) -> pd.Series | pd.DatetimeIndex:
    """Move dates forward by a whole number of weeks (weekday and season are preserved)."""
    return values + pd.Timedelta(weeks=weeks)


def week_start(d: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    """Monday of the ISO week containing each date."""
    if isinstance(d, pd.Series):
        return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()
    return (d - pd.to_timedelta(d.dayofweek, unit="D")).normalize()


def _in_shutdown(days: pd.DatetimeIndex, start_md: tuple[int, int], end_md: tuple[int, int]) -> np.ndarray:
    md = days.month * 100 + days.day
    s = start_md[0] * 100 + start_md[1]
    e = end_md[0] * 100 + end_md[1]
    if s <= e:
        return (md >= s) & (md <= e)
    return (md >= s) | (md <= e)


def projected_trading_days(start: date | pd.Timestamp, end: date | pd.Timestamp, cfg: dict) -> pd.DatetimeIndex:
    """Trading days after the end of the data, using the pattern observed in the data."""
    cal = cfg["calendar"]
    days = pd.date_range(start, end, freq="D")
    closed = np.isin(days.dayofweek, cal["closed_weekdays"])
    shut = _in_shutdown(days, tuple(cal["shutdown_start_month_day"]), tuple(cal["shutdown_end_month_day"]))
    return days[~closed & ~shut]


class TradingCalendar:
    """Observed trading days inside the data, projected trading days after it."""

    def __init__(self, observed_days: pd.DatetimeIndex, data_end: pd.Timestamp, cfg: dict):
        self.observed = pd.DatetimeIndex(sorted(set(observed_days.normalize())))
        self.data_end = pd.Timestamp(data_end).normalize()
        self.cfg = cfg

    def trading_days(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        inside = self.observed[(self.observed >= start) & (self.observed <= min(end, self.data_end))]
        if end <= self.data_end:
            return inside
        after = projected_trading_days(max(start, self.data_end + pd.Timedelta(days=1)), end, self.cfg)
        return inside.append(after)

    def is_trading(self, days: pd.DatetimeIndex) -> np.ndarray:
        days = pd.DatetimeIndex(days).normalize()
        inside = days <= self.data_end
        out = np.zeros(len(days), dtype=bool)
        out[inside] = days[inside].isin(self.observed)
        if (~inside).any():
            proj = projected_trading_days(days[~inside].min(), days[~inside].max(), self.cfg)
            out[~inside] = days[~inside].isin(proj)
        return out

    def weekly_exposure(self, weeks: pd.DatetimeIndex) -> np.ndarray:
        """Trading days in each week (weeks given by their Monday)."""
        weeks = pd.DatetimeIndex(weeks)
        if len(weeks) == 0:
            return np.zeros(0)
        days = pd.date_range(weeks.min(), weeks.max() + pd.Timedelta(days=6), freq="D")
        trading = pd.Series(self.is_trading(days).astype(float), index=days)
        per_week = trading.groupby(week_start(trading.index)).sum()
        return per_week.reindex(weeks).fillna(0.0).to_numpy()
