"""Seasonal prior: a pooled monthly shape, fixed before the planner runs.

Why not per-SKU Holt-Winters: the history holds at most two yearly cycles, and at the moment
QStats takes over it holds one. A 52-week seasonal index per SKU would have no degrees of freedom
left. Instead the planner uses a prior built once, from other products:

    panel      SKUs outside the demo and dev populations that sold through the whole first year
               (a constant panel, so launches and deaths do not create a fake shape)
    per SKU    monthly rate per trading day; for non-seasonal products a log-linear trend
               fitted on January-August is removed; each month is divided by the SKU's own mean
    pooled     median across SKUs, per month, in two groups chosen by description keyword
               (Christmas-type products and the rest), never by realised shape
    shrunk     in log space, log s = lambda log s_raw with lambda = n / (n + k), n = panel SKUs in
               the group and k fixed in the config (multiplicative factors shrink toward 1
               evenly above and below it), then renormalised to mean 1
    daily      linear interpolation between mid-month points, so weeks change smoothly

The prior is applied by policy and ablated in the comparison ("Legacy + the same prior" is a row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MID_MONTH_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)


class SeasonalPrior:
    def __init__(self, monthly: dict[int, np.ndarray], sku_group: np.ndarray, info: dict | None = None):
        self.monthly = monthly          # group -> 12 factors (mean 1)
        self.sku_group = np.asarray(sku_group, int)
        self.info = info or {}

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(cls, daily: pd.DataFrame, exclude: set[str], start: pd.Timestamp, weeks: int,
            keywords: list[str], shrink_k: float, min_units: float = 100.0) -> SeasonalPrior:
        end = pd.Timestamp(start) + pd.Timedelta(weeks=weeks) - pd.Timedelta(days=1)
        d = daily[(daily["date"] >= start) & (daily["date"] <= end) & ~daily["sku"].isin(exclude)]
        span = d.groupby("sku")["date"].agg(["min", "max"])
        const = span[(span["min"] <= start + pd.Timedelta(weeks=4)) & (span["max"] >= end - pd.Timedelta(days=6))].index
        d = d[d["sku"].isin(const)]
        tot = d.groupby("sku")["units"].sum()
        d = d[d["sku"].isin(tot[tot >= min_units].index)]
        d = d.assign(month=d["date"].dt.month)
        rate = d.groupby(["sku", "month"])["units"].sum() / d.groupby(["sku", "month"]).size()
        rate = rate.unstack("month").reindex(columns=range(1, 13))
        # remove each SKU's January-August log-linear trend (calendar order of the window)
        order = (np.arange(1, 13) - start.month) % 12
        x = order.astype(float)
        off = np.isin(np.arange(1, 13), range(1, 9))
        logr = np.log(rate.to_numpy() + 0.05)
        xo = x[off] - x[off].mean()
        slope = ((logr[:, off] - logr[:, off].mean(axis=1, keepdims=True)) * xo).sum(axis=1) / (xo**2).sum()
        desc = daily.drop_duplicates("sku").set_index("sku")["description"]
        is_seasonal = rate.index.map(lambda s: any(k in str(desc.get(s, "")) for k in keywords)).to_numpy(bool)
        # Christmas-type products build up from January to August: that is their season, not a
        # trend, so they are not de-trended.
        slope = np.where(is_seasonal, 0.0, slope)
        detr = np.exp(logr - slope[:, None] * (x - x.mean())[None, :])
        ratio = detr / detr.mean(axis=1, keepdims=True)
        monthly, info = {}, {"panel_skus": int(len(rate)), "seasonal_panel_skus": int(is_seasonal.sum())}
        for g, mask in ((0, ~is_seasonal), (1, is_seasonal)):
            if mask.sum() < 10:  # too few products: fall back to the whole panel
                mask = np.ones(len(ratio), bool)
            raw = np.nanmedian(ratio[mask], axis=0)
            raw = raw / raw.mean()
            lam = mask.sum() / (mask.sum() + shrink_k)
            f = np.exp(lam * np.log(raw))
            monthly[g] = f / f.mean()
            info[f"group_{g}_raw"] = np.round(raw, 3).tolist()
            info[f"group_{g}_lambda"] = round(float(lam), 3)
        info["shrink_k"] = shrink_k
        return cls(monthly, np.zeros(0, int), info)

    def for_skus(self, descriptions: pd.Series, keywords: list[str]) -> SeasonalPrior:
        grp = descriptions.map(lambda s: int(any(k in str(s) for k in keywords))).to_numpy()
        return SeasonalPrior(self.monthly, grp, self.info)

    def neutral(self) -> SeasonalPrior:
        """Same interface, every factor 1 (the 'no seasonal prior' arm of the ablation)."""
        return SeasonalPrior({g: np.ones(12) for g in self.monthly}, self.sku_group, {"neutral": True})

    # ------------------------------------------------------------------ evaluate

    def daily_factors(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """(n_dates, n_sku) factor for each calendar day."""
        doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(float)
        xs = np.concatenate([MID_MONTH_DOY - 365, MID_MONTH_DOY, MID_MONTH_DOY + 365])
        cols = {}
        for g, f in self.monthly.items():
            cols[g] = np.interp(doy, xs, np.tile(f, 3))
        out = np.empty((len(doy), len(self.sku_group)))
        for g, v in cols.items():
            out[:, self.sku_group == g] = v[:, None]
        return out

    def weekly_from_daily(self, days: pd.DatetimeIndex, n_weeks: int) -> np.ndarray:
        f = self.daily_factors(days[: n_weeks * 7])
        return f.reshape(n_weeks, 7, -1).mean(axis=1)

    def weekly_factors(self, view, past: bool = True) -> np.ndarray:
        n_weeks = (view.t + 1) // 7
        return self.weekly_from_daily(view.all_days, n_weeks)

    def mean_factor_ahead(self, view, horizon_days: np.ndarray) -> np.ndarray:
        """Average daily factor over the next `horizon_days` (per SKU)."""
        h = int(np.max(horizon_days)) + 1
        ahead = pd.date_range(view.date + pd.Timedelta(days=1), periods=h, freq="D")
        f = self.daily_factors(ahead)
        csum = np.cumsum(f, axis=0)
        idx = np.clip(np.asarray(horizon_days, int) - 1, 0, h - 1)
        return csum[idx, np.arange(f.shape[1])] / (idx + 1)
