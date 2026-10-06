"""Daily SKU demand and SKU lifecycles.

Missing SKU-days are filled with zero only inside a SKU's active life (first sale to last sale,
or to the end of the data when the SKU has not disappeared) and only on trading days. A SKU is not
extended backwards before its first sale or forwards after it has clearly stopped selling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.domain.contract import LineType


def sku_lifecycle(demand: pd.DataFrame, data_end: pd.Timestamp, disappeared_after_weeks: int) -> pd.DataFrame:
    life = demand.groupby("sku")["date"].agg(first_seen_date="min", last_seen_date="max")
    cutoff = pd.Timestamp(data_end) - pd.Timedelta(weeks=disappeared_after_weeks)
    life["disappeared"] = life["last_seen_date"] < cutoff
    life["active_until"] = life["last_seen_date"].where(life["disappeared"], pd.Timestamp(data_end))
    life["history_days"] = (life["active_until"] - life["first_seen_date"]).dt.days + 1
    return life.reset_index()


def daily_sku_demand(clean: pd.DataFrame, trading_days: pd.DatetimeIndex, life: pd.DataFrame) -> pd.DataFrame:
    """date, sku, description, units, revenue, average_price, gross_units_sold, returns, net_units."""
    sale = clean["line_type"] == LineType.SALE
    demand = sale & ~clean["cancelled"] & ~clean["bulk_order"]
    g = (
        clean.assign(
            units=np.where(demand, clean["quantity"], 0.0),
            revenue=np.where(demand, clean["quantity"] * clean["unit_price"], 0.0),
            gross_units_sold=np.where(sale, clean["quantity"], 0.0),
            returns=np.where(clean["line_type"] == LineType.RETURN, -clean["quantity"], 0.0),
        )
        .groupby(["sku", "date"])[["units", "revenue", "gross_units_sold", "returns"]]
        .sum()
    )

    frames = []
    trading = pd.DatetimeIndex(trading_days)
    for row in life.itertuples(index=False):
        days = trading[(trading >= row.first_seen_date) & (trading <= row.active_until)]
        frames.append(pd.DataFrame({"sku": row.sku, "date": days}))
    grid = pd.concat(frames, ignore_index=True)
    out = grid.merge(g.reset_index(), on=["sku", "date"], how="left").fillna(
        {"units": 0.0, "revenue": 0.0, "gross_units_sold": 0.0, "returns": 0.0}
    )
    out["net_units"] = out["gross_units_sold"] - out["returns"]
    out["average_price"] = np.where(out["units"] > 0, out["revenue"] / out["units"].where(out["units"] > 0), np.nan)
    desc = clean.drop_duplicates("sku").set_index("sku")["description"]
    out["description"] = out["sku"].map(desc)
    cols = ["date", "sku", "description", "units", "revenue", "average_price", "gross_units_sold", "returns", "net_units"]
    return out[cols].sort_values(["sku", "date"]).reset_index(drop=True)


def window_daily(daily: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Daily demand up to `end`, zero-filled on trading days from each SKU's first sale to `end`.

    Uses only rows dated on or before `end`, so a SKU that went quiet near the end of the window
    looks the same whether or not it sold again later.
    """
    end = pd.Timestamp(end)
    d = daily[daily["date"] <= end]
    trading = pd.DatetimeIndex(sorted(d["date"].unique()))
    first = d[d["units"] > 0].groupby("sku")["date"].min()
    frames = [pd.DataFrame({"sku": sku, "date": trading[trading >= f]}) for sku, f in first.items()]
    grid = pd.concat(frames, ignore_index=True)
    out = grid.merge(d, on=["sku", "date"], how="left")
    for col in ("units", "revenue", "gross_units_sold", "returns", "net_units"):
        if col in out:
            out[col] = out[col].fillna(0.0)
    if "description" in out:
        out["description"] = out["sku"].map(d.drop_duplicates("sku").set_index("sku")["description"])
    return out.sort_values(["sku", "date"]).reset_index(drop=True)
