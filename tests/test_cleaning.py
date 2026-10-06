import numpy as np
import pandas as pd

from qstats_planner.adapters.uci import classify, combine_sheets
from qstats_planner.demand.cleaning import clean_sales_lines, demand_lines, match_cancellations
from qstats_planner.demand.daily import daily_sku_demand, sku_lifecycle
from qstats_planner.domain.contract import LineType


def lines(rows):
    df = pd.DataFrame(
        rows,
        columns=["order_id", "order_ts", "sku", "description", "quantity", "unit_price", "customer_id", "country", "line_type"],
    )
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    for c in ("order_id", "sku", "description", "customer_id", "country", "line_type"):
        df[c] = df[c].astype("string")
    df["quantity"] = df["quantity"].astype(float)
    df["unit_price"] = df["unit_price"].astype(float)
    return df


def test_uci_classification():
    raw = pd.DataFrame(
        {
            "Invoice": ["536365", "C536379", "A563185", "536366", "536367"],
            "StockCode": ["85123A", "22423", "B", "POST", "21212"],
            "Quantity": [6, -1, 1, 1, -5],
        }
    )
    kinds = classify(raw, r"^\d{5}[A-Z]{0,3}$").tolist()
    assert kinds == [LineType.SALE, LineType.RETURN, LineType.ADJUSTMENT, LineType.NON_MERCHANDISE, LineType.ADJUSTMENT]


def test_sheet_overlap_kept_once():
    a = pd.DataFrame({"Invoice": ["1", "2"], "x": [1, 2]})
    b = pd.DataFrame({"Invoice": ["2", "3"], "x": [2, 3]})
    out, rep = combine_sheets({"Year 2009-2010": a, "Year 2010-2011": b})
    assert out["Invoice"].tolist() == ["1", "2", "3"] and rep["sheet_overlap_rows"] == 1


def test_cleaning_rules(cfg):
    df = lines(
        [
            ("1", "2024-01-02 10:00", "10001", "MUG", 12, 1.0, "c1", "UK", "SALE"),
            ("1", "2024-01-02 10:00", "10001", "MUG", 12, 1.0, "c1", "UK", "SALE"),  # exact duplicate
            ("2", "2024-01-02 11:00", "POST", "POSTAGE", 1, 18.0, "c1", "UK", "NON_MERCHANDISE"),
            ("3", "2024-01-03 09:00", "10002", "BAG", 5, 0.0, "c2", "UK", "SALE"),  # zero price
            ("4", "2024-01-03 10:00", "10002", "BAG", 6, 2.0, "c2", "UK", "SALE"),  # later cancelled
            ("C5", "2024-01-05 10:00", "10002", "BAG", -6, 2.0, "c2", "UK", "RETURN"),
            ("6", "2024-01-06 10:00", "10002", "BAG", 3, 2.0, "c3", "UK", "SALE"),
            ("C7", "2024-01-07 10:00", "10002", "BAG", -2, 2.0, "c3", "UK", "RETURN"),  # no matching quantity
            ("8", "2024-01-08 10:00", "10003", None, 4, 1.5, None, "UK", "SALE"),  # missing customer kept
            ("8", "2024-01-09 10:00", "10003", "CARD", 1, 1.5, None, "UK", "SALE"),
        ]
    )
    clean, rep = clean_sales_lines(df, cfg)
    assert rep["exact_duplicates_removed"] == 1
    assert rep["non_merchandise_rows_removed"] == 1
    assert rep["non_positive_price_rows_removed"] == 1
    assert rep["cancelled_sale_rows"] == 1 and rep["returns_matched_to_sale"] == 1
    d = demand_lines(clean)
    assert d.loc[d["sku"] == "10002", "units"].sum() == 3  # the cancelled 6 is not demand
    assert d.loc[d["sku"] == "10003", "units"].sum() == 5  # no customer: kept
    assert (clean.loc[clean["sku"] == "10003", "description"] == "CARD").all()  # filled with the SKU's name


def test_cancellation_matches_one_to_one():
    df = lines(
        [
            ("1", "2024-01-02", "10001", "MUG", 6, 1.0, "c1", "UK", "SALE"),
            ("C2", "2024-01-03", "10001", "MUG", -6, 1.0, "c1", "UK", "RETURN"),
            ("C3", "2024-01-04", "10001", "MUG", -6, 1.0, "c1", "UK", "RETURN"),
        ]
    )
    out = match_cancellations(df)
    assert out["cancelled"].sum() == 1 and out["matched_sale"].sum() == 1


def test_bulk_lot_flagged(cfg):
    rows = [("1", f"2024-01-{d:02d}", "10001", "MUG", 12, 1.0, f"c{d}", "UK", "SALE") for d in range(1, 29)]
    rows.append(("99", "2024-01-29", "10001", "MUG", 5000, 0.5, "big", "UK", "SALE"))
    clean, rep = clean_sales_lines(lines(rows), cfg)
    assert rep["bulk_rows"] == 1
    assert demand_lines(clean)["units"].sum() == 12 * 28


def test_lifecycle_and_zero_fill():
    days = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    trading = days[days.dayofweek != 5]
    d = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10", "2024-01-19", "2024-02-01", "2024-03-29"]),
            "sku": ["A", "A", "B", "B"],
            "units": [1.0, 2.0, 3.0, 4.0],
        }
    )
    life = sku_lifecycle(d, days[-1], disappeared_after_weeks=4).set_index("sku")
    assert life.at["A", "disappeared"] and not life.at["B", "disappeared"]
    assert life.at["A", "first_seen_date"] == pd.Timestamp("2024-01-10")
    clean = d.rename(columns={"units": "quantity"}).assign(
        line_type="SALE", cancelled=False, bulk_order=False, unit_price=1.0, description=d["sku"]
    )
    daily = daily_sku_demand(clean, trading, life.reset_index())
    a = daily[daily["sku"] == "A"]
    assert a["date"].min() == pd.Timestamp("2024-01-10") and a["date"].max() == pd.Timestamp("2024-01-19")
    assert (a["date"].dt.dayofweek != 5).all()  # closed days are not zero-demand days
    assert a["units"].sum() == 3 and (a["units"] == 0).sum() == len(a) - 2
    assert np.isnan(a.loc[a["units"] == 0, "average_price"]).all()
