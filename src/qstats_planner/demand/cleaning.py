"""Cleaning of canonical sales lines. Source-agnostic: works on any adapter's output.

Decisions (explained in docs/data_methodology.md):

1. Exact duplicate lines are removed.
2. Adjustments (bad debt, stock corrections) and non-merchandise lines (postage, fees, vouchers,
   test products) are not demand and are removed.
3. Lines with a non-positive price are write-offs or corrections, not sales, and are removed.
4. Returns are never subtracted from demand blindly. A return that exactly matches an earlier sale
   (same customer, SKU and quantity) marks that sale as cancelled: the order was not fulfilled
   demand, so the sale is removed at its own date ("netted at the sale date"). A return with no
   match stays a return on its own date and does not reduce demand.
5. Very large single lines (wholesale lots) are flagged. They stay in the record, and they are
   left out of the demand the simulated business faces, because a handful of them would decide
   every fill-rate metric on their own.
6. Missing descriptions are filled with the SKU's most common description.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.domain.contract import LineType


def drop_exact_duplicates(lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(lines)
    out = lines.drop_duplicates(
        subset=["order_id", "order_ts", "sku", "description", "quantity", "unit_price", "customer_id", "country"]
    )
    return out.reset_index(drop=True), before - len(out)


def fill_descriptions(lines: pd.DataFrame) -> pd.DataFrame:
    known = lines.dropna(subset=["description"])
    known = known[known["description"].str.len() > 0]
    modal = known.groupby("sku")["description"].agg(lambda s: s.value_counts().index[0])
    out = lines.copy()
    out["description"] = out["description"].fillna(out["sku"].map(modal))
    out["description"] = out["description"].fillna("ITEM " + out["sku"])
    out["description"] = out["sku"].map(modal).fillna(out["description"])  # one name per SKU
    return out


def match_cancellations(lines: pd.DataFrame) -> pd.DataFrame:
    """Flag sales cancelled by a later return of the same customer, SKU and quantity.

    Each return is matched to the most recent earlier sale with the same key; a sale is matched at
    most once. Lines without a customer cannot be matched and stay as they are.
    """
    out = lines.copy()
    out["cancelled"] = False
    out["matched_sale"] = False
    sales = out[(out["line_type"] == LineType.SALE) & out["customer_id"].notna()].copy()
    rets = out[(out["line_type"] == LineType.RETURN) & out["customer_id"].notna()].copy()
    if sales.empty or rets.empty:
        return out
    sales["key"] = sales["customer_id"] + "|" + sales["sku"] + "|" + sales["quantity"].astype(str)
    rets["key"] = rets["customer_id"] + "|" + rets["sku"] + "|" + (-rets["quantity"]).astype(str)
    sales = sales.reset_index(names="sale_idx").sort_values("order_ts")
    rets = rets.reset_index(names="ret_idx").sort_values("order_ts")
    matched = pd.merge_asof(
        rets[["ret_idx", "order_ts", "key"]],
        sales[["sale_idx", "order_ts", "key"]],
        on="order_ts",
        by="key",
        direction="backward",
    ).dropna(subset=["sale_idx"])
    matched = matched.drop_duplicates(subset=["sale_idx"], keep="last")
    out.loc[matched["sale_idx"].astype(int).to_numpy(), "cancelled"] = True
    out.loc[matched["ret_idx"].astype(int).to_numpy(), "matched_sale"] = True
    return out


def flag_bulk_lines(lines: pd.DataFrame, multiple: float, min_units: float) -> pd.DataFrame:
    out = lines.copy()
    sale = (out["line_type"] == LineType.SALE) & ~out["cancelled"]
    p95 = out[sale].groupby("sku")["quantity"].quantile(0.95)
    limit = out["sku"].map(p95).fillna(np.inf) * multiple
    out["bulk_order"] = sale & (out["quantity"] > limit) & (out["quantity"] >= min_units)
    return out


def clean_sales_lines(lines: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply every cleaning rule; return the cleaned lines and a count of what each rule did."""
    c = cfg["cleaning"]
    report: dict[str, int] = {"input_rows": len(lines)}
    out, report["exact_duplicates_removed"] = drop_exact_duplicates(lines)

    for kind in (LineType.ADJUSTMENT, LineType.NON_MERCHANDISE):
        mask = out["line_type"] == kind
        report[f"{kind.lower()}_rows_removed"] = int(mask.sum())
        out = out[~mask]
    bad_price = out["unit_price"] <= 0
    report["non_positive_price_rows_removed"] = int(bad_price.sum())
    out = out[~bad_price]
    report["missing_description_rows"] = int(out["description"].isna().sum())
    report["missing_customer_rows"] = int(out["customer_id"].isna().sum())

    out = fill_descriptions(out.reset_index(drop=True))
    out = match_cancellations(out)
    out = flag_bulk_lines(out, c["bulk_line_multiple_of_sku_p95"], c["bulk_line_min_units"])
    out["date"] = out["order_ts"].dt.normalize()

    sale = out["line_type"] == LineType.SALE
    ret = out["line_type"] == LineType.RETURN
    report.update(
        sale_rows=int(sale.sum()),
        return_rows=int(ret.sum()),
        returns_matched_to_sale=int(out["matched_sale"].sum()),
        cancelled_sale_rows=int(out["cancelled"].sum()),
        bulk_rows=int(out["bulk_order"].sum()),
        gross_units_sold=float(out.loc[sale, "quantity"].sum()),
        return_units=float(-out.loc[ret, "quantity"].sum()),
        cancelled_units=float(out.loc[out["cancelled"], "quantity"].sum()),
        bulk_units=float(out.loc[out["bulk_order"], "quantity"].sum()),
        output_rows=len(out),
    )
    return out.reset_index(drop=True), report


def demand_lines(clean: pd.DataFrame) -> pd.DataFrame:
    """Lines that count as fulfilled demand: sales, not cancelled, not wholesale lots."""
    mask = (clean["line_type"] == LineType.SALE) & ~clean["cancelled"] & ~clean["bulk_order"]
    cols = ["date", "order_ts", "order_id", "sku", "customer_id", "quantity", "unit_price"]
    out = clean.loc[mask, cols].rename(columns={"quantity": "units"})
    # A line without a customer keeps its order as the unit that is assigned to a channel.
    out["customer_key"] = out["customer_id"].fillna("order:" + out["order_id"])
    return out.reset_index(drop=True)
