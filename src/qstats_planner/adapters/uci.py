"""UCI Online Retail II adapter.

The only module in the project that knows the UCI column names (Invoice, StockCode, Description,
Quantity, InvoiceDate, Price, Customer ID, Country). It maps the file to canonical sales lines and
classifies each line; it does not drop anything. Cleaning decisions are made downstream, on the
canonical table, by `demand.cleaning`.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D. CC BY 4.0.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qstats_planner.adapters.base import SourceAdapter
from qstats_planner.domain.contract import LineType, validate_frame

SHEETS = ("Year 2009-2010", "Year 2010-2011")
_DTYPES = {"Invoice": str, "StockCode": str, "Description": str}


def read_workbook(path: Path, xlsx_name: str = "online_retail_II.xlsx") -> dict[str, pd.DataFrame]:
    """Read both sheets from the zip (as downloaded) or from a loose xlsx (manual placement)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
            data = io.BytesIO(zf.read(name))
        return pd.read_excel(data, sheet_name=None, dtype=_DTYPES)
    return pd.read_excel(path, sheet_name=None, dtype=_DTYPES)


def combine_sheets(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Stack the two sheets. They overlap (1-9 Dec 2010); invoices in the first sheet win."""
    first, second = sheets[SHEETS[0]], sheets[SHEETS[1]]
    overlap = second["Invoice"].isin(set(first["Invoice"]))
    out = pd.concat([first, second[~overlap]], ignore_index=True)
    return out, {"rows_in_file": len(first) + len(second), "sheet_overlap_rows": int(overlap.sum())}


def classify(raw: pd.DataFrame, merchandise_pattern: str) -> pd.Series:
    """Line type from the UCI conventions.

    "C" invoices are cancellations; "A" invoices are bad-debt adjustments; stock codes that are
    not product codes (POST, DOT, M, BANK CHARGES, AMAZONFEE, gift vouchers, TEST...) are
    non-merchandise. A negative quantity on a normal invoice is a stock adjustment, not a sale.
    """
    invoice = raw["Invoice"].astype(str).str.strip()
    code = raw["StockCode"].astype(str).str.strip().str.upper()
    merch = code.str.match(merchandise_pattern)
    kind = np.select(
        [
            invoice.str.startswith("A"),
            ~merch,
            invoice.str.startswith("C"),
            raw["Quantity"] < 0,
        ],
        [LineType.ADJUSTMENT, LineType.NON_MERCHANDISE, LineType.RETURN, LineType.ADJUSTMENT],
        default=LineType.SALE,
    )
    return pd.Series(kind, index=raw.index, dtype="string")


def to_sales_lines(raw: pd.DataFrame, merchandise_pattern: str) -> pd.DataFrame:
    customer = raw["Customer ID"]
    customer = customer.map(lambda v: None if pd.isna(v) else str(int(v)))
    out = pd.DataFrame(
        {
            "order_id": raw["Invoice"].astype("string").str.strip(),
            "order_ts": pd.to_datetime(raw["InvoiceDate"]),
            "sku": raw["StockCode"].astype("string").str.strip().str.upper(),
            "description": raw["Description"].astype("string").str.strip(),
            "quantity": raw["Quantity"].astype(float),
            "unit_price": raw["Price"].astype(float),
            "customer_id": customer.astype("string"),
            "country": raw["Country"].astype("string"),
            "line_type": classify(raw, merchandise_pattern),
        }
    )
    return validate_frame(out, "sales_lines")


class UCIAdapter(SourceAdapter):
    name = "uci_online_retail_ii"

    def __init__(self, path: Path, merchandise_pattern: str = r"^\d{5}[A-Z]{0,3}$"):
        self.path = Path(path)
        self.merchandise_pattern = merchandise_pattern
        self.report: dict = {}

    def sales_lines(self) -> pd.DataFrame:
        sheets = read_workbook(self.path)
        raw, report = combine_sheets(sheets)
        self.report = report
        return to_sales_lines(raw, self.merchandise_pattern)


def is_merchandise(code: str, pattern: str = r"^\d{5}[A-Z]{0,3}$") -> bool:
    return bool(re.match(pattern, code.strip().upper()))
