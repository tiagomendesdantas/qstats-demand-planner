"""CSV client adapter.

A client exports its tables as CSV files with the canonical column names (see
`docs/client_onboarding.md`). This adapter reads them and validates them against the contract.
It exists to prove the seam: the planner runs on these frames exactly as it runs on UCI-derived
ones. A PostgreSQL or Azure SQL adapter would replace `pd.read_csv` with a query per table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qstats_planner.adapters.base import SourceAdapter
from qstats_planner.domain.contract import TABLES, validate_frame


def _read(path: Path, table: str) -> pd.DataFrame:
    spec = TABLES[table]
    dates = [c for c, kind in spec.items() if kind == "datetime"]
    df = pd.read_csv(path, parse_dates=dates)
    for col, kind in spec.items():
        if kind == "string" and col in df:
            df[col] = df[col].astype("string")
        elif kind == "bool" and col in df:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return validate_frame(df, table)


class CSVClientAdapter(SourceAdapter):
    name = "csv_client"

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def _table(self, table: str) -> pd.DataFrame | None:
        path = self.folder / f"{table}.csv"
        return _read(path, table) if path.exists() else None

    def sales_lines(self) -> pd.DataFrame:
        frame = self._table("sales_lines")
        if frame is None:
            raise FileNotFoundError(self.folder / "sales_lines.csv")
        return frame

    def products(self) -> pd.DataFrame | None:
        return self._table("products")

    def suppliers(self) -> pd.DataFrame | None:
        return self._table("suppliers")

    def purchase_orders(self) -> pd.DataFrame | None:
        return self._table("purchase_orders")

    def inventory_snapshots(self) -> pd.DataFrame | None:
        return self._table("inventory_snapshots")

    def locations(self) -> pd.DataFrame | None:
        return self._table("locations")
