"""Adapter interface.

An adapter turns one source system into canonical tables (see `domain.contract`). The planner
depends on this interface only. Swapping `UCIAdapter` for a client's `PostgresAdapter` or
`AzureSQLAdapter` means writing the mapping queries, not touching forecasting or planning code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SourceAdapter(ABC):
    """Every method returns a frame that passes `validate_frame` for its table."""

    name: str = "source"

    @abstractmethod
    def sales_lines(self) -> pd.DataFrame:
        """Order lines: sales, returns, adjustments, non-merchandise lines."""

    def products(self) -> pd.DataFrame | None:
        """Product master, when the source has one (the UCI file does not)."""
        return None

    def suppliers(self) -> pd.DataFrame | None:
        return None

    def purchase_orders(self) -> pd.DataFrame | None:
        return None

    def inventory_snapshots(self) -> pd.DataFrame | None:
        return None

    def locations(self) -> pd.DataFrame | None:
        return None
