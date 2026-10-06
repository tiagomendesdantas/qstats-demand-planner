"""Order quantity rounding: MOQ and case packs.

The policy computes a raw requirement (units needed to bring the inventory position up to the
order-up-to level). The supplier sells whole cases and will not take an order below its minimum.
Rule: order max(raw requirement, MOQ), rounded UP to a whole number of cases. Rounding up (not to
the nearest case) keeps the service target; the extra units are reported as MOQ/case-pack overbuy.

    >>> round_order(1047, case_pack=24, moq=600)
    1056
"""

from __future__ import annotations

import numpy as np


def round_order(raw: float, case_pack: int, moq: int) -> int:
    if raw <= 0:
        return 0
    qty = max(raw, moq)
    return int(np.ceil(qty / case_pack - 1e-9) * case_pack)


def round_orders(raw: np.ndarray, case_pack: np.ndarray, moq: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, float)
    qty = np.maximum(raw, moq)
    out = np.ceil(qty / case_pack - 1e-9) * case_pack
    return np.where(raw > 0, out, 0).astype(int)


def round_up_to_pack(raw: np.ndarray, case_pack: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, float)
    return np.where(raw > 0, np.ceil(raw / case_pack - 1e-9) * case_pack, 0).astype(int)


def split_by_share(
    qty: np.ndarray, east_share: np.ndarray, case_pack: np.ndarray, preferred_east: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split an order between the two DCs in whole cases; the remainder goes to the preferred DC."""
    qty = np.asarray(qty, int)
    cases = qty // case_pack
    east_cases = np.floor(cases * np.clip(east_share, 0, 1)).astype(int)
    rest = cases - east_cases
    # the rounding remainder (at most one case) goes to the preferred DC
    frac = cases * np.clip(east_share, 0, 1) - east_cases
    move = (frac > 0) & preferred_east
    east_cases = east_cases + (move & (rest > 0)).astype(int)
    east = east_cases * case_pack
    return east, qty - east
