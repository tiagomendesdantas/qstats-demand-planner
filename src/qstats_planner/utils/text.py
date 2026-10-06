"""Text helpers shared by the data layer and the planner."""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=64)
def _pattern(keywords: tuple[str, ...], plurals: bool) -> re.Pattern:
    suffix = r"(?:S|ES)?" if plurals else ""
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")" + suffix + r"\b")


def has_keyword(text: object, keywords: list[str] | tuple[str, ...], plurals: bool = False) -> bool:
    """True when any keyword appears as a whole word (TREE matches "TREE" but not "STREET"; TIN does
    not match "GREETING"). With `plurals`, an S or ES ending also counts (MUG matches "MUGS")."""
    return bool(keywords) and _pattern(tuple(keywords), plurals).search(str(text).upper()) is not None
