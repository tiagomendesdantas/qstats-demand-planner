"""Text helpers shared by the data layer and the planner."""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=32)
def _pattern(keywords: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b")


def has_keyword(text: object, keywords: list[str] | tuple[str, ...]) -> bool:
    """True when any keyword appears as a whole word (TREE matches "TREE" but not "STREET")."""
    return bool(keywords) and _pattern(tuple(keywords)).search(str(text).upper()) is not None
