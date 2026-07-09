"""
Metadata Filtering
--------------------
The equality-only metadata predicate the planner and service layer use.
Deliberately simple (exact-match AND across keys, no OR/range/NOT).
"""
from __future__ import annotations

from typing import Any


def matches_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """True if `metadata` satisfies every key/value pair in `filters`
    (implicit AND, exact match only)."""
    return all(metadata.get(key) == expected for key, expected in filters.items())
