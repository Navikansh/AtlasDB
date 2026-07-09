"""
Query Planner
--------------
Decides how to execute a search, not *what* the answer is. Two
independent decisions, each a simple heuristic today but each a real
extension point:

1. Which index strategy to use (brute force vs. ANN), based on collection
   size. Below a threshold, building/traversing an ANN structure costs more
   than just scanning everything.
2. Execution order when a metadata filter is present: filter-first (apply
   the metadata predicate, then brute-force search only the surviving
   candidates) vs. search-first (run ANN search, then drop results that
   fail the filter). Filter-first wins when the filter is highly selective;
   search-first wins when it isn't.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IndexStrategy(str, Enum):
    BRUTE_FORCE = "brute_force"
    ANN = "ann"


class FilterOrder(str, Enum):
    NONE = "none"
    FILTER_FIRST = "filter_first"
    SEARCH_FIRST = "search_first"


@dataclass
class ExecutionPlan:
    index_strategy: IndexStrategy
    filter_order: FilterOrder
    reason: str


class QueryPlanner:
    def __init__(self, brute_force_threshold: int = 5000,
                 selective_filter_threshold: float = 0.1):
        """
        brute_force_threshold: collections smaller than this always use
            brute force, regardless of which ANN index is configured --
            below this size an ANN structure adds overhead without a
            latency win.
        selective_filter_threshold: if we have an estimate of what fraction
            of the collection a metadata filter will pass, fractions below
            this trigger filter-first (the filter is selective enough that
            scanning the smaller filtered set is cheaper than over-fetching
            from ANN search).
        """
        self.brute_force_threshold = brute_force_threshold
        self.selective_filter_threshold = selective_filter_threshold

    def plan(self, collection_size: int, has_metadata_filter: bool,
             estimated_filter_selectivity: float | None = None) -> ExecutionPlan:
        if collection_size < self.brute_force_threshold:
            index_strategy = IndexStrategy.BRUTE_FORCE
            reason = f"collection size {collection_size} < threshold {self.brute_force_threshold}"
        else:
            index_strategy = IndexStrategy.ANN
            reason = f"collection size {collection_size} >= threshold {self.brute_force_threshold}"

        if not has_metadata_filter:
            filter_order = FilterOrder.NONE
        elif (estimated_filter_selectivity is not None
              and estimated_filter_selectivity < self.selective_filter_threshold):
            filter_order = FilterOrder.FILTER_FIRST
            reason += f"; filter selectivity {estimated_filter_selectivity:.3f} is selective -> filter-first"
        elif estimated_filter_selectivity is not None:
            filter_order = FilterOrder.SEARCH_FIRST
            reason += f"; filter selectivity {estimated_filter_selectivity:.3f} is loose -> search-first"
        else:
            # no selectivity estimate available - default to filter-first,
            # since a wrong guess there is cheaper (worst case: filtering a
            # slightly-too-large set) than a wrong guess the other way
            # (worst case: not enough ANN results survive the filter at all)
            filter_order = FilterOrder.FILTER_FIRST
            reason += "; no selectivity estimate -> defaulting to filter-first"

        return ExecutionPlan(index_strategy=index_strategy, filter_order=filter_order, reason=reason)
