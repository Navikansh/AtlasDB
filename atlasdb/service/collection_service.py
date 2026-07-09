"""
Collection Service
---------------------
One `CollectionService` per collection. Owns that collection's storage, its two index instances 
(a brute-force one kept around for small/no-filter queries, and whichever ANN index the config specifies), 
its planner, and its metrics collector.

This is where the query pipeline actually executes:

    query vector
      -> QueryPlanner.plan()          (brute force vs ANN; filter-first vs search-first)
      -> metadata filter (if any)      (atlasdb/service/filtering.py)
      -> VectorIndex.search()           (whichever index the plan picked)
      -> ranked top-k results

`AtlasService` (atlas_service.py) is the layer above this one - it holds one `CollectionService` per collection name, 
plus the shared embedding provider and cache.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from atlasdb.config.loader import AtlasConfig
from atlasdb.distance import top_k
from atlasdb.indexes.base import VectorIndex
from atlasdb.metrics.collector import MetricsCollector
from atlasdb.planner.planner import IndexStrategy, QueryPlanner
from atlasdb.service.filtering import matches_filter
from atlasdb.service.index_factory import make_index
from atlasdb.storage.collection_manager import CollectionManager

logger = logging.getLogger("atlasdb.service")


class CollectionService:
    def __init__(self, root: str, name: str, config: AtlasConfig, dim: int | None = None):
        self.name = name
        self.config = config
        self.storage = CollectionManager(root, name, dim=dim, distance_metric=config.distance_metric)
        self.planner = QueryPlanner(config.brute_force_threshold, config.selective_filter_threshold)
        self.metrics = MetricsCollector()

        self._brute_force_index: VectorIndex | None = None
        self._ann_index: VectorIndex | None = None
        self._dirty = True  # whether the in-memory indexes need a rebuild from storage

    # -- write path -----------------------------------------------------------

    def insert(self, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        self.storage.insert(id, vector, metadata)
        self.metrics.record_insert()
        self._dirty = True

    def delete(self, id: str) -> None:
        self.storage.delete(id)
        self._dirty = True

    def _rebuild_indexes_if_needed(self) -> None:
        if not self._dirty:
            return
        dim = self.storage.dim
        ids = self.storage.all_ids()
        vectors = np.stack([self.storage.get(i).vector for i in ids]) if ids else np.empty((0, dim or 0))

        self._brute_force_index = make_index(self.config, "brute_force", dim)
        self._brute_force_index.build(ids, vectors)

        self._ann_index = make_index(self.config, self.config.index_type, dim)
        self._ann_index.build(ids, vectors)

        self._dirty = False
        logger.info("rebuilt indexes for collection '%s' (%d vectors)", self.name, len(ids))

    # -- read path --------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int,
               metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        self._rebuild_indexes_if_needed()
        collection_size = len(self.storage)
        has_filter = bool(metadata_filter)

        selectivity = self._estimate_selectivity(metadata_filter) if has_filter else None
        plan = self.planner.plan(collection_size, has_filter, selectivity)
        index = (self._brute_force_index if plan.index_strategy == IndexStrategy.BRUTE_FORCE
                 else self._ann_index)

        with self.metrics.time_search():
            if plan.filter_order.value == "filter_first" and has_filter:
                results = self._search_filter_first(query_vector, k, metadata_filter)
            else:
                results = self._search_index_first(index, query_vector, k, metadata_filter)

        return {
            "results": [{"id": id_, "score": s} for id_, s in results],
            "plan": {
                "index_strategy": plan.index_strategy.value,
                "filter_order": plan.filter_order.value,
                "reason": plan.reason,
            },
        }

    def _estimate_selectivity(self, metadata_filter: dict[str, Any]) -> float | None:
        collection_size = len(self.storage)
        if collection_size == 0:
            return None
        matches = sum(
            1 for id_ in self.storage.all_ids()
            if matches_filter(self.storage.get(id_).metadata, metadata_filter)
        )
        return matches / collection_size

    def _search_filter_first(self, query_vector: np.ndarray, k: int,
                              metadata_filter: dict[str, Any]) -> list[tuple[str, float]]:
        candidate_ids = [
            id_ for id_ in self.storage.all_ids()
            if matches_filter(self.storage.get(id_).metadata, metadata_filter)
        ]
        if not candidate_ids:
            return []
        matrix = np.stack([self.storage.get(i).vector for i in candidate_ids])
        return top_k(self.config.distance_metric, query_vector, matrix, candidate_ids, k)

    def _search_index_first(self, index: VectorIndex, query_vector: np.ndarray, k: int,
                             metadata_filter: dict[str, Any] | None) -> list[tuple[str, float]]:
        has_filter = bool(metadata_filter)
        # over-fetch when filtering post-hoc, since some results will be dropped
        fetch_k = k * 5 if has_filter else k
        raw = index.search(query_vector, fetch_k)
        if has_filter:
            raw = [(id_, s) for id_, s in raw if matches_filter(self.storage.get(id_).metadata, metadata_filter)]
        return raw[:k]

    # -- observability ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        self._rebuild_indexes_if_needed()
        return {
            "collection": self.name,
            "size": len(self.storage),
            "dim": self.storage.dim,
            "distance_metric": self.storage.distance_metric,
            "index_type": self.config.index_type,
            **self.metrics.snapshot(),
        }
