"""
HNSW Index
------------
Wires `HNSWGraph` (the data structure, graph.py) and the greedy search
algorithm (search.py) into the `VectorIndex` interface every index in
this project implements. 

Insertion algorithm:
1. Pick a random level for the new node (exponentially distributed - most nodes only exist at layer 0).
2. Starting from the current entry point, greedily descend from the top layer down to just above the new node's level, 
    using ef=1 at each layer.
3. From the new node's level down to layer 0, run a wider search (ef=ef_construction) to find candidate neighbors, 
    connect to the M closest, and for each of those neighbors - trim their neighbor list back down to M if this 
    new edge pushed them over the cap.
"""
from __future__ import annotations

import numpy as np

from atlasdb.distance import higher_is_better, score
from atlasdb.indexes.base import VectorIndex
from atlasdb.indexes.hnsw.graph import HNSWGraph
from atlasdb.indexes.hnsw.search import search_layer, select_closest


class HNSWIndex(VectorIndex):
    def __init__(self, dim: int, distance_metric: str = "cosine",
                 M: int = 16, ef_construction: int = 100, ef_search: int = 50,
                 seed: int = 0):
        super().__init__(dim, distance_metric)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._graph = HNSWGraph(M=M, seed=seed)
        self._higher_is_better = higher_is_better(distance_metric)

    # -- introspection (used by benchmarks/run_benchmarks.py for a memory estimate) --

    @property
    def M(self) -> int:
        return self._graph.M

    @property
    def M0(self) -> int:
        return self._graph.M0

    # -- distance helper -------------------------------------------------------

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Lower-is-better distance, regardless of the configured metric's
        native direction, so graph.py / search.py stay metric-agnostic."""
        s = float(score(self.distance_metric, a, b.reshape(1, -1))[0])
        return -s if self._higher_is_better else s

    # -- build / maintain ----------------------------------------------------

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        self._graph = HNSWGraph(M=self._graph.M, seed=0)
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        for id_, vec in zip(ids, vectors):
            self.insert(id_, vec)

    def insert(self, id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(self.dim)
        g = self._graph
        if id in g.vectors:
            self.delete(id)

        level = g.random_level()

        if g.entry_point is None:
            g.add_node(id, vector, level)
            return

        entry = {g.entry_point}
        # descend from the top layer to just above this node's level, ef=1
        for layer in range(g.max_level, level, -1):
            nearest = search_layer(g.vectors, g.neighbors[layer], vector, entry,
                                    ef=1, dist=self._dist)
            if nearest:
                entry = {nearest[0][1]}

        g.add_node(id, vector, level)

        # at and below this node's level, connect to the best candidates found
        for layer in range(min(level, g.max_level), -1, -1):
            candidates = search_layer(g.vectors, g.neighbors[layer], vector, entry,
                                       ef=self.ef_construction, dist=self._dist)
            m = g.max_neighbors_for_layer(layer)
            candidate_ids = [c_id for _, c_id in candidates]
            chosen = select_closest(g.vectors, vector, candidate_ids, m, self._dist)

            for neighbor_id in chosen:
                g.connect(layer, id, neighbor_id)
                self._trim_neighbors(neighbor_id, layer, m)

            entry = set(candidate_ids) if candidate_ids else entry

    def _trim_neighbors(self, id: str, layer: int, m: int) -> None:
        g = self._graph
        current = g.neighbors_at(layer, id)
        if len(current) <= m:
            return
        kept = select_closest(g.vectors, g.vectors[id], current, m, self._dist)
        g.set_neighbors(layer, id, set(kept))

    def delete(self, id: str) -> None:
        if id not in self._graph.vectors:
            raise KeyError(id)
        self._graph.remove_node(id)

    # -- search ----------------------------------------------------------------

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        g = self._graph
        if g.entry_point is None:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(self.dim)

        entry = {g.entry_point}
        for layer in range(g.max_level, 0, -1):
            nearest = search_layer(g.vectors, g.neighbors[layer], query, entry,
                                    ef=1, dist=self._dist)
            if nearest:
                entry = {nearest[0][1]}

        ef = max(self.ef_search, k)
        results = search_layer(g.vectors, g.neighbors[0], query, entry, ef=ef, dist=self._dist)[:k]

        return [
            (id_, -dist if self._higher_is_better else dist)
            for dist, id_ in results
        ]

    def __len__(self) -> int:
        return len(self._graph)
