"""
Brute Force Index
------------------
O(n) exact search: scores every vector in the collection against the query.
This is the baseline every other index is benchmarked against - 100% recall by definition, and the ceiling on latency. 
Also what the planner falls back to for small collections, where the overhead of building an HNSW graph or IVF clusters 
isn't worth it.
"""
from __future__ import annotations

import numpy as np

from atlasdb.distance import top_k
from atlasdb.indexes.base import VectorIndex


class BruteForceIndex(VectorIndex):
    def __init__(self, dim: int, distance_metric: str = "cosine"):
        super().__init__(dim, distance_metric)
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._matrix: np.ndarray = np.empty((0, dim), dtype=np.float32)

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        self._ids = list(ids)
        self._id_to_row = {id_: i for i, id_ in enumerate(self._ids)}
        self._matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)

    def insert(self, id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(1, self.dim)
        if id in self._id_to_row:
            self._matrix[self._id_to_row[id]] = vector
            return
        self._id_to_row[id] = len(self._ids)
        self._ids.append(id)
        self._matrix = np.vstack([self._matrix, vector]) if self._matrix.size else vector

    def delete(self, id: str) -> None:
        if id not in self._id_to_row:
            raise KeyError(id)
        row = self._id_to_row.pop(id)
        self._ids.pop(row)
        self._matrix = np.delete(self._matrix, row, axis=0)
        # row indices shifted by one for everything after the deleted row
        for other_id, other_row in self._id_to_row.items():
            if other_row > row:
                self._id_to_row[other_id] = other_row - 1

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        return top_k(self.distance_metric, query, self._matrix, self._ids, k)

    def __len__(self) -> int:
        return len(self._ids)
