"""
IVF (Inverted File) Index
---------------------------
1. Train `n_clusters` centroids over the collection with k-means.
2. Assign every vector to its nearest centroid -> inverted lists
   (centroid_id -> [vector ids in that cluster]).
3. At search time, find the `nprobe` nearest centroids to the query, then
   brute-force search only within those clusters' inverted lists.

This is the accuracy/speed knob: nprobe=n_clusters degrades to
brute force (100% recall, no speedup); nprobe=1 is fastest but can miss
vectors sitting near a cluster boundary that got assigned to a different
centroid than the query is closest to.
"""
from __future__ import annotations

import numpy as np

from atlasdb.distance import top_k
from atlasdb.indexes.base import VectorIndex
from atlasdb.indexes.ivf import kmeans


class IVFIndex(VectorIndex):
    def __init__(self, dim: int, distance_metric: str = "cosine",
                 n_clusters: int = 32, nprobe: int = 4):
        super().__init__(dim, distance_metric)
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self._centroids: np.ndarray | None = None
        self._lists: dict[int, list[str]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._assignment: dict[str, int] = {}

    # -- build / maintain ----------------------------------------------------

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        self._vectors = {id_: vectors[i] for i, id_ in enumerate(ids)}
        if len(ids) == 0:
            self._centroids = np.empty((0, self.dim), dtype=np.float32)
            self._lists = {}
            self._assignment = {}
            return

        self._centroids = kmeans.fit(vectors, self.n_clusters).astype(np.float32)
        assignments = kmeans.assign(vectors, self._centroids)

        self._lists = {c: [] for c in range(len(self._centroids))}
        self._assignment = {}
        for id_, cluster in zip(ids, assignments):
            cluster = int(cluster)
            self._lists[cluster].append(id_)
            self._assignment[id_] = cluster

    def insert(self, id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(self.dim)
        if self._centroids is None or len(self._centroids) == 0:
            # not built yet -- fall back to a single implicit cluster
            self._centroids = vector.reshape(1, self.dim)
            self._lists = {0: []}
        if id in self._assignment:
            self.delete(id)
        c = kmeans.nearest_centroid(vector, self._centroids)
        self._vectors[id] = vector
        self._lists.setdefault(c, []).append(id)
        self._assignment[id] = c

    def delete(self, id: str) -> None:
        if id not in self._assignment:
            raise KeyError(id)
        c = self._assignment.pop(id)
        self._lists[c].remove(id)
        del self._vectors[id]

    # -- search ----------------------------------------------------------------

    def _probe_clusters(self, query: np.ndarray) -> list[int]:
        d = np.linalg.norm(self._centroids - query, axis=1)
        nprobe = min(self.nprobe, len(self._centroids))
        return list(np.argsort(d)[:nprobe])

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._centroids is None or len(self._centroids) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(self.dim)
        candidate_ids: list[str] = []
        for c in self._probe_clusters(query):
            candidate_ids.extend(self._lists.get(c, []))
        if not candidate_ids:
            return []
        matrix = np.stack([self._vectors[i] for i in candidate_ids])
        return top_k(self.distance_metric, query, matrix, candidate_ids, k)

    def __len__(self) -> int:
        return len(self._vectors)
