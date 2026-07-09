"""
HNSW Graph
------------
The pure data structure underneath HNSW: which vectors exist, which layer each one lives on, 
and which neighbors each vector has at each layer.

Layer assignment: each node's top layer is drawn from an exponential distribution (`_random_level`), 
so higher layers are exponentially sparser than layer 0 - that's what gives HNSW its "skip list"-like
behavior of covering long distances cheaply in the sparse layers.
"""
from __future__ import annotations

import math
import random

import numpy as np


class HNSWGraph:
    def __init__(self, M: int, seed: int = 0):
        self.M = M            # max neighbors per node at layers >= 1
        self.M0 = M * 2        # max neighbors per node at layer 0 (standard HNSW choice)
        self._ml = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        self.vectors: dict[str, np.ndarray] = {}
        self.levels: dict[str, int] = {}
        # neighbors[layer][id] -> set of neighbor ids at that layer
        self.neighbors: list[dict[str, set[str]]] = [dict()]
        self.entry_point: str | None = None
        self.max_level: int = -1

    def __len__(self) -> int:
        return len(self.vectors)

    def random_level(self) -> int:
        return int(-math.log(self._rng.random() + 1e-12) * self._ml)

    def ensure_layer(self, layer: int) -> None:
        while len(self.neighbors) <= layer:
            self.neighbors.append(dict())

    def add_node(self, id: str, vector: np.ndarray, level: int) -> None:
        self.vectors[id] = vector
        self.levels[id] = level
        self.ensure_layer(level)
        for layer in range(level + 1):
            self.neighbors[layer].setdefault(id, set())

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
        elif level > self.max_level:
            self.max_level = level
            self.entry_point = id

    def connect(self, layer: int, a: str, b: str) -> None:
        """Add an undirected edge between a and b at the given layer."""
        self.neighbors[layer].setdefault(a, set()).add(b)
        self.neighbors[layer].setdefault(b, set()).add(a)

    def neighbors_at(self, layer: int, id: str) -> set[str]:
        return self.neighbors[layer].get(id, set())

    def set_neighbors(self, layer: int, id: str, neighbor_ids: set[str]) -> None:
        self.neighbors[layer][id] = set(neighbor_ids)

    def max_neighbors_for_layer(self, layer: int) -> int:
        return self.M0 if layer == 0 else self.M

    def remove_node(self, id: str) -> None:
        level = self.levels[id]
        for layer in range(level + 1):
            for neighbor_id in self.neighbors[layer].get(id, set()):
                self.neighbors[layer][neighbor_id].discard(id)
            self.neighbors[layer].pop(id, None)
        del self.vectors[id]
        del self.levels[id]

        if id == self.entry_point:
            self._pick_new_entry_point()

    def _pick_new_entry_point(self) -> None:
        remaining = list(self.vectors.keys())
        if remaining:
            self.entry_point = remaining[0]
            self.max_level = self.levels[self.entry_point]
        else:
            self.entry_point = None
            self.max_level = -1
