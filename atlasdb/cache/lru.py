"""
LRU Embedding Cache
----------------------
Caches text -> embedding so repeated queries (or repeated inserts of near-duplicate text) skip the model call entirely. 
Hand-rolled with OrderedDict rather than `functools.lru_cache` for two reasons: (1) the key needs to be a string, 
not a model instance, and (2) exposing hit/miss/evict counts for the `/stats` endpoint requires wrapping it anyway, 
so there's no real time saved by using the stdlib decorator.

O(1) get/put via OrderedDict.move_to_end() / popitem(last=False).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class LRUCache:
    def __init__(self, capacity: int = 10_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> np.ndarray | None:
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }
