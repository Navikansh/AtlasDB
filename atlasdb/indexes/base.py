"""
VectorIndex Interface
-----------------------
Every ANN algorithm (brute force, IVF, HNSW) implements this same contract.
The query planner (atlasdb/planner/planner.py) dispatches against this
interface, not against any concrete class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    """Common interface for all ANN index implementations."""

    def __init__(self, dim: int, distance_metric: str = "cosine"):
        self.dim = dim
        self.distance_metric = distance_metric

    @abstractmethod
    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        """Bulk-build the index from scratch given all ids/vectors at once."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, id: str, vector: np.ndarray) -> None:
        """Insert a single vector into an already-built (or empty) index."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: str) -> None:
        """Remove a vector from the index by id."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (id, score) pairs, best match first."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
