"""
EmbeddingProvider Interface
------------------------------
Decouples the rest of AtlasDB from any specific embedding model. The index
and storage layers only ever see vectors of a known `dim` -- they don't
know or care whether those vectors came from MiniLM, BGE, or anything
else. Swapping models is a config change (`configs/config.yaml`), not a
code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a 1-D float32 array of length self.dim."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Default batch implementation: embed one at a time. Providers with
        a real batched model call (e.g. sentence-transformers) should
        override this for throughput."""
        return np.stack([self.embed(t) for t in texts])
