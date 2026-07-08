"""
Hashing Fallback Provider
----------------------------
Deterministic, dependency-free, and NOT semantically meaningful. Just used a a fallback for purpose of testing.

Implementation: hashed bag-of-words. Each token's MD5 hash picks a dimension and a sign; the resulting vector is L2-normalized. 
Same text always produces the same vector (useful for cache-hit tests), different text produces different vectors, 
but there is no actual semantic relationship between similar meanings and similar vectors.
"""
from __future__ import annotations

import hashlib

import numpy as np

from atlasdb.embeddings.base import EmbeddingProvider


class HashingFallbackProvider(EmbeddingProvider):
    name = "hashing-fallback"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
