"""
Vectorized Distance Metrics
------------------------------
Batched NumPy implementations: one query vector of shape (d,) scored
against a whole matrix of shape (n, d) in a single call, with no explicit
Python loop over n. This is what every index implementation actually
calls -- see `atlasdb/distance/registry.py` for the metric-name lookup
that dispatches to these.
"""
from __future__ import annotations

import numpy as np


def cosine_vectorized(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between `query` (d,) and each row of `matrix` (n, d).
    Higher is more similar (range roughly [-1, 1])."""
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    q_norm = np.linalg.norm(query) + 1e-12
    m_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (m_norms * q_norm)


def euclidean_vectorized(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean distance. Lower is more similar."""
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    diff = matrix - query
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def inner_product_vectorized(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Raw dot product. Higher is more similar."""
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix @ query
