"""
Metric Registry
-----------------
The single place that maps a metric name ("cosine", "euclidean",
"inner_product") to its vectorized implementation and to whether higher
or lower scores are better. Every index (`brute_force`, `ivf`, `hnsw`)
calls `score()` / `top_k()` from here rather than importing
`vectorized.py` directly -- that indirection is what makes adding a fourth
metric a one-line registry edit instead of a change in three index files.
"""
from __future__ import annotations

import numpy as np

from atlasdb.distance.vectorized import (
    cosine_vectorized,
    euclidean_vectorized,
    inner_product_vectorized,
)

# metric name -> (vectorized function, higher_score_is_better)
_METRICS = {
    "cosine": (cosine_vectorized, True),
    "euclidean": (euclidean_vectorized, False),
    "inner_product": (inner_product_vectorized, True),
}


def available_metrics() -> list[str]:
    return list(_METRICS)


def score(metric: str, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if metric not in _METRICS:
        raise ValueError(f"unknown distance metric '{metric}'. choices: {available_metrics()}")
    fn, _ = _METRICS[metric]
    return fn(query, matrix)


def higher_is_better(metric: str) -> bool:
    if metric not in _METRICS:
        raise ValueError(f"unknown distance metric '{metric}'. choices: {available_metrics()}")
    return _METRICS[metric][1]


def top_k(metric: str, query: np.ndarray, matrix: np.ndarray, ids: list[str], k: int) -> list[tuple[str, float]]:
    """Return the top-k (id, score) pairs, ordered best-first for the given metric."""
    if matrix.shape[0] == 0:
        return []
    scores = score(metric, query, matrix)
    k = min(k, len(ids))
    better = higher_is_better(metric)
    if better:
        idx = np.argpartition(-scores, k - 1)[:k]
    else:
        idx = np.argpartition(scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx] if better else scores[idx])]
    return [(ids[i], float(scores[i])) for i in idx]
