"""
Public API: `from atlasdb import distance` then `distance.score(...)`,
`distance.top_k(...)`, `distance.cosine_naive(...)`, etc. -- callers never
need to know these live across three files internally.
"""
from atlasdb.distance.naive import cosine_naive, euclidean_naive, inner_product_naive
from atlasdb.distance.registry import available_metrics, higher_is_better, score, top_k
from atlasdb.distance.vectorized import (
    cosine_vectorized,
    euclidean_vectorized,
    inner_product_vectorized,
)

__all__ = [
    "cosine_naive", "euclidean_naive", "inner_product_naive",
    "cosine_vectorized", "euclidean_vectorized", "inner_product_vectorized",
    "score", "top_k", "higher_is_better", "available_metrics",
]
