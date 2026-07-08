"""
Naive Distance Metrics
------------------------
Pure-Python, single-pair implementations. Nobody should call these in the
hot path -- they exist so `benchmarks/run_benchmarks.py` can measure the
"vectorized NumPy vs. naive Python loop" speedup, which is a real number
worth quoting rather than an assumed one.
"""
from __future__ import annotations

import math


def cosine_naive(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_naive(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def inner_product_naive(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
