"""
SIFT1M Dataset Loader
------------------------
Reads the standard `.fvecs` (float vectors) and `.ivecs` (int vectors,
used for ground-truth neighbor lists) formats used by the SIFT1M / texmex
ANN benchmark corpus. Format for both is identical framing, different
dtype: each vector is stored as [int32 dim][dim x value], repeated.

Falls back to a synthetic Gaussian dataset if the real files aren't present
under datasets/sift/ -- see datasets/README.md for why, and for how to get
the real dataset when you have network access.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("atlasdb.benchmarks")

SIFT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "sift"


def _read_fvecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    data = data.view(np.float32).reshape(-1, dim + 1)
    return data[:, 1:].copy()


def _read_ivecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].copy()


def load_sift1m(subset_size: int = 100_000, n_queries: int = 200):
    """Returns (base_vectors, query_vectors, ground_truth) or, if the real
    dataset isn't on disk, a synthetic Gaussian stand-in of the same shape
    (ground_truth computed via brute-force on the synthetic set)."""
    base_path = SIFT_DIR / "sift_base.fvecs"
    query_path = SIFT_DIR / "sift_query.fvecs"
    gt_path = SIFT_DIR / "sift_groundtruth.ivecs"

    if base_path.exists() and query_path.exists() and gt_path.exists():
        logger.info("loading real SIFT1M data from %s", SIFT_DIR)
        base = _read_fvecs(base_path)[:subset_size]
        queries = _read_fvecs(query_path)[:n_queries]
        gt = compute_ground_truth(base, queries)
        return base, queries, gt, "sift1m"

    logger.warning(
        "SIFT1M files not found under %s -- falling back to synthetic Gaussian "
        "data. Recall numbers from this fallback are NOT comparable to published "
        "ANN benchmarks. See datasets/README.md to download the real dataset.",
        SIFT_DIR,
    )
    rng = np.random.default_rng(42)
    dim = 128  # matches SIFT1M's native dimensionality
    base = rng.normal(size=(subset_size, dim)).astype(np.float32)
    queries = rng.normal(size=(n_queries, dim)).astype(np.float32)

    # brute-force ground truth on the synthetic set (Euclidean, matching SIFT's native metric)
    gt = np.zeros((n_queries, 100), dtype=np.int64)
    for i, q in enumerate(queries):
        d = np.linalg.norm(base - q, axis=1)
        gt[i] = np.argsort(d)[:100]
    return base, queries, gt, "synthetic"

def compute_ground_truth(base, queries, k=100):
    gt = np.empty((len(queries), k), dtype=np.int64)

    for i, q in enumerate(queries):
        d = np.linalg.norm(base - q, axis=1)
        gt[i] = np.argsort(d)[:k]

    return gt
