"""
K-Means Clustering
--------------------
Trains the centroids IVFIndex partitions vectors around. 

Assign each point to its nearest centroid, recompute centroids as the mean
of their assigned points, repeat until convergence or `iters` is reached.
"""
from __future__ import annotations

import numpy as np


def fit(vectors: np.ndarray, n_clusters: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Returns an (n_clusters, dim) array of centroids."""
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    n_clusters = min(n_clusters, n)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = vectors[init_idx].copy()

    for _ in range(iters):
        assignments = assign(vectors, centroids)
        new_centroids = centroids.copy()
        for c in range(n_clusters):
            members = vectors[assignments == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def assign(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Returns, for each row in `vectors`, the index of its nearest centroid."""
    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def nearest_centroid(vector: np.ndarray, centroids: np.ndarray) -> int:
    """Same as `assign` but for a single vector -- the hot path at search/insert time."""
    d = np.linalg.norm(centroids - vector, axis=1)
    return int(np.argmin(d))
