"""
HNSW Greedy Search
--------------------
The traversal algorithm that walks an `HNSWGraph` (see graph.py). Two
entry points into this file, matching the two places HNSW actually
searches:

search_layer(...): given a set of entry points at one layer, greedily
  expand toward the query until `ef` candidates have been explored without
  improvement. Used both for descending through the sparse upper layers
  (ef=1, just "which single node is closest") and for the final, wider
  search at layer 0 (ef=ef_search, the real top-k candidate pool).

select_closest(...): given a candidate pool, keep the `m` closest to
  a reference vector. Used both at insertion time (deciding which
  neighbors a new node connects to) and when trimming a node's neighbor
  list back down to its cap after a new edge pushed it over.

All distances here follow a "lower is better" convention regardless of
the underlying metric's native direction -- `index.py` is responsible for flipping cosine/inner-product's "higher is better" 
scores into distances before calling into this file, so the traversal logic itself never has tobranch on metric type.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DistanceFn = Callable[[np.ndarray, np.ndarray], float]


def search_layer(vectors: dict[str, np.ndarray], neighbors: dict[str, set[str]],
                  query: np.ndarray, entry_points: set[str],
                  ef: int, dist: DistanceFn) -> list[tuple[float, str]]:
    """Greedy best-first search within a single layer.

    Returns up to `ef` (distance, id) pairs, sorted closest-first.
    `neighbors` is the adjacency dict for this one layer (i.e. an entry of
    `HNSWGraph.neighbors`), not the whole graph.
    """
    visited = set(entry_points)
    candidates = [(dist(query, vectors[e]), e) for e in entry_points]
    candidates.sort(key=lambda x: x[0])
    results = list(candidates)

    while candidates:
        dist_c, c = candidates.pop(0)
        if results and dist_c > results[-1][0] and len(results) >= ef:
            break
        for neighbor in neighbors.get(c, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            d = dist(query, vectors[neighbor])
            if len(results) < ef or d < results[-1][0]:
                candidates.append((d, neighbor))
                results.append((d, neighbor))
                results.sort(key=lambda x: x[0])
                results = results[:ef]
                candidates.sort(key=lambda x: x[0])
    return results


def select_closest(vectors: dict[str, np.ndarray], reference: np.ndarray,
                    candidate_ids: set[str] | list[str], m: int,
                    dist: DistanceFn) -> list[str]:
    """Keep the `m` candidates closest to `reference`. Uses the simplest
    "M nearest" rule rather than HNSW's optional heuristic diversification
    (which also tries to keep neighbors spread across directions, not just
    close) - a reasonable, documented simplification for this project's scope."""
    scored = sorted(candidate_ids, key=lambda c: dist(reference, vectors[c]))
    return scored[:m]
