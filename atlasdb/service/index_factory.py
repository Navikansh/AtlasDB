"""
Index Factory
---------------
Translates a config string (`index_type: hnsw`) plus the relevant tuning knobs into a constructed `VectorIndex`. 
Pulled out of the service layer so "which index types exist and how are they configured" is answerable by
reading one small file.
"""
from __future__ import annotations

from atlasdb.config.loader import AtlasConfig
from atlasdb.indexes.base import VectorIndex
from atlasdb.indexes.brute_force import BruteForceIndex
from atlasdb.indexes.hnsw import HNSWIndex
from atlasdb.indexes.ivf import IVFIndex

_INDEX_CLASSES: dict[str, type[VectorIndex]] = {
    "brute_force": BruteForceIndex,
    "ivf": IVFIndex,
    "hnsw": HNSWIndex,
}


def available_index_types() -> list[str]:
    return list(_INDEX_CLASSES)


def make_index(config: AtlasConfig, index_type: str, dim: int) -> VectorIndex:
    if index_type not in _INDEX_CLASSES:
        raise ValueError(f"unknown index type '{index_type}'. choices: {available_index_types()}")

    if index_type == "hnsw":
        return HNSWIndex(dim, config.distance_metric, M=config.hnsw_m,
                          ef_construction=config.hnsw_ef_construction,
                          ef_search=config.hnsw_ef_search)
    if index_type == "ivf":
        return IVFIndex(dim, config.distance_metric, n_clusters=config.ivf_n_clusters,
                         nprobe=config.ivf_nprobe)
    return BruteForceIndex(dim, config.distance_metric)
