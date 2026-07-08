"""
Config Loader
---------------
Loads configs/config.yaml once at startup into a plain dataclass. Every
other module (service layer, planner, cache, embeddings) reads its settings
from this object instead of hardcoding values -- so "switch from HNSW to
IVF" or "bump the cache size" is a one-line YAML edit, not a code change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class AtlasConfig:
    index_type: str = "hnsw"                 # brute_force | ivf | hnsw
    distance_metric: str = "cosine"           # cosine | euclidean | inner_product
    embedding_model: str = "minilm"           # minilm | bge-small
    cache_size: int = 10_000
    data_dir: str = "./data"

    # planner knobs
    brute_force_threshold: int = 5000
    selective_filter_threshold: float = 0.1

    # HNSW knobs
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 50

    # IVF knobs
    ivf_n_clusters: int = 32
    ivf_nprobe: int = 4

    # logging
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "AtlasConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        return cls(**filtered)
