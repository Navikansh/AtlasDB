"""
Atlas Service
---------------
The top-level object the API layer holds a single instance of
(`app.state.atlas_service` in `atlasdb/api/app.py`). Manages the shared embedding provider and 
cache both are process-wide, not per-collection and a dict of `CollectionService` instances, 
one per collection name.
"""
from __future__ import annotations

import logging

import numpy as np

from atlasdb.cache.lru import LRUCache
from atlasdb.config.loader import AtlasConfig
from atlasdb.embeddings.base import EmbeddingProvider
from atlasdb.embeddings.providers import get_provider
from atlasdb.service.collection_service import CollectionService

logger = logging.getLogger("atlasdb.service")


class AtlasService:
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.embedding_provider: EmbeddingProvider = get_provider(config.embedding_model)
        self.embedding_cache = LRUCache(capacity=config.cache_size)
        self._collections: dict[str, CollectionService] = {}

    # -- collection management ---------------------------------------------------

    def create_collection(self, name: str, dim: int | None = None) -> CollectionService:
        if name in self._collections:
            return self._collections[name]
        svc = CollectionService(self.config.data_dir, name, self.config, dim=dim)
        self._collections[name] = svc
        logger.info("created collection '%s'", name)
        return svc

    def get_collection(self, name: str) -> CollectionService:
        if name not in self._collections:
            raise KeyError(f"collection '{name}' does not exist")
        return self._collections[name]

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())

    # -- shared embedding pipeline ------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        vector = self.embedding_provider.embed(text)
        self.embedding_cache.put(text, vector)
        return vector

    def cache_stats(self) -> dict:
        return self.embedding_cache.stats()
