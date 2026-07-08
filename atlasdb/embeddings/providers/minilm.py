"""
MiniLM Provider
-----------------
Fast, 384-dim sentence embeddings via sentence-transformers'
`all-MiniLM-L6-v2`. Good default: small model, quick to embed, solid
general-purpose retrieval quality. Requires network access on first use
to download model weights (see `atlasdb/embeddings/providers/fallback.py`
for what happens when that's not available).
"""
from __future__ import annotations

import numpy as np

from atlasdb.embeddings.base import EmbeddingProvider


class MiniLMProvider(EmbeddingProvider):
    name = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # local import: heavy, optional dep
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts), dtype=np.float32)
