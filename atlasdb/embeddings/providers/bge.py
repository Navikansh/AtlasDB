"""
BGE-Small Provider
--------------------
Stronger retrieval quality than MiniLM at the same 384 dimensions, via
sentence-transformers' `BAAI/bge-small-en-v1.5`. Swapping between this and
MiniLM is a one-line `config.yaml` change (`embedding_model: bge-small`)
-- the rest of the system only ever sees "a 384-dim vector," never a
model-specific detail. That decoupling is the actual point of the
`EmbeddingProvider` interface; which two models are wired in is secondary.
"""
from __future__ import annotations

import numpy as np

from atlasdb.embeddings.base import EmbeddingProvider


class BGESmallProvider(EmbeddingProvider):
    name = "bge-small-en-v1.5"
    dim = 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # local import: heavy, optional dep
        self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts), dtype=np.float32)
