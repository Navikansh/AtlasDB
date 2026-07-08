"""
Provider Registry
--------------------
The one place that maps a config string ("minilm", "bge-small") to a
provider class, and the one function (`get_provider`) every caller uses
instead of importing a concrete provider class directly. That indirection
is what makes `embedding_model: bge-small` in config.yaml "just work"
without any code change, and what makes the network-unavailable fallback
automatic instead of something every call site has to handle.
"""
from __future__ import annotations

import logging

from atlasdb.embeddings.base import EmbeddingProvider
from atlasdb.embeddings.providers.bge import BGESmallProvider
from atlasdb.embeddings.providers.fallback import HashingFallbackProvider
from atlasdb.embeddings.providers.minilm import MiniLMProvider

logger = logging.getLogger("atlasdb.embeddings")

_PROVIDERS = {
    "minilm": MiniLMProvider,
    "bge-small": BGESmallProvider,
}


def available_providers() -> list[str]:
    return list(_PROVIDERS)


def get_provider(name: str) -> EmbeddingProvider:
    """Instantiate the configured provider, falling back to the offline
    hashing provider if the real model can't be loaded (no network, missing
    dependency, etc.) so the rest of the system stays runnable."""
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"unknown embedding provider '{name}'. choices: {available_providers()}")
    try:
        return cls()
    except Exception as exc:  # ImportError, network error, etc.
        logger.warning(
            "could not load embedding provider '%s' (%s: %s) -- "
            "falling back to HashingFallbackProvider. Not semantically meaningful; "
            "install sentence-transformers and ensure network access for real embeddings.",
            name, type(exc).__name__, exc,
        )
        return HashingFallbackProvider()
