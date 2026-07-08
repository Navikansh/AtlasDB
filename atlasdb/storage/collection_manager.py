"""
Collection Manager
-------------------
Top of the storage stack. A "collection" is AtlasDB's equivalent of a table:
a named set of vectors + metadata with one config (dimension, distance
metric). On disk each collection is a directory:

    collections/<name>/
        vectors.page      <- PageManager-backed flat file
        vectors.directory <- RecordManager's id -> offset index
        config.json

Everything above this layer (indexes, planner, service) talks to
CollectionManager, never to PageManager or RecordManager directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from atlasdb.storage.page_manager import PageManager
from atlasdb.storage.record_manager import RecordManager
from atlasdb.storage.serializer import VectorRecord, decode, encode

logger = logging.getLogger("atlasdb.storage")


class CollectionManager:
    def __init__(self, root: str | Path, name: str, dim: int | None = None,
                 distance_metric: str = "cosine"):
        self.root = Path(root)
        self.name = name
        self.dir = self.root / "collections" / name
        self.dir.mkdir(parents=True, exist_ok=True)

        self._config_path = self.dir / "config.json"
        self.dim, self.distance_metric = self._load_or_init_config(dim, distance_metric)

        self._pages = PageManager(self.dir / "vectors.page")
        self._records = RecordManager(self._pages, self.dir / "vectors.directory")

        logger.info("collection '%s' opened (dim=%s, metric=%s, %d records)",
                    name, self.dim, self.distance_metric, len(self._records))

    def _load_or_init_config(self, dim, distance_metric) -> tuple[int | None, str]:
        if self._config_path.exists():
            with open(self._config_path) as f:
                cfg = json.load(f)
            return cfg["dim"], cfg["distance_metric"]
        cfg = {"dim": dim, "distance_metric": distance_metric}
        with open(self._config_path, "w") as f:
            json.dump(cfg, f)
        return dim, distance_metric

    def _persist_config(self) -> None:
        with open(self._config_path, "w") as f:
            json.dump({"dim": self.dim, "distance_metric": self.distance_metric}, f)

    # -- CRUD ---------------------------------------------------------------

    def insert(self, record_id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        vector = np.asarray(vector, dtype="<f4")
        if self.dim is None:
            self.dim = vector.shape[0]
            self._persist_config()
        elif vector.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.shape[0]}")

        record = VectorRecord(id=record_id, vector=vector, metadata=metadata or {})
        self._records.append(record_id, encode(record))
        logger.debug("inserted record %s into collection %s", record_id, self.name)

    def insert_batch(self, records: list[tuple[str, np.ndarray, dict[str, Any]]]) -> None:
        for record_id, vector, metadata in records:
            self.insert(record_id, vector, metadata)

    def get(self, record_id: str) -> VectorRecord:
        raw = self._records.read(record_id)
        record, _ = decode(raw)
        return record

    def delete(self, record_id: str) -> None:
        self._records.delete(record_id)
        logger.debug("deleted record %s from collection %s", record_id, self.name)

    def update(self, record_id: str, vector: np.ndarray | None = None,
               metadata: dict[str, Any] | None = None) -> None:
        existing = self.get(record_id)
        new_vector = existing.vector if vector is None else np.asarray(vector, dtype="<f4")
        new_meta = existing.metadata if metadata is None else metadata
        self._records.delete(record_id)
        self.insert(record_id, new_vector, new_meta)

    def exists(self, record_id: str) -> bool:
        return self._records.exists(record_id)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[VectorRecord]:
        for record_id in self._records.ids():
            yield self.get(record_id)

    def all_ids(self) -> list[str]:
        return self._records.ids()
