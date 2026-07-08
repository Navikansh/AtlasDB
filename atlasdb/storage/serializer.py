"""
Binary Serializer
------------------
Hand-rolled encode/decode for vector records 

Record wire format (little-endian):
    [ id_len: u16 ][ id: utf-8 bytes ]
    [ dim: u32 ]
    [ vector: dim * f32 ]
    [ meta_len: u32 ][ meta: utf-8 JSON bytes ]
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_HEADER = struct.Struct("<H")       # id_len
_DIM = struct.Struct("<I")          # dim
_META_LEN = struct.Struct("<I")     # meta_len


@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def encode(record: VectorRecord) -> bytes:
    id_bytes = record.id.encode("utf-8")
    vec = np.asarray(record.vector, dtype="<f4")
    meta_bytes = json.dumps(record.metadata, separators=(",", ":")).encode("utf-8")

    parts = [
        _HEADER.pack(len(id_bytes)),
        id_bytes,
        _DIM.pack(vec.shape[0]),
        vec.tobytes(),
        _META_LEN.pack(len(meta_bytes)),
        meta_bytes,
    ]
    return b"".join(parts)


def decode(buf: bytes, offset: int = 0) -> tuple[VectorRecord, int]:
    """Decode one record starting at `offset`. Returns (record, next_offset)."""
    (id_len,) = _HEADER.unpack_from(buf, offset)
    offset += _HEADER.size
    record_id = buf[offset:offset + id_len].decode("utf-8")
    offset += id_len

    (dim,) = _DIM.unpack_from(buf, offset)
    offset += _DIM.size
    vector = np.frombuffer(buf, dtype="<f4", count=dim, offset=offset).copy()
    offset += dim * 4

    (meta_len,) = _META_LEN.unpack_from(buf, offset)
    offset += _META_LEN.size
    metadata = json.loads(buf[offset:offset + meta_len].decode("utf-8"))
    offset += meta_len

    return VectorRecord(id=record_id, vector=vector, metadata=metadata), offset


def encoded_size(record: VectorRecord) -> int:
    return len(encode(record))
