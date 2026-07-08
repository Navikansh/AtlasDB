"""
API Schemas
-------------
Pydantic models for request/response bodies. Kept separate from routes.py
so the wire format is easy to scan without wading through route logic.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str
    dim: Optional[int] = None


class CreateCollectionResponse(BaseModel):
    name: str
    dim: Optional[int]
    created: bool


class InsertRequest(BaseModel):
    id: str
    text: Optional[str] = None            # embedded via the configured provider if vector is omitted
    vector: Optional[list[float]] = None  # or supply a raw vector directly
    metadata: dict[str, Any] = Field(default_factory=dict)


class InsertResponse(BaseModel):
    id: str
    inserted: bool


class SearchRequest(BaseModel):
    text: Optional[str] = None
    vector: Optional[list[float]] = None
    k: int = 10
    filter: Optional[dict[str, Any]] = None


class SearchResultItem(BaseModel):
    id: str
    score: float


class SearchPlan(BaseModel):
    index_strategy: str
    filter_order: str
    reason: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    plan: SearchPlan


class DeleteResponse(BaseModel):
    id: str
    deleted: bool


class StatsResponse(BaseModel):
    collection: str
    size: int
    dim: Optional[int]
    distance_metric: str
    index_type: str
    search_count: int
    insert_count: int
    p50_latency_ms: float
    p99_latency_ms: float
    qps: float
    cache_hit_rate: Optional[float] = None
