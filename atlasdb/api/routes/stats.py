"""
Stats Route
-------------
    GET /stats - collection size, dim, index type, latency percentiles, QPS, and the shared embedding cache's hit rate.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from atlasdb.api.routes.deps import get_service
from atlasdb.api.schemas import StatsResponse

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def stats(collection: str, request: Request):
    svc = get_service(request)
    try:
        coll = svc.get_collection(collection)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    s = coll.stats()
    s["cache_hit_rate"] = svc.cache_stats()["hit_rate"]
    return s
