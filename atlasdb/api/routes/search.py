"""
Search Route
--------------
    POST /search - the whole query pipeline (embed if needed -> planner -> filter -> ANN/brute-force search -> ranked top-k),
entirely delegated to CollectionService.search().
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from atlasdb.api.routes.deps import get_service, resolve_vector
from atlasdb.api.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(collection: str, body: SearchRequest, request: Request):
    svc = get_service(request)
    try:
        coll = svc.get_collection(collection)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    vector = resolve_vector(svc, body.text, body.vector)
    return coll.search(vector, body.k, metadata_filter=body.filter)
