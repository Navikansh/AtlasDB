"""
Collection Routes
--------------------
    POST /collections   - create a collection
    GET  /collections    - list existing collections

Each handler parses the request, calls one
`AtlasService` method, and serializes the response.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from atlasdb.api.routes.deps import get_service
from atlasdb.api.schemas import CreateCollectionRequest, CreateCollectionResponse

router = APIRouter(tags=["collections"])


@router.post("/collections", response_model=CreateCollectionResponse)
def create_collection(body: CreateCollectionRequest, request: Request):
    svc = get_service(request)
    existed = body.name in svc.list_collections()
    svc.create_collection(body.name, dim=body.dim)
    return CreateCollectionResponse(name=body.name, dim=body.dim, created=not existed)


@router.get("/collections")
def list_collections(request: Request):
    return {"collections": get_service(request).list_collections()}
