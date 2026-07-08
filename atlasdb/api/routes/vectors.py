"""
Vector Routes
---------------
    POST   /insert          - insert a vector (by raw vector or by text, embedded via the service)
    DELETE /vector/{id}      - delete a vector by id
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from atlasdb.api.routes.deps import get_service, resolve_vector
from atlasdb.api.schemas import DeleteResponse, InsertRequest, InsertResponse

router = APIRouter(tags=["vectors"])


@router.post("/insert", response_model=InsertResponse)
def insert(collection: str, body: InsertRequest, request: Request):
    svc = get_service(request)
    try:
        coll = svc.get_collection(collection)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    vector = resolve_vector(svc, body.text, body.vector)
    coll.insert(body.id, vector, body.metadata)
    return InsertResponse(id=body.id, inserted=True)


@router.delete("/vector/{record_id}", response_model=DeleteResponse)
def delete_vector(record_id: str, collection: str, request: Request):
    svc = get_service(request)
    try:
        coll = svc.get_collection(collection)
        coll.delete(record_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return DeleteResponse(id=record_id, deleted=True)
