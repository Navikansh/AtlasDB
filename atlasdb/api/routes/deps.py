"""
Route Dependencies
--------------------
Tiny shared helpers every route file needs: getting the `AtlasService`
singleton off the request, and resolving a request body's `text`/`vector`
into an actual vector (embedding it via the shared cache+provider if only
text was given). Kept here instead of duplicated in every route file.
"""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException, Request

from atlasdb.service.atlas_service import AtlasService


def get_service(request: Request) -> AtlasService:
    return request.app.state.atlas_service


def resolve_vector(svc: AtlasService, text: str | None, vector: list[float] | None) -> np.ndarray:
    if vector is not None:
        return np.asarray(vector, dtype=np.float32)
    if text is not None:
        return svc.embed(text)
    raise HTTPException(status_code=400, detail="must supply either 'text' or 'vector'")
