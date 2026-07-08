"""
Aggregates every resource's router into one `api_router` that
`atlasdb/api/app.py` mounts with a single `app.include_router(api_router)`.
Adding a new resource (e.g. a future `/collections/{name}` delete route)
means adding a new file here and one line in this list -- nothing in
app.py changes.
"""
from fastapi import APIRouter

from atlasdb.api.routes import collections, search, stats, vectors

api_router = APIRouter()
api_router.include_router(collections.router)
api_router.include_router(vectors.router)
api_router.include_router(search.router)
api_router.include_router(stats.router)

__all__ = ["api_router"]
