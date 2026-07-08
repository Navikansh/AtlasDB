"""
FastAPI App Entrypoint
-------------------------
Run with:  uvicorn atlasdb.api.app:app --reload

Startup does three things: configure logging, load config.yaml, and construct one AtlasService instance 
stashed on app.state. Every route handler pulls that same instance via `request.app.state.atlas_service`
there's exactly one place that constructs the service, which is what keeps collections/caches shared
across requests instead of re-created per call.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from atlasdb.api.routes import api_router
from atlasdb.config.loader import AtlasConfig
from atlasdb.logging_setup import configure_logging
from atlasdb.service.atlas_service import AtlasService


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AtlasConfig.load()
    configure_logging(config.log_level)
    app.state.atlas_service = AtlasService(config)
    yield


app = FastAPI(
    title="AtlasDB",
    description="A from-scratch vector database: custom storage engine, "
                 "ANN indexes (brute force / IVF / HNSW), a query planner, "
                 "and a REST API.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(api_router)


@app.get("/")
def root():
    return {"service": "AtlasDB", "docs": "/docs"}
