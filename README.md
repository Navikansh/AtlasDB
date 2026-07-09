# AtlasDB

A vector database built from scratch in Python — a layered storage engine
(no `pickle`), three ANN index implementations behind a common interface
(brute force, IVF, HNSW), a cost-aware query planner, a FastAPI REST layer,
an LRU embedding cache, and a benchmark suite with real recall/latency
numbers.

Every module in this repo does exactly one thing, and every file is small
enough to explain in a couple of sentences.

## Table of contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [API usage](#api-usage)
- [Design decisions worth explaining](#design-decisions-worth-explaining)
- [What's real vs. a documented stand-in](#whats-real-vs-a-documented-stand-in)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Explicitly out of scope](#explicitly-out-of-scope)

## Architecture

```
API (FastAPI, one router per resource, zero business logic)
  │
Service Layer (AtlasService: collections + shared cache/embeddings)
  │            (CollectionService: one per collection)
  │
Query Planner  ──decides──▶  brute force vs ANN · filter-first vs search-first
  │
Metadata Filter
  │
VectorIndex interface  ──dispatches to──▶  BruteForceIndex | IVFIndex | HNSWIndex
  │
Storage Layer: PageManager → RecordManager → CollectionManager
```

Every arrow above is a real interface, not a comment: `VectorIndex(ABC)`
(`atlasdb/indexes/base.py`) is what lets the planner swap index strategies
without touching search code, and `EmbeddingProvider(ABC)`
(`atlasdb/embeddings/base.py`) does the same for embedding models.

## Repository layout

```
atlasdb/
├── api/
│   ├── app.py               FastAPI app: config, logging, one AtlasService instance
│   ├── schemas.py             pydantic request/response models
│   └── routes/                one file per resource, zero shared state
│       ├── collections.py       POST/GET /collections
│       ├── vectors.py             POST /insert, DELETE /vector/{id}
│       ├── search.py               POST /search
│       ├── stats.py                 GET /stats
│       └── deps.py                   shared request helpers
│
├── service/
│   ├── atlas_service.py       top-level: collections + shared embedding/cache
│   ├── collection_service.py   one per collection: planner + storage + indexes
│   ├── index_factory.py         config -> concrete VectorIndex instance
│   └── filtering.py               metadata predicate matching
│
├── planner/
│   └── planner.py              QueryPlanner: index-strategy + filter-order heuristics
│
├── indexes/
│   ├── base.py                  VectorIndex(ABC) — the shared contract
│   ├── brute_force.py            O(n) exact search, the recall/latency baseline
│   ├── ivf/
│   │   ├── kmeans.py               centroid training (Lloyd's algorithm)
│   │   └── index.py                 inverted lists + probing, built on kmeans.py
│   └── hnsw/
│       ├── graph.py                 the layered graph data structure
│       ├── search.py                  greedy layer search + neighbor selection
│       └── index.py                    wires graph.py + search.py into VectorIndex
│
├── embeddings/
│   ├── base.py                  EmbeddingProvider(ABC)
│   └── providers/
│       ├── minilm.py              sentence-transformers: all-MiniLM-L6-v2
│       ├── fallback.py               offline, deterministic, non-semantic stand-in
│       └── registry.py                 config string -> provider, with fallback
│
├── storage/
│   ├── page_manager.py           fixed-size page read/write on a raw file
│   ├── record_manager.py          logical record id -> byte-offset directory
│   ├── serializer.py                binary encode/decode, not pickle
│   └── collection_manager.py         top-level CRUD, ties the above together
│
├── distance/
│   ├── naive.py                  pure-Python single-pair implementations
│   ├── vectorized.py               batched NumPy implementations
│   └── registry.py                   metric name -> function, + the shared top_k()
│
├── cache/lru.py                 hand-rolled LRU cache (OrderedDict-based, O(1))
├── metrics/collector.py          latency percentiles + QPS for /stats
├── config/loader.py               loads configs/config.yaml into a typed dataclass
└── logging_setup.py                 one place that configures logging; no print() anywhere

benchmarks/          run_benchmarks.py + SIFT1M loader (synthetic fallback if no network)
tests/                mirrors the atlasdb/ tree file-for-file — 95 tests, all passing
docker/               Dockerfile + docker-compose.yml
.github/workflows/   CI: lint (ruff) + test (pytest) on every push
configs/config.yaml   index type, distance metric, embedding model, cache size, all knobs
```

The `tests/` tree mirrors `atlasdb/` on purpose.

## Quickstart

```bash
git clone <this-repo>
cd atlasdb
pip install -r requirements.txt

# run the API
uvicorn atlasdb.api.app:app --reload
# -> interactive docs at http://localhost:8000/docs

# or via Docker
docker compose -f docker/docker-compose.yml up --build

# run the tests (95 tests)
pytest tests/ -v
# or, no pytest required:
python3 -m unittest discover -s tests -p "test_*.py"

# run the benchmark suite
# see datasets/README.md for more info
python3 benchmarks/run_benchmarks.py --subset-size 20000 --queries 200
```

## API usage

```bash
curl -X POST localhost:8000/collections -H 'Content-Type: application/json' \
  -d '{"name": "docs", "dim": 384}'

curl -X POST "localhost:8000/insert?collection=docs" -H 'Content-Type: application/json' -d '{
  "id": "doc1", "text": "vector databases are cool", "metadata": {"category": "tech"}
}'

curl -X POST "localhost:8000/search?collection=docs" -H 'Content-Type: application/json' -d '{
  "text": "tell me about vector search", "k": 5, "filter": {"category": "tech"}
}'

curl "localhost:8000/stats?collection=docs"
```

## What's real vs. a documented stand-in

| Component | Status |
|---|---|
| Storage engine (page/record manager, binary serializer) | Fully implemented, no pickle, tested including reload-from-disk |
| `BruteForceIndex`, `IVFIndex`, `HNSWIndex` | Fully implemented from scratch, all conform to `VectorIndex(ABC)` |
| Query planner | Real heuristic logic (size threshold, filter selectivity), not cost-based estimation — a natural next step, not built here |
| FastAPI service | Fully implemented and layered correctly; run it yourself with `pip install -r requirements.txt` to confirm end to end |
| MiniLM  provider | Real sentence-transformers wrappers |
| `HashingFallbackProvider` | A deterministic, non-semantic stand-in used automatically when the real models can't load, so the rest of the system stays runnable offline. **Not a real embedding model.** |
| LRU cache, config system, logging, metrics | Fully implemented and tested |
| Benchmark suite | Fully implemented and runs end to end; defaults to a synthetic Gaussian dataset if SIFT1M isn't downloaded locally (see `datasets/README.md`) |

## Benchmarks

`benchmarks/run_benchmarks.py` compares all three indexes on build time,
p50/p99 latency, recall@k, throughput, and approximate memory, and writes
`results.csv` + `results.png`. See `datasets/README.md` for
how to get the real dataset.

## Testing

```bash
pytest tests/ -v
```

95 tests, structured 1:1 with the source tree: storage persistence
(including reload-from-disk), all three indexes' correctness and recall
against a brute-force ground truth, the HNSW graph and search algorithm in
isolation, k-means clustering in isolation, the planner's branching logic,
the LRU cache's eviction behavior, the embedding fallback provider, and the
service layer end to end.
