"""
Benchmark Suite
------------------
Compares BruteForceIndex, IVFIndex, and HNSWIndex on:
  - build time
  - search latency (p50 / p99)
  - recall@10 against ground truth
  - throughput (QPS)
  - approximate memory footprint

Outputs a CSV (benchmarks/results.csv) and a comparison chart
(benchmarks/results.png). Run with:

    python3 benchmarks/run_benchmarks.py [--subset-size N] [--queries N]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlasdb.indexes.brute_force import BruteForceIndex
from atlasdb.indexes.hnsw import HNSWIndex
from atlasdb.indexes.ivf import IVFIndex
from atlasdb.logging_setup import configure_logging
from benchmarks.load_sift import load_sift1m

OUTPUT_DIR = Path(__file__).resolve().parent


def recall_at_k(retrieved_ids: list[int], ground_truth_ids: np.ndarray, k: int) -> float:
    gt_set = set(int(i) for i in ground_truth_ids[:k])
    retrieved_set = set(int(i) for i in retrieved_ids[:k])
    return len(gt_set & retrieved_set) / k


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = min(int(len(data) * p), len(data) - 1)
    return data[idx]


def approx_memory_bytes(index) -> int:
    """Rough estimate: sum of stored float32 vectors. Good enough for a
    relative comparison across index types, not a precise RSS measurement."""
    n = len(index)
    dim = index.dim
    base = n * dim * 4
    if isinstance(index, HNSWIndex):
        # neighbor lists: roughly M0 + (avg extra layers * M) ints per node
        avg_edges_per_node = index.M0 + index.M  # rough upper bound
        base += n * avg_edges_per_node * 8  # python set/int overhead, approximate
    return base


def benchmark_index(name: str, index_cls, build_kwargs: dict,
                     base_ids: list[str], base_vectors: np.ndarray,
                     queries: np.ndarray, ground_truth: np.ndarray, k: int) -> dict:
    dim = base_vectors.shape[1]
    index = index_cls(dim=dim, distance_metric="euclidean", **build_kwargs)

    build_start = time.perf_counter()
    index.build(base_ids, base_vectors)
    build_time_s = time.perf_counter() - build_start

    latencies_ms = []
    recalls = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        results = index.search(query, k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        retrieved_ids = [int(id_) for id_, _ in results]
        recalls.append(recall_at_k(retrieved_ids, ground_truth[i], k))

    total_search_time_s = sum(latencies_ms) / 1000
    qps = len(queries) / total_search_time_s if total_search_time_s > 0 else float("inf")

    return {
        "index": name,
        "n_vectors": len(base_ids),
        "build_time_s": round(build_time_s, 4),
        "p50_latency_ms": round(percentile(latencies_ms, 0.50), 4),
        "p99_latency_ms": round(percentile(latencies_ms, 0.99), 4),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "qps": round(qps, 2),
        "approx_memory_mb": round(approx_memory_bytes(index) / (1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=20_000,
                         help="number of base vectors to benchmark against (default 20k)")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    configure_logging("INFO")

    base, queries, ground_truth, source = load_sift1m(subset_size=args.subset_size, n_queries=args.queries)
    base_ids = [str(i) for i in range(len(base))]
    print(f"benchmarking against {source} data: {len(base)} base vectors, "
          f"{len(queries)} queries, dim={base.shape[1]}")

    configs = [
        ("brute_force", BruteForceIndex, {}),
        ("ivf", IVFIndex, {"n_clusters": max(8, int(len(base) ** 0.5)), "nprobe": 8}),
        ("hnsw", HNSWIndex, {"M": 16, "ef_construction": 100, "ef_search": 50}),
    ]

    rows = []
    for name, cls, kwargs in configs:
        print(f"running {name}...")
        row = benchmark_index(name, cls, kwargs, base_ids, base, queries, ground_truth, args.k)
        rows.append(row)
        print(f"  {row}")

    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    _plot(rows, OUTPUT_DIR / "results.png")
    print(f"wrote {OUTPUT_DIR / 'results.png'}")


def _plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["index"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(names, [r["p99_latency_ms"] for r in rows], color=["#888", "#4c72b0", "#55a868"])
    axes[0].set_title("p99 Search Latency (ms)")
    axes[0].set_ylabel("ms")

    axes[1].bar(names, [r["recall_at_k"] for r in rows], color=["#888", "#4c72b0", "#55a868"])
    axes[1].set_title("Recall@K")
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(names, [r["qps"] for r in rows], color=["#888", "#4c72b0", "#55a868"])
    axes[2].set_title("Throughput (QPS)")

    fig.suptitle("AtlasDB Index Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
