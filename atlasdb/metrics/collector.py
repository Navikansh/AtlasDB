"""
Metrics Collector
--------------------
Tracks the numbers the `/stats` endpoint reports: latency percentiles,
throughput, and whatever the cache/index expose about themselves.
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager


class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self._latencies_ms: deque[float] = deque(maxlen=window_size)
        self._search_count = 0
        self._insert_count = 0
        self._start_time = time.monotonic()

    @contextmanager
    def time_search(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._latencies_ms.append(elapsed_ms)
            self._search_count += 1

    def record_insert(self) -> None:
        self._insert_count += 1

    def _percentile(self, p: float) -> float:
        if not self._latencies_ms:
            return 0.0
        data = sorted(self._latencies_ms)
        idx = min(int(len(data) * p), len(data) - 1)
        return data[idx]

    def qps(self) -> float:
        elapsed = time.monotonic() - self._start_time
        return self._search_count / elapsed if elapsed > 0 else 0.0

    def snapshot(self) -> dict:
        return {
            "search_count": self._search_count,
            "insert_count": self._insert_count,
            "p50_latency_ms": round(self._percentile(0.50), 3),
            "p99_latency_ms": round(self._percentile(0.99), 3),
            "qps": round(self.qps(), 3),
        }
