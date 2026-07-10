import unittest

import numpy as np

from atlasdb.cache.lru import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_put_get(self):
        cache = LRUCache(capacity=10)
        cache.put("a", np.array([1.0, 2.0]))
        got = cache.get("a")
        np.testing.assert_allclose(got, [1.0, 2.0])

    def test_miss_returns_none(self):
        cache = LRUCache(capacity=10)
        self.assertIsNone(cache.get("missing"))

    def test_eviction_at_capacity(self):
        cache = LRUCache(capacity=2)
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        cache.put("c", np.array([3.0]))  # evicts "a" (least recently used)
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))
        self.assertEqual(cache.evictions, 1)

    def test_get_refreshes_recency(self):
        cache = LRUCache(capacity=2)
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        cache.get("a")  # "a" is now most-recently-used
        cache.put("c", np.array([3.0]))  # should evict "b", not "a"
        self.assertIsNotNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))

    def test_hit_rate_tracking(self):
        cache = LRUCache(capacity=10)
        cache.put("a", np.array([1.0]))
        cache.get("a")  # hit
        cache.get("missing")  # miss
        self.assertAlmostEqual(cache.hit_rate, 0.5)

    def test_invalid_capacity_raises(self):
        with self.assertRaises(ValueError):
            LRUCache(capacity=0)


if __name__ == "__main__":
    unittest.main()
