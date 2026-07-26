from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digital_oracle.cache import TTLCache
from digital_oracle.providers.eastmoney import EastmoneyProvider, EastmoneySectorFlowQuery
from digital_oracle.proxy_pool import DIRECT, ProxyPool, proxies_from_env

SAMPLE_SECTOR = {
    "data": {
        "diff": [
            {"f3": -0.01, "f12": "BK1036", "f14": "半导体", "f62": 3305525504.0, "f184": 0.95}
        ]
    }
}


class TTLCacheTests(unittest.TestCase):
    def test_second_call_within_ttl_does_not_hit_factory(self) -> None:
        cache = TTLCache()
        calls = []
        value = cache.get_or_call("k", 60, lambda: calls.append(1) or "v")
        again = cache.get_or_call("k", 60, lambda: calls.append(1) or "v")
        self.assertEqual(value, again)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_zero_ttl_bypasses_the_cache(self) -> None:
        cache = TTLCache()
        calls = []
        for _ in range(3):
            cache.get_or_call("k", 0, lambda: calls.append(1) or "v")
        self.assertEqual(len(calls), 3)

    def test_expired_entry_is_refetched(self) -> None:
        cache = TTLCache()
        calls = []
        cache.get_or_call("k", 0.01, lambda: calls.append(1) or "v")
        import time

        time.sleep(0.05)
        cache.get_or_call("k", 0.01, lambda: calls.append(1) or "v")
        self.assertEqual(len(calls), 2)

    def test_distinct_keys_are_isolated(self) -> None:
        cache = TTLCache()
        self.assertEqual(cache.get_or_call("a", 60, lambda: 1), 1)
        self.assertEqual(cache.get_or_call("b", 60, lambda: 2), 2)

    def test_does_not_grow_past_max_entries(self) -> None:
        cache = TTLCache(max_entries=8)
        for i in range(50):
            cache.get_or_call(i, 60, lambda i=i: i)
        self.assertLessEqual(cache.stats()["entries"], 8)

    def test_concurrent_readers_do_not_deadlock(self) -> None:
        cache = TTLCache()
        errors: list[BaseException] = []

        def worker(n: int) -> None:
            try:
                for _ in range(50):
                    cache.get_or_call(n % 5, 60, lambda n=n: n)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertFalse(any(t.is_alive() for t in threads))


class CountingClient:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.calls = 0

    def get_json(self, url: str, *, params: Mapping[str, object] | None = None) -> Any:
        self.calls += 1
        return self.data


class ProviderCacheTests(unittest.TestCase):
    def test_repeated_sector_query_collapses_to_one_upstream_call(self) -> None:
        """Every A-share question pulls this same market-wide table."""
        client = CountingClient(SAMPLE_SECTOR)
        provider = EastmoneyProvider(http_client=client)
        for _ in range(10):
            provider.list_sector_fund_flow(EastmoneySectorFlowQuery(limit=1))
        self.assertEqual(client.calls, 1)

    def test_different_arguments_are_cached_separately(self) -> None:
        client = CountingClient(SAMPLE_SECTOR)
        provider = EastmoneyProvider(http_client=client)
        provider.list_sector_fund_flow(EastmoneySectorFlowQuery(limit=1))
        provider.list_sector_fund_flow(EastmoneySectorFlowQuery(kind="concept", limit=1))
        self.assertEqual(client.calls, 2)


class ProxyPoolTests(unittest.TestCase):
    def test_empty_pool_is_direct_only(self) -> None:
        pool = ProxyPool()
        self.assertFalse(pool.configured)
        self.assertEqual(pool.candidates(), [DIRECT])

    def test_direct_is_always_the_last_resort(self) -> None:
        pool = ProxyPool(proxies=("http://a:1", "http://b:2"))
        self.assertEqual(pool.candidates()[-1], DIRECT)

    def test_rotates_starting_point_across_calls(self) -> None:
        pool = ProxyPool(proxies=("http://a:1", "http://b:2", "http://c:3"))
        firsts = [pool.candidates()[0] for _ in range(3)]
        self.assertEqual(sorted(firsts), ["http://a:1", "http://b:2", "http://c:3"])

    def test_failed_proxy_is_demoted_below_healthy_ones(self) -> None:
        pool = ProxyPool(proxies=("http://a:1", "http://b:2"))
        pool.mark_failure("http://a:1")
        order = [p for p in pool.candidates() if p != DIRECT]
        self.assertEqual(order[-1], "http://a:1")

    def test_success_clears_the_cooldown(self) -> None:
        pool = ProxyPool(proxies=("http://a:1", "http://b:2"))
        pool.mark_failure("http://a:1")
        pool.mark_success("http://a:1")
        self.assertEqual(pool.stats()["cooling_down"], 0)

    def test_all_cooling_still_offers_them_rather_than_only_direct(self) -> None:
        pool = ProxyPool(proxies=("http://a:1", "http://b:2"))
        pool.mark_failure("http://a:1")
        pool.mark_failure("http://b:2")
        self.assertEqual(len(pool.candidates()), 3)

    def test_reads_comma_separated_env(self) -> None:
        import os

        os.environ["_DO_TEST_PROXIES"] = " http://a:1 , http://b:2 ,"
        try:
            self.assertEqual(
                proxies_from_env("_DO_TEST_PROXIES"), ("http://a:1", "http://b:2")
            )
        finally:
            del os.environ["_DO_TEST_PROXIES"]

    def test_missing_env_yields_empty_pool(self) -> None:
        self.assertEqual(proxies_from_env("_DO_TEST_ABSENT"), ())


if __name__ == "__main__":
    unittest.main()
