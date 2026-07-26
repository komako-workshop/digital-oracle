"""Small TTL cache for provider responses.

Upstream throttling is usually a duplication problem before it is a volume
problem: every China A-share question asks for the same market-wide sector
ranking, and popular tickers repeat across users within minutes. Collapsing
those into one upstream call is cheaper and more robust than spreading the same
requests across more source addresses.

In-process and per-worker by design — no external store. With N uvicorn workers
you get at most N upstream calls per TTL window instead of one per request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable


@dataclass
class TTLCache:
    """Thread-safe cache with a per-entry time-to-live, in seconds."""

    max_entries: int = 512
    _entries: dict[Hashable, tuple[float, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    hits: int = 0
    misses: int = 0

    def get_or_call(self, key: Hashable, ttl_seconds: float, factory: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, or call ``factory`` and store it.

        ``factory`` runs outside the lock: a slow upstream call must not block
        readers of unrelated keys. Two callers racing the same cold key will both
        fetch, which costs one extra request and avoids holding a global lock
        across the network.
        """
        if ttl_seconds <= 0:
            return factory()

        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] > now:
                self.hits += 1
                return entry[1]
            self.misses += 1

        value = factory()

        with self._lock:
            if len(self._entries) >= self.max_entries:
                self._evict_expired(time.monotonic())
            self._entries[key] = (time.monotonic() + ttl_seconds, value)
        return value

    def _evict_expired(self, now: float) -> None:
        """Caller holds the lock."""
        expired = [k for k, (deadline, _) in self._entries.items() if deadline <= now]
        for key in expired:
            del self._entries[key]
        if len(self._entries) >= self.max_entries:
            # Still full of live entries — drop the soonest to expire.
            oldest = min(self._entries, key=lambda k: self._entries[k][0])
            del self._entries[oldest]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}
