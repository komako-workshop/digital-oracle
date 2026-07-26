"""Rotate outbound requests across a pool of proxies, with per-proxy cool-down.

Sources that throttle by client address (Eastmoney's quote hosts are the case
this was written for) can be spread across several egress addresses. This is the
mechanism only — it ships with an empty pool and stays a no-op until proxies are
configured, because routing user queries through unvetted public proxy lists
trades one failure mode for a worse one.

Configure with ``EASTMONEY_PROXIES`` / ``DIGITAL_ORACLE_PROXIES``, comma
separated, e.g. ``http://user:pass@host:8080,http://host2:8080``. Direct
connection is always kept as the last resort, so a fully-burned pool degrades to
today's behaviour rather than failing outright.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

#: A proxy that just failed is skipped for this long before being tried again.
DEFAULT_COOLDOWN_SECONDS = 120.0

#: Sentinel for "no proxy" — a direct connection.
DIRECT = ""


def proxies_from_env(*names: str) -> tuple[str, ...]:
    for name in names:
        raw = os.environ.get(name, "")
        if raw.strip():
            return tuple(p.strip() for p in raw.split(",") if p.strip())
    return ()


@dataclass
class ProxyPool:
    """Round-robin over proxies, skipping ones that recently failed."""

    proxies: Sequence[str] = ()
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    include_direct: bool = True
    _cursor: int = field(default=0, repr=False)
    _blocked_until: dict[str, float] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls, *names: str) -> "ProxyPool":
        return cls(proxies=proxies_from_env(*names))

    @property
    def configured(self) -> bool:
        return bool(self.proxies)

    def candidates(self) -> list[str]:
        """Endpoints to try in order: healthy proxies first, direct last.

        When every proxy is cooling down we still return them rather than only
        direct — a stale cool-down is a worse outcome than one wasted attempt.
        """
        now = time.monotonic()
        with self._lock:
            if not self.proxies:
                return [DIRECT] if self.include_direct else []
            size = len(self.proxies)
            ordered = [self.proxies[(self._cursor + i) % size] for i in range(size)]
            self._cursor = (self._cursor + 1) % size
            blocked = dict(self._blocked_until)

        healthy = [p for p in ordered if blocked.get(p, 0.0) <= now]
        cooling = [p for p in ordered if blocked.get(p, 0.0) > now]
        result = healthy + cooling
        if self.include_direct:
            result.append(DIRECT)
        return result

    def mark_failure(self, proxy: str) -> None:
        if not proxy:
            return
        with self._lock:
            self._blocked_until[proxy] = time.monotonic() + self.cooldown_seconds

    def mark_success(self, proxy: str) -> None:
        if not proxy:
            return
        with self._lock:
            self._blocked_until.pop(proxy, None)

    def stats(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            cooling = sum(1 for t in self._blocked_until.values() if t > now)
            return {
                "configured": len(self.proxies),
                "cooling_down": cooling,
                "include_direct": self.include_direct,
            }
