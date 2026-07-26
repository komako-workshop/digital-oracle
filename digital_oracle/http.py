from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .proxy_pool import DIRECT, ProxyPool


class HttpClientError(RuntimeError):
    pass


class JsonHttpClient(Protocol):
    def get_json(self, url: str, *, params: Mapping[str, object] | None = None) -> Any: ...


class TextHttpClient(Protocol):
    def get_text(self, url: str, *, params: Mapping[str, object] | None = None) -> str: ...


def _serialize_query_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_url(url: str, params: Mapping[str, object] | None) -> str:
    if not params:
        return url
    query_items = [
        (key, _serialize_query_value(value))
        for key, value in params.items()
        if value is not None
    ]
    if not query_items:
        return url
    return f"{url}?{urlencode(query_items)}"


@dataclass
class UrllibJsonClient:
    timeout_seconds: float = 20.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    headers: Mapping[str, str] = field(
        default_factory=lambda: {
            "Accept": "application/json,text/csv,text/plain,application/xml",
            "User-Agent": "digital-oracle/0.1",
        }
    )
    #: Optional egress rotation for sources that throttle by client address.
    #: Unset means every request goes out directly, exactly as before.
    proxy_pool: ProxyPool | None = None

    def get_json(self, url: str, *, params: Mapping[str, object] | None = None) -> Any:
        request_url = self._build_request(url, params)
        try:
            with self._open(request_url) as response:
                return json.load(response)
        except HTTPError as exc:
            raise HttpClientError(f"request failed: {request_url.full_url}") from exc
        except json.JSONDecodeError as exc:
            raise HttpClientError(f"invalid json payload: {request_url.full_url}") from exc

    def get_text(self, url: str, *, params: Mapping[str, object] | None = None) -> str:
        request_url = self._build_request(url, params)
        try:
            with self._open(request_url) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            raise HttpClientError(f"request failed: {request_url.full_url}") from exc

    def _build_request(self, url: str, params: Mapping[str, object] | None) -> Request:
        request_url = _build_url(url, params)
        return Request(request_url, headers=dict(self.headers))

    def _open(self, request: Request):
        if self.proxy_pool is None or not self.proxy_pool.configured:
            return self._open_via(request, DIRECT)

        # urllib mutates a Request in place while opening it — ProxyHandler
        # rewrites .host and sets tunnel state — so a Request that has been
        # through a proxy cannot be reused for the next attempt.
        url = request.full_url
        last_error: Exception | None = None
        for endpoint in self.proxy_pool.candidates():
            try:
                response = self._open_via(Request(url, headers=dict(self.headers)), endpoint)
            except HTTPError as exc:
                # A throttle answers with a status code, so it has to burn the
                # proxy too — otherwise rotation never kicks in for the exact
                # case it exists for.
                self.proxy_pool.mark_failure(endpoint)
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 — try the next endpoint
                self.proxy_pool.mark_failure(endpoint)
                last_error = exc
                continue
            self.proxy_pool.mark_success(endpoint)
            return response
        if isinstance(last_error, HTTPError):
            raise last_error
        raise HttpClientError(f"request failed: {request.full_url}") from last_error

    def _open_via(self, request: Request, proxy: str):
        opener = build_opener(ProxyHandler({"http": proxy, "https": proxy})) if proxy else None
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if opener is not None:
                    return opener.open(request, timeout=self.timeout_seconds)
                return urlopen(request, timeout=self.timeout_seconds)
            except HTTPError:
                raise
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                time.sleep(self.retry_delay_seconds)
        raise HttpClientError(f"request failed: {request.full_url}") from last_error
