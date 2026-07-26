"""China A-share trading data via Eastmoney's public quote endpoints.

Covers the signal layer Yahoo cannot reach for mainland listings: order-size
fund flow and sector rotation. Free, keyless, JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from digital_oracle.http import JsonHttpClient, UrllibJsonClient

from ._coerce import _coerce_float, _coerce_int
from .base import ProviderError, ProviderParseError, SignalProvider

QUOTE_PATH = "/api/qt/stock/get"
KLINE_PATH = "/api/qt/stock/kline/get"
FUND_FLOW_PATH = "/api/qt/stock/fflow/daykline/get"
SECTOR_PATH = "/api/qt/clist/get"

# Eastmoney throttles the realtime hosts per source IP — a burst of requests from
# one datacenter address gets 502s and connection timeouts within minutes. The
# delay host stays up under the same load and serves every one of these paths, so
# it is worth a second attempt before giving up. Its quotes lag ~15 minutes.
LIVE_HOSTS = ("https://push2.eastmoney.com", "https://push2delay.eastmoney.com")
HISTORY_HOSTS = ("https://push2his.eastmoney.com", "https://push2delay.eastmoney.com")

# OHLCV comes from Tencent first. push2his is the host Eastmoney throttles hardest
# and the delay host answers the kline path with an empty array rather than an
# error, so Eastmoney is the weaker source for exactly this one call. Tencent has
# no such limit and ships the same field order (minus turnover).
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_TENCENT_PERIODS = {"daily": "day", "weekly": "week", "monthly": "month"}
_TENCENT_ADJUST = {"none": "", "forward": "qfq", "backward": "hfq"}

# Eastmoney rejects the stdlib default agent, so every request needs a browser UA.
_BROWSER_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

# Shanghai-listed prefixes; everything else on the mainland boards is Shenzhen.
_SHANGHAI_PREFIXES = ("5", "6", "9", "110", "113", "118", "132", "204")

_KLINE_PERIODS = {"daily": "101", "weekly": "102", "monthly": "103"}
_ADJUST_MODES = {"none": "0", "forward": "1", "backward": "2"}


def to_secid(symbol: str) -> str:
    """Map a bare A-share code (or an already-qualified secid) to Eastmoney's secid.

    ``600519`` -> ``1.600519``, ``000977`` -> ``0.000977``.
    """
    cleaned = symbol.strip().upper()
    market_part, _, code_part = cleaned.partition(".")
    if code_part.isdigit() and market_part.isdigit() and len(market_part) <= 3:
        return cleaned
    for suffix, market in ((".SH", "1"), (".SS", "1"), (".SZ", "0"), (".BJ", "0")):
        if cleaned.endswith(suffix):
            return f"{market}.{cleaned[: -len(suffix)]}"
    if not cleaned.isdigit() or len(cleaned) != 6:
        raise ProviderError(
            f"not an A-share code: {symbol!r} (expected 6 digits, e.g. 600519 or 000977)"
        )
    market = "1" if cleaned.startswith(_SHANGHAI_PREFIXES) else "0"
    return f"{market}.{cleaned}"


@dataclass(frozen=True)
class EastmoneyQuoteQuery:
    symbol: str


@dataclass(frozen=True)
class EastmoneyQuote:
    symbol: str
    name: str
    last: float | None
    prev_close: float | None
    open: float | None
    high: float | None
    low: float | None
    change: float | None
    change_pct: float | None
    volume_lots: int | None
    turnover_cny: float | None
    turnover_rate_pct: float | None
    pe_ttm: float | None
    pb: float | None
    market_cap_cny: float | None
    float_market_cap_cny: float | None


@dataclass(frozen=True)
class EastmoneyKlineQuery:
    symbol: str
    period: str = "daily"  # daily | weekly | monthly
    adjust: str = "forward"  # none | forward | backward
    limit: int = 120


@dataclass(frozen=True)
class EastmoneyBar:
    date: str
    open: float | None
    close: float | None
    high: float | None
    low: float | None
    volume_lots: int | None
    turnover_cny: float | None


@dataclass(frozen=True)
class EastmoneyKline:
    symbol: str
    name: str
    period: str
    adjust: str
    bars: tuple[EastmoneyBar, ...]


@dataclass(frozen=True)
class EastmoneyFundFlowQuery:
    symbol: str
    limit: int = 20


@dataclass(frozen=True)
class EastmoneyFundFlowDay:
    """Net inflow in CNY, split by order size. Positive = buying pressure."""

    date: str
    main_net: float | None  # 主力 = extra_large + large
    extra_large_net: float | None
    large_net: float | None
    medium_net: float | None
    small_net: float | None
    main_net_pct: float | None


@dataclass(frozen=True)
class EastmoneyFundFlow:
    symbol: str
    name: str
    days: tuple[EastmoneyFundFlowDay, ...]


@dataclass(frozen=True)
class EastmoneySectorFlowQuery:
    kind: str = "industry"  # industry | concept
    limit: int = 20


@dataclass(frozen=True)
class EastmoneySectorFlow:
    code: str
    name: str
    change_pct: float | None
    main_net_cny: float | None
    main_net_pct: float | None


_SECTOR_BOARDS = {"industry": "m:90+t:2", "concept": "m:90+t:3"}


def _scaled(raw: object, decimals: int | None) -> float | None:
    """Eastmoney ships prices as integers plus a decimal-place count (f59)."""
    value = _coerce_float(raw)
    if value is None:
        return None
    return value / (10 ** (decimals if decimals is not None else 2))


def _split_row(row: object, expected: int) -> list[str]:
    if not isinstance(row, str):
        raise ProviderParseError("expected kline row to be a comma-separated string")
    parts = row.split(",")
    if len(parts) < expected:
        raise ProviderParseError(
            f"expected at least {expected} fields in kline row, got {len(parts)}"
        )
    return parts


def _payload(response: object) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        raise ProviderParseError("expected Eastmoney response to be an object")
    data = response.get("data")
    if data is None:
        raise ProviderError(
            "Eastmoney returned no data — the code may be delisted, suspended, "
            "or not a mainland listing"
        )
    if not isinstance(data, Mapping):
        raise ProviderParseError("expected Eastmoney 'data' to be an object")
    return data


@dataclass
class EastmoneyProvider(SignalProvider):
    """A-share quotes, OHLCV history, order-size fund flow and sector rotation.

    Quotes, fund flow and sector rotation come from Eastmoney. ``get_history``
    prefers Tencent and only falls back to Eastmoney, because push2his is the
    host Eastmoney throttles hardest and its delay host does not serve klines.
    """

    provider_id: str = "eastmoney"
    display_name: str = "Eastmoney (China A-share)"
    capabilities: tuple[str, ...] = (
        "price_quote",
        "price_history",
        "fund_flow",
        "sector_fund_flow",
    )

    def __init__(self, http_client: JsonHttpClient | None = None) -> None:
        self.http_client: JsonHttpClient = http_client or UrllibJsonClient(
            headers=_BROWSER_HEADERS
        )

    def _fetch(
        self,
        hosts: Sequence[str],
        path: str,
        params: Mapping[str, Any],
        *,
        require: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Mapping[str, Any]:
        """Try each host in turn; ``require`` rejects a 200 that carries no rows.

        The delay host answers the kline path with an empty ``klines`` array
        rather than an error, which would otherwise surface as a silently empty
        chart instead of a failure worth reporting.
        """
        failures = []
        for host in hosts:
            try:
                data = _payload(self.http_client.get_json(host + path, params=params))
            except ProviderParseError:
                raise
            except Exception as exc:  # noqa: BLE001 — try the next host, report all
                failures.append(f"{host}: {type(exc).__name__}: {exc}")
                continue
            if require is not None and not require(data):
                failures.append(f"{host}: responded 200 but carried no rows")
                continue
            return data
        raise ProviderError("every Eastmoney host failed — " + "; ".join(failures))

    def get_quote(self, query: EastmoneyQuoteQuery) -> EastmoneyQuote:
        data = self._fetch(
            LIVE_HOSTS,
            QUOTE_PATH,
            {
                "secid": to_secid(query.symbol),
                "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f59,f60,f116,f117,f162,f167,f168,f169,f170",
            },
        )
        decimals = _coerce_int(data.get("f59"))
        return EastmoneyQuote(
            symbol=str(data.get("f57", "")),
            name=str(data.get("f58", "")),
            last=_scaled(data.get("f43"), decimals),
            prev_close=_scaled(data.get("f60"), decimals),
            open=_scaled(data.get("f46"), decimals),
            high=_scaled(data.get("f44"), decimals),
            low=_scaled(data.get("f45"), decimals),
            change=_scaled(data.get("f169"), decimals),
            change_pct=_scaled(data.get("f170"), 2),
            volume_lots=_coerce_int(data.get("f47")),
            turnover_cny=_coerce_float(data.get("f48")),
            turnover_rate_pct=_scaled(data.get("f168"), 2),
            pe_ttm=_scaled(data.get("f162"), 2),
            pb=_scaled(data.get("f167"), 2),
            market_cap_cny=_coerce_float(data.get("f116")),
            float_market_cap_cny=_coerce_float(data.get("f117")),
        )

    def get_history(self, query: EastmoneyKlineQuery) -> EastmoneyKline:
        period = _KLINE_PERIODS.get(query.period)
        if period is None:
            raise ProviderError(
                f"unknown period {query.period!r}; expected one of {sorted(_KLINE_PERIODS)}"
            )
        adjust = _ADJUST_MODES.get(query.adjust)
        if adjust is None:
            raise ProviderError(
                f"unknown adjust {query.adjust!r}; expected one of {sorted(_ADJUST_MODES)}"
            )
        limit = max(1, query.limit)
        failures = []
        try:
            payload = self.http_client.get_json(
                TENCENT_KLINE_URL,
                params={
                    "param": ",".join(
                        [
                            to_tencent_symbol(query.symbol),
                            _TENCENT_PERIODS[query.period],
                            "",
                            "",
                            str(limit),
                            _TENCENT_ADJUST[query.adjust],
                        ]
                    )
                },
            )
            bars = _tencent_bars(
                payload, _TENCENT_PERIODS[query.period], _TENCENT_ADJUST[query.adjust]
            )
            return EastmoneyKline(
                symbol=to_secid(query.symbol).partition(".")[2],
                name="",  # Tencent's kline payload carries no display name
                period=query.period,
                adjust=query.adjust,
                bars=bars,
            )
        except Exception as exc:  # noqa: BLE001 — any Tencent problem falls through
            # Includes parse errors: for a fallback chain, a malformed response
            # from the preferred source is a reason to try the next one, not to
            # fail the call.
            failures.append(f"{TENCENT_KLINE_URL}: {type(exc).__name__}: {exc}")

        data = self._fetch(
            HISTORY_HOSTS,
            KLINE_PATH,
            {
                "secid": to_secid(query.symbol),
                "fields1": "f1,f2,f3",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
                "klt": period,
                "fqt": adjust,
                "end": "20500101",
                "lmt": limit,
            },
            require=_has_rows,
        )
        bars = []
        for row in _rows(data):
            p = _split_row(row, 7)
            bars.append(
                EastmoneyBar(
                    date=p[0],
                    open=_coerce_float(p[1]),
                    close=_coerce_float(p[2]),
                    high=_coerce_float(p[3]),
                    low=_coerce_float(p[4]),
                    volume_lots=_coerce_int(p[5]),
                    turnover_cny=_coerce_float(p[6]),
                )
            )
        return EastmoneyKline(
            symbol=str(data.get("code", "")),
            name=str(data.get("name", "")),
            period=query.period,
            adjust=query.adjust,
            bars=tuple(bars),
        )

    def get_fund_flow(self, query: EastmoneyFundFlowQuery) -> EastmoneyFundFlow:
        data = self._fetch(
            HISTORY_HOSTS,
            FUND_FLOW_PATH,
            {
                "secid": to_secid(query.symbol),
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",
                "lmt": max(1, query.limit),
            },
            require=_has_rows,
        )
        days = []
        for row in _rows(data):
            p = _split_row(row, 7)
            days.append(
                EastmoneyFundFlowDay(
                    date=p[0],
                    main_net=_coerce_float(p[1]),
                    small_net=_coerce_float(p[2]),
                    medium_net=_coerce_float(p[3]),
                    large_net=_coerce_float(p[4]),
                    extra_large_net=_coerce_float(p[5]),
                    main_net_pct=_coerce_float(p[6]),
                )
            )
        return EastmoneyFundFlow(
            symbol=str(data.get("code", "")),
            name=str(data.get("name", "")),
            days=tuple(days),
        )

    def list_sector_fund_flow(
        self, query: EastmoneySectorFlowQuery | None = None
    ) -> list[EastmoneySectorFlow]:
        query = query or EastmoneySectorFlowQuery()
        board = _SECTOR_BOARDS.get(query.kind)
        if board is None:
            raise ProviderError(
                f"unknown sector kind {query.kind!r}; expected one of {sorted(_SECTOR_BOARDS)}"
            )
        data = self._fetch(
            LIVE_HOSTS,
            SECTOR_PATH,
            {
                "fid": "f62",  # rank by main-force net inflow
                "po": 1,
                "pz": max(1, query.limit),
                "pn": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fs": board,
                "fields": "f3,f12,f14,f62,f184",
            },
        )
        rows = data.get("diff")
        if not isinstance(rows, Sequence):
            raise ProviderParseError("expected Eastmoney sector 'diff' to be a list")
        sectors = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            sectors.append(
                EastmoneySectorFlow(
                    code=str(row.get("f12", "")),
                    name=str(row.get("f14", "")),
                    change_pct=_coerce_float(row.get("f3")),
                    main_net_cny=_coerce_float(row.get("f62")),
                    main_net_pct=_coerce_float(row.get("f184")),
                )
            )
        return sectors


def to_tencent_symbol(symbol: str) -> str:
    """``600519`` -> ``sh600519``, ``002156`` -> ``sz002156``."""
    market, _, code = to_secid(symbol).partition(".")
    return ("sh" if market == "1" else "sz") + code


def _tencent_bars(payload: object, period: str, adjust: str) -> tuple[EastmoneyBar, ...]:
    if not isinstance(payload, Mapping):
        raise ProviderParseError("expected Tencent response to be an object")
    data = payload.get("data")
    if not isinstance(data, Mapping) or not data:
        raise ProviderError("Tencent returned no data for this code")
    quote = next(iter(data.values()))
    if not isinstance(quote, Mapping):
        raise ProviderParseError("expected Tencent 'data' entry to be an object")
    rows = quote.get(f"{adjust}{period}")
    if not isinstance(rows, Sequence) or not rows:
        raise ProviderError("Tencent returned no bars for this code")
    bars = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 6:
            raise ProviderParseError("expected Tencent kline row to have 6 fields")
        bars.append(
            EastmoneyBar(
                date=str(row[0]),
                open=_coerce_float(row[1]),
                close=_coerce_float(row[2]),
                high=_coerce_float(row[3]),
                low=_coerce_float(row[4]),
                volume_lots=_coerce_int(_coerce_float(row[5])),
                turnover_cny=None,  # Tencent does not ship turnover on this endpoint
            )
        )
    return tuple(bars)


def _has_rows(data: Mapping[str, Any]) -> bool:
    klines = data.get("klines")
    return isinstance(klines, Sequence) and not isinstance(klines, (str, bytes)) and bool(klines)


def _rows(data: Mapping[str, Any]) -> Sequence[Any]:
    klines = data.get("klines")
    if klines is None:
        return ()
    if not isinstance(klines, Sequence) or isinstance(klines, (str, bytes)):
        raise ProviderParseError("expected Eastmoney 'klines' to be a list")
    return klines
