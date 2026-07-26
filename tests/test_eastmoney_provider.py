from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digital_oracle.providers.base import ProviderError, ProviderParseError
from digital_oracle.providers.eastmoney import (
    EastmoneyFundFlowQuery,
    EastmoneyKlineQuery,
    EastmoneyProvider,
    EastmoneyQuoteQuery,
    EastmoneySectorFlowQuery,
    to_secid,
)

# Trimmed from a live push2.eastmoney.com response for 002156 (通富微电).
SAMPLE_QUOTE = {
    "rc": 0,
    "data": {
        "f43": 7664,
        "f44": 7680,
        "f45": 6803,
        "f46": 6803,
        "f47": 2634097,
        "f48": 19_486_000_000.0,
        "f57": "002156",
        "f58": "通富微电",
        "f59": 2,
        "f60": 6982,
        "f116": 116_300_000_000.0,
        "f117": 115_000_000_000.0,
        "f162": 8836,
        "f167": 512,
        "f168": 1736,
        "f169": 682,
        "f170": 977,
    },
}

SAMPLE_KLINE = {
    "rc": 0,
    "data": {
        "code": "002156",
        "name": "通富微电",
        "klines": [
            "2026-07-22,71.71,74.90,76.23,70.88,2478838,18000000000.00",
            "2026-07-23,75.99,69.82,76.78,69.30,2057105,15000000000.00",
        ],
    },
}

SAMPLE_FUND_FLOW = {
    "rc": 0,
    "data": {
        "code": "002156",
        "name": "通富微电",
        "klines": [
            "2026-07-22,2331000000.0,-920000000.0,-1411000000.0,-1263000000.0,3594000000.0,12.51,-4.94,-7.57,-6.78,19.29",
        ],
    },
}

SAMPLE_SECTOR = {
    "rc": 0,
    "data": {
        "total": 496,
        "diff": [
            {"f3": -0.01, "f12": "BK1036", "f14": "半导体", "f62": 3305525504.0, "f184": 0.95},
            {"f3": 1.87, "f12": "BK1328", "f14": "集成电路封测", "f62": 3234836736.0, "f184": 6.24},
        ],
    },
}


SAMPLE_TENCENT_KLINE = {
    "code": 0,
    "data": {
        "sz002156": {
            "qfqday": [
                ["2026-07-22", "71.710", "74.900", "76.230", "70.880", "2478838.000"],
                ["2026-07-23", "75.990", "69.820", "76.780", "69.300", "2057105.000"],
            ]
        }
    },
}


class FakeJsonClient:
    """Serves the Tencent sample to gtimg URLs and the Eastmoney sample elsewhere."""

    def __init__(
        self,
        data: Any = None,
        *,
        fail_hosts: tuple[str, ...] = (),
        tencent_data: Any = None,
    ) -> None:
        self.data = data
        self.tencent_data = tencent_data
        self.fail_hosts = fail_hosts
        self.calls: list[tuple[str, Mapping[str, object] | None]] = []

    def get_json(self, url: str, *, params: Mapping[str, object] | None = None) -> Any:
        self.calls.append((url, params))
        if any(host in url for host in self.fail_hosts):
            raise RuntimeError(f"simulated 502 from {url}")
        if "gtimg.cn" in url:
            if self.tencent_data is None:
                raise RuntimeError("simulated Tencent outage")
            return self.tencent_data
        return self.data


class SecidTests(unittest.TestCase):
    def test_shanghai_prefixes(self) -> None:
        self.assertEqual(to_secid("600519"), "1.600519")
        self.assertEqual(to_secid("601138"), "1.601138")
        self.assertEqual(to_secid("688981"), "1.688981")

    def test_shenzhen_prefixes(self) -> None:
        self.assertEqual(to_secid("000977"), "0.000977")
        self.assertEqual(to_secid("002156"), "0.002156")
        self.assertEqual(to_secid("300750"), "0.300750")
        self.assertEqual(to_secid("159870"), "0.159870")

    def test_exchange_suffixes(self) -> None:
        self.assertEqual(to_secid("601138.SS"), "1.601138")
        self.assertEqual(to_secid("601138.SH"), "1.601138")
        self.assertEqual(to_secid("000977.SZ"), "0.000977")

    def test_already_qualified_secid_passes_through(self) -> None:
        self.assertEqual(to_secid("1.600519"), "1.600519")
        self.assertEqual(to_secid("116.00700"), "116.00700")

    def test_non_a_share_rejected(self) -> None:
        for bad in ("NVDA", "43xxxx", "60051", "6005190"):
            with self.assertRaises(ProviderError):
                to_secid(bad)


class QuoteTests(unittest.TestCase):
    def test_prices_are_descaled_by_decimal_field(self) -> None:
        client = FakeJsonClient(SAMPLE_QUOTE)
        quote = EastmoneyProvider(http_client=client).get_quote(
            EastmoneyQuoteQuery(symbol="002156")
        )
        self.assertEqual(quote.symbol, "002156")
        self.assertEqual(quote.name, "通富微电")
        self.assertAlmostEqual(quote.last, 76.64)
        self.assertAlmostEqual(quote.prev_close, 69.82)
        self.assertAlmostEqual(quote.high, 76.80)
        self.assertAlmostEqual(quote.change, 6.82)
        self.assertAlmostEqual(quote.change_pct, 9.77)
        self.assertAlmostEqual(quote.pe_ttm, 88.36)
        self.assertAlmostEqual(quote.turnover_rate_pct, 17.36)
        self.assertEqual(quote.volume_lots, 2634097)

    def test_sends_resolved_secid(self) -> None:
        client = FakeJsonClient(SAMPLE_QUOTE)
        EastmoneyProvider(http_client=client).get_quote(EastmoneyQuoteQuery(symbol="002156"))
        _, params = client.calls[0]
        self.assertEqual(params["secid"], "0.002156")

    def test_missing_data_raises(self) -> None:
        client = FakeJsonClient({"rc": 0, "data": None})
        with self.assertRaises(ProviderError):
            EastmoneyProvider(http_client=client).get_quote(
                EastmoneyQuoteQuery(symbol="002156")
            )


class HostFallbackTests(unittest.TestCase):
    """Eastmoney throttles the realtime hosts per IP; the delay host must cover."""

    def test_quote_falls_back_to_delay_host(self) -> None:
        client = FakeJsonClient(SAMPLE_QUOTE, fail_hosts=("push2.eastmoney.com",))
        quote = EastmoneyProvider(http_client=client).get_quote(
            EastmoneyQuoteQuery(symbol="002156")
        )
        self.assertEqual(quote.name, "通富微电")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("push2.eastmoney.com", client.calls[0][0])
        self.assertIn("push2delay.eastmoney.com", client.calls[1][0])

    def test_history_falls_back_to_delay_host(self) -> None:
        client = FakeJsonClient(SAMPLE_KLINE, fail_hosts=("push2his.eastmoney.com", "gtimg.cn"))
        kline = EastmoneyProvider(http_client=client).get_history(
            EastmoneyKlineQuery(symbol="002156")
        )
        self.assertEqual(len(kline.bars), 2)
        self.assertIn("push2delay.eastmoney.com", client.calls[-1][0])

    def test_primary_host_is_not_retried_when_it_works(self) -> None:
        client = FakeJsonClient(SAMPLE_QUOTE)
        EastmoneyProvider(http_client=client).get_quote(EastmoneyQuoteQuery(symbol="002156"))
        self.assertEqual(len(client.calls), 1)

    def test_empty_rows_from_fallback_is_a_failure_not_an_empty_chart(self) -> None:
        """push2delay answers the kline path with klines: [] instead of an error."""

        class EmptyThenNever:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def get_json(self, url: str, *, params: Any = None) -> Any:
                self.calls.append(url)
                if "gtimg.cn" in url:
                    raise RuntimeError("simulated Tencent outage")
                return {"data": {"code": "002156", "name": "通富微电", "klines": []}}

        client = EmptyThenNever()
        provider = EastmoneyProvider(http_client=client)
        with self.assertRaises(ProviderError) as ctx:
            provider.get_history(EastmoneyKlineQuery(symbol="002156"))
        self.assertIn("carried no rows", str(ctx.exception))
        self.assertEqual(len(client.calls), 3)  # Tencent + both Eastmoney hosts

    def test_dead_host_is_probed_once_not_on_every_call(self) -> None:
        """Retrying a throttled host on every call is what sustains the throttle."""
        client = FakeJsonClient(SAMPLE_QUOTE, fail_hosts=("push2.eastmoney.com",))
        provider = EastmoneyProvider(http_client=client)
        for _ in range(5):
            provider.cache.clear()  # isolate host behaviour from response caching
            provider.get_quote(EastmoneyQuoteQuery(symbol="002156"))
        dead = [u for u, _ in client.calls if "push2.eastmoney.com" in u]
        alive = [u for u, _ in client.calls if "push2delay" in u]
        self.assertEqual(len(dead), 1, "throttled host should be parked after one failure")
        self.assertEqual(len(alive), 5)

    def test_recovered_host_is_preferred_again(self) -> None:
        client = FakeJsonClient(SAMPLE_QUOTE, fail_hosts=("push2.eastmoney.com",))
        provider = EastmoneyProvider(http_client=client)
        provider.get_quote(EastmoneyQuoteQuery(symbol="002156"))
        client.fail_hosts = ()
        provider._host_down_until.clear()  # simulate the cooldown elapsing
        provider.cache.clear()
        provider.get_quote(EastmoneyQuoteQuery(symbol="002156"))
        self.assertIn("push2.eastmoney.com", client.calls[-1][0])

    def test_all_hosts_failing_reports_every_attempt(self) -> None:
        client = FakeJsonClient(SAMPLE_QUOTE, fail_hosts=("eastmoney.com",))
        provider = EastmoneyProvider(http_client=client)
        with self.assertRaises(ProviderError) as ctx:
            provider.get_quote(EastmoneyQuoteQuery(symbol="002156"))
        self.assertIn("push2.eastmoney.com", str(ctx.exception))
        self.assertIn("push2delay.eastmoney.com", str(ctx.exception))


class KlineTests(unittest.TestCase):
    def test_prefers_tencent_and_parses_its_rows(self) -> None:
        client = FakeJsonClient(SAMPLE_KLINE, tencent_data=SAMPLE_TENCENT_KLINE)
        kline = EastmoneyProvider(http_client=client).get_history(
            EastmoneyKlineQuery(symbol="002156", limit=2)
        )
        self.assertIn("gtimg.cn", client.calls[0][0])
        self.assertEqual(len(client.calls), 1)  # Eastmoney not touched
        self.assertEqual(len(kline.bars), 2)
        self.assertAlmostEqual(kline.bars[0].open, 71.71)
        self.assertAlmostEqual(kline.bars[0].close, 74.90)
        self.assertEqual(kline.bars[0].volume_lots, 2478838)
        self.assertIsNone(kline.bars[0].turnover_cny)

    def test_tencent_param_encodes_market_period_and_adjust(self) -> None:
        client = FakeJsonClient(SAMPLE_KLINE, tencent_data=SAMPLE_TENCENT_KLINE)
        EastmoneyProvider(http_client=client).get_history(
            EastmoneyKlineQuery(symbol="601138", period="weekly", adjust="backward", limit=5)
        )
        self.assertEqual(client.calls[0][1]["param"], "sh601138,week,,,5,hfq")

    def test_falls_back_to_eastmoney_when_tencent_is_down(self) -> None:
        client = FakeJsonClient(SAMPLE_KLINE)  # tencent_data=None -> Tencent raises
        kline = EastmoneyProvider(http_client=client).get_history(
            EastmoneyKlineQuery(symbol="002156", limit=2)
        )
        self.assertIn("gtimg.cn", client.calls[0][0])
        self.assertIn("eastmoney.com", client.calls[1][0])
        self.assertEqual(len(kline.bars), 2)
        self.assertAlmostEqual(kline.bars[0].turnover_cny, 18000000000.0)

    def test_parses_eastmoney_rows_on_fallback(self) -> None:
        client = FakeJsonClient(SAMPLE_KLINE)
        kline = EastmoneyProvider(http_client=client).get_history(
            EastmoneyKlineQuery(symbol="002156", limit=2)
        )
        self.assertEqual(len(kline.bars), 2)
        first = kline.bars[0]
        self.assertEqual(first.date, "2026-07-22")
        self.assertAlmostEqual(first.open, 71.71)
        self.assertAlmostEqual(first.close, 74.90)
        self.assertAlmostEqual(first.high, 76.23)
        self.assertAlmostEqual(first.low, 70.88)
        self.assertEqual(first.volume_lots, 2478838)

    def test_period_and_adjust_map_to_api_codes(self) -> None:
        client = FakeJsonClient(SAMPLE_KLINE)
        EastmoneyProvider(http_client=client).get_history(
            EastmoneyKlineQuery(symbol="002156", period="weekly", adjust="backward")
        )
        _, params = client.calls[1]  # calls[0] is the Tencent attempt
        self.assertEqual(params["klt"], "102")
        self.assertEqual(params["fqt"], "2")

    def test_unknown_period_raises(self) -> None:
        provider = EastmoneyProvider(http_client=FakeJsonClient(SAMPLE_KLINE))
        with self.assertRaises(ProviderError):
            provider.get_history(EastmoneyKlineQuery(symbol="002156", period="hourly"))

    def test_short_row_raises(self) -> None:
        client = FakeJsonClient({"data": {"code": "1", "name": "x", "klines": ["2026-07-22,1,2"]}})
        # Tencent has no data in this fixture, so the Eastmoney parser runs
        with self.assertRaises(ProviderParseError):
            EastmoneyProvider(http_client=client).get_history(
                EastmoneyKlineQuery(symbol="002156")
            )


class FundFlowTests(unittest.TestCase):
    def test_main_equals_large_plus_extra_large(self) -> None:
        client = FakeJsonClient(SAMPLE_FUND_FLOW)
        flow = EastmoneyProvider(http_client=client).get_fund_flow(
            EastmoneyFundFlowQuery(symbol="002156", limit=1)
        )
        day = flow.days[0]
        self.assertEqual(day.date, "2026-07-22")
        self.assertAlmostEqual(day.main_net, day.large_net + day.extra_large_net, places=2)
        self.assertAlmostEqual(day.extra_large_net, 3_594_000_000.0)
        self.assertAlmostEqual(day.small_net, -920_000_000.0)
        self.assertAlmostEqual(day.main_net_pct, 12.51)


class SectorFlowTests(unittest.TestCase):
    def test_ranks_by_main_net_inflow(self) -> None:
        client = FakeJsonClient(SAMPLE_SECTOR)
        sectors = EastmoneyProvider(http_client=client).list_sector_fund_flow(
            EastmoneySectorFlowQuery(limit=2)
        )
        self.assertEqual([s.name for s in sectors], ["半导体", "集成电路封测"])
        self.assertAlmostEqual(sectors[0].main_net_cny, 3305525504.0)
        self.assertAlmostEqual(sectors[1].change_pct, 1.87)
        _, params = client.calls[0]
        self.assertEqual(params["fid"], "f62")
        self.assertEqual(params["fs"], "m:90+t:2")

    def test_concept_board_uses_different_filter(self) -> None:
        client = FakeJsonClient(SAMPLE_SECTOR)
        EastmoneyProvider(http_client=client).list_sector_fund_flow(
            EastmoneySectorFlowQuery(kind="concept")
        )
        _, params = client.calls[0]
        self.assertEqual(params["fs"], "m:90+t:3")

    def test_unknown_kind_raises(self) -> None:
        provider = EastmoneyProvider(http_client=FakeJsonClient(SAMPLE_SECTOR))
        with self.assertRaises(ProviderError):
            provider.list_sector_fund_flow(EastmoneySectorFlowQuery(kind="etf"))


if __name__ == "__main__":
    unittest.main()
