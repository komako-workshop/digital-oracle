from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digital_oracle.providers.bis import (
    CREDIT_GAP_SERIES,
    BisCreditGapQuery,
    BisProvider,
    BisRateQuery,
)

SAMPLE_POLICY_RATES_CSV = """\
FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
M,US,2026-01,4.50
M,US,2025-12,4.50
M,US,2025-11,4.75
M,JP,2026-01,0.50
M,JP,2025-12,0.25
"""

SAMPLE_CREDIT_GAP_CSV = """\
FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
Q,US,2025-Q3,2.1
Q,US,2025-Q2,1.8
Q,CN,2025-Q3,15.3
Q,CN,2025-Q2,14.9
"""

SAMPLE_EMPTY_CSV = """\
FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
"""

SAMPLE_WITH_MISSING_VALUES_CSV = """\
FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
M,US,2026-01,4.50
M,US,2025-12,
M,GB,2026-01,4.75
"""


class FakeTextClient:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls: list[tuple[str, Mapping[str, object] | None]] = []

    def get_text(self, url: str, *, params: Mapping[str, object] | None = None) -> str:
        self.calls.append((url, params))
        return self.text


class BisProviderPolicyRatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_client = FakeTextClient(SAMPLE_POLICY_RATES_CSV)
        self.provider = BisProvider(http_client=self.fake_client)

    def test_parse_policy_rates(self) -> None:
        rates = self.provider.get_policy_rates(
            BisRateQuery(countries=("US", "JP"), start_year=2025)
        )

        self.assertEqual(len(rates), 5)
        self.assertEqual(rates[0].country, "US")
        self.assertEqual(rates[0].period, "2026-01")
        self.assertAlmostEqual(rates[0].rate, 4.50)
        self.assertEqual(rates[3].country, "JP")
        self.assertEqual(rates[3].period, "2026-01")
        self.assertAlmostEqual(rates[3].rate, 0.50)

    def test_builds_correct_url_and_params(self) -> None:
        self.provider.get_policy_rates(
            BisRateQuery(countries=("US", "JP"), start_year=2025)
        )

        self.assertEqual(len(self.fake_client.calls), 1)
        url, params = self.fake_client.calls[0]
        self.assertIn("WS_CBPOL/M.US+JP", url)
        assert params is not None
        self.assertEqual(params["startPeriod"], 2025)
        self.assertEqual(params["detail"], "dataonly")
        self.assertEqual(params["format"], "csv")

    def test_empty_csv_returns_empty_list(self) -> None:
        client = FakeTextClient(SAMPLE_EMPTY_CSV)
        provider = BisProvider(http_client=client)

        rates = provider.get_policy_rates(BisRateQuery())
        self.assertEqual(len(rates), 0)

    def test_missing_obs_value_rows_skipped(self) -> None:
        client = FakeTextClient(SAMPLE_WITH_MISSING_VALUES_CSV)
        provider = BisProvider(http_client=client)

        rates = provider.get_policy_rates(BisRateQuery(countries=("US", "GB")))
        self.assertEqual(len(rates), 2)
        self.assertEqual(rates[0].country, "US")
        self.assertAlmostEqual(rates[0].rate, 4.50)
        self.assertEqual(rates[1].country, "GB")
        self.assertAlmostEqual(rates[1].rate, 4.75)

    def test_default_query_uses_us_and_2020(self) -> None:
        self.provider.get_policy_rates()

        url, params = self.fake_client.calls[0]
        self.assertIn("WS_CBPOL/M.US", url)
        assert params is not None
        self.assertEqual(params["startPeriod"], 2020)


class BisProviderCreditGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_client = FakeTextClient(SAMPLE_CREDIT_GAP_CSV)
        self.provider = BisProvider(http_client=self.fake_client)

    def test_parse_credit_gaps(self) -> None:
        gaps = self.provider.get_credit_to_gdp(
            BisCreditGapQuery(countries=("US", "CN"), start_year=2025)
        )

        self.assertEqual(len(gaps), 4)
        self.assertEqual(gaps[0].country, "US")
        self.assertEqual(gaps[0].period, "2025-Q3")
        self.assertAlmostEqual(gaps[0].gap_pct, 2.1)
        self.assertEqual(gaps[2].country, "CN")
        self.assertAlmostEqual(gaps[2].gap_pct, 15.3)

    def test_builds_correct_url_for_credit_gap(self) -> None:
        self.provider.get_credit_to_gdp(
            BisCreditGapQuery(countries=("US", "CN"), start_year=2015)
        )

        url, params = self.fake_client.calls[0]
        self.assertIn("WS_CREDIT_GAP/Q.US+CN.P.A.C", url)
        assert params is not None
        self.assertEqual(params["startPeriod"], 2015)

    def test_default_credit_gap_query(self) -> None:
        self.provider.get_credit_to_gdp()

        url, params = self.fake_client.calls[0]
        self.assertIn("WS_CREDIT_GAP/Q.US.P.A.C", url)
        assert params is not None
        self.assertEqual(params["startPeriod"], 2015)

    def test_ratio_series_is_selectable(self) -> None:
        self.provider.get_credit_to_gdp(
            BisCreditGapQuery(countries=("US",), series=CREDIT_GAP_SERIES["ratio"])
        )

        url, _ = self.fake_client.calls[0]
        self.assertIn("WS_CREDIT_GAP/Q.US.P.A.A", url)


class BisProviderMetadataTests(unittest.TestCase):
    def test_describe(self) -> None:
        provider = BisProvider(http_client=FakeTextClient())
        meta = provider.describe()

        self.assertEqual(meta.provider_id, "bis")
        self.assertEqual(meta.display_name, "Bank for International Settlements")
        self.assertEqual(meta.capabilities, ("policy_rates", "credit_gaps"))


if __name__ == "__main__":
    unittest.main()
