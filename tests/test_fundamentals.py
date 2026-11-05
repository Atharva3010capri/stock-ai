"""Ratios checked against hand calculations."""

import pytest

from fundamentals import compute_ratios, latest_and_previous, period_sort_key, trend_table


def entry(period, value):
    return {"period": period, "value": value, "page": 1, "source": "t.pdf", "evidence": "x"}


SAMPLE = {
    "revenue": [entry("FY2023", 800.0), entry("FY2024", 1000.0)],
    "net_profit": [entry("FY2024", 150.0), entry("FY2023", 120.0)],
    "operating_profit": [entry("FY2024", 200.0)],
    "eps": [entry("FY2024", 15.0), entry("FY2023", 12.0)],
    "total_debt": [entry("FY2024", 300.0)],
    "equity": [entry("FY2024", 600.0)],
    "total_assets": [entry("FY2024", 1500.0)],
    "current_assets": [entry("FY2024", 500.0)],
    "current_liabilities": [entry("FY2024", 250.0)],
    "operating_cash_flow": [entry("FY2024", 180.0)],
    "capex": [entry("FY2024", -70.0)],  # reports often show capex as an outflow
    "dividends": [],
}


def by_key(rows):
    return {row["key"]: row for row in rows}


def test_hand_calculated_ratios():
    rows = by_key(compute_ratios(SAMPLE))
    assert rows["net_margin"]["value"] == pytest.approx(15.0)          # 150/1000*100
    assert rows["operating_margin"]["value"] == pytest.approx(20.0)    # 200/1000*100
    assert rows["revenue_growth"]["value"] == pytest.approx(25.0)      # (1000-800)/800
    assert rows["profit_growth"]["value"] == pytest.approx(25.0)       # (150-120)/120
    assert rows["roe"]["value"] == pytest.approx(25.0)                 # 150/600*100
    assert rows["debt_to_equity"]["value"] == pytest.approx(0.5)       # 300/600
    assert rows["current_ratio"]["value"] == pytest.approx(2.0)        # 500/250
    assert rows["free_cash_flow"]["value"] == pytest.approx(110.0)     # 180-|−70|
    assert rows["eps_trend"]["value"] == pytest.approx(25.0)           # (15-12)/12


def test_every_ratio_shows_formula_and_inputs():
    for row in compute_ratios(SAMPLE):
        assert row["formula"]
        assert row["inputs"]
        assert row["explanation"]


def test_missing_inputs_are_skipped_not_guessed():
    figures = {"revenue": [entry("FY2024", 1000.0)]}
    rows = by_key(compute_ratios(figures))
    assert rows["net_margin"]["status"] == "skipped"
    assert "net_profit" in rows["net_margin"]["reason"]
    assert rows["net_margin"]["value"] is None
    assert rows["revenue_growth"]["status"] == "skipped"  # only one period


def test_zero_denominator_is_skipped():
    figures = {"net_profit": [entry("FY2024", 10.0)], "revenue": [entry("FY2024", 0.0)]}
    rows = by_key(compute_ratios(figures))
    assert rows["net_margin"]["status"] == "skipped"
    assert "zero" in rows["net_margin"]["reason"].lower()


def test_latest_period_is_detected_regardless_of_order():
    latest, previous = latest_and_previous([entry("FY2022", 1), entry("FY2024", 3), entry("FY2023", 2)])
    assert latest["period"] == "FY2024"
    assert previous["period"] == "FY2023"


@pytest.mark.parametrize(
    "period,expected",
    [("FY2024", (2024, 0)), ("FY 23-24", (2024, 0)), ("Q3 FY2025", (2025, 3)), ("2021", (2021, 0)), ("unknown", (0, 0))],
)
def test_period_sort_key(period, expected):
    assert period_sort_key(period) == expected


def test_trend_table_is_oldest_first():
    rows = trend_table(SAMPLE, fields=("revenue",))
    assert [row["period"] for row in rows] == ["FY2023", "FY2024"]
