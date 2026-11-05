"""Fundamental ratios computed in plain Python from extracted figures.

The LLM never computes ratios. Every ratio shows its formula and inputs,
and is skipped (not guessed) when an input is missing or zero.
"""

from __future__ import annotations

import re
from typing import Any

Entry = dict[str, Any]

EXPLANATIONS = {
    "net_margin": "Of every 100 units of revenue, how much is left as profit after all costs and tax.",
    "operating_margin": "Profit from the core business per 100 units of revenue, before interest and tax.",
    "revenue_growth": "How much sales grew (or shrank) versus the previous period.",
    "profit_growth": "How much net profit grew (or shrank) versus the previous period.",
    "roe": "Return on equity: profit earned per 100 units of shareholders' money.",
    "debt_to_equity": "How much the company borrowed for each unit of shareholders' money. Normal levels differ a lot by sector.",
    "current_ratio": "Short-term assets divided by short-term bills; above 1 means bills are covered on paper.",
    "free_cash_flow": "Cash from operations minus money spent on long-term assets; cash the business actually generates.",
    "eps_trend": "Change in earnings per share versus the previous period.",
}


def period_sort_key(period: str) -> tuple[int, int]:
    """Turn 'FY2024', 'FY 23-24', 'Q3 2025', '2023' into something sortable.

    Returns (year, quarter). Unknown periods sort first (oldest).
    """
    text = str(period)
    years = [int(y) for y in re.findall(r"(?<!\d)(\d{4})(?!\d)", text)]
    if not years:
        short = re.findall(r"(?<!\d)(\d{2})(?!\d)", text)
        years = [2000 + int(s) for s in short]
    year = max(years) if years else 0
    quarter_match = re.search(r"[Qq]([1-4])", text)
    quarter = int(quarter_match.group(1)) if quarter_match else 0
    return (year, quarter)


def latest_and_previous(entries: list[Entry]) -> tuple[Entry | None, Entry | None]:
    """Pick the most recent period and the one before it."""
    if not entries:
        return None, None
    ordered = sorted(entries, key=lambda entry: period_sort_key(entry.get("period", "")), reverse=True)
    latest = ordered[0]
    previous = ordered[1] if len(ordered) > 1 else None
    return latest, previous


def _value(entry: Entry | None) -> float | None:
    if entry is None:
        return None
    value = entry.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _ratio(
    key: str,
    name: str,
    formula: str,
    inputs: dict[str, float | None],
    compute,
    unit: str,
    period: str | None,
) -> dict[str, Any]:
    """Build one ratio row, skipping cleanly when inputs are missing."""
    missing = [label for label, value in inputs.items() if value is None]
    row = {
        "key": key,
        "name": name,
        "formula": formula,
        "inputs": inputs,
        "unit": unit,
        "period": period or "",
        "explanation": EXPLANATIONS[key],
        "value": None,
        "status": "ok",
        "reason": "",
    }
    if missing:
        row["status"] = "skipped"
        row["reason"] = "Missing input(s): " + ", ".join(missing)
        return row
    try:
        row["value"] = compute()
    except ZeroDivisionError:
        row["status"] = "skipped"
        row["reason"] = "Denominator is zero"
    return row


def compute_ratios(figures: dict[str, list[Entry]]) -> list[dict[str, Any]]:
    """Compute all ratios from {field: [entries]}. Pure function, easy to test."""
    latest = {field: latest_and_previous(entries)[0] for field, entries in figures.items()}
    previous = {field: latest_and_previous(entries)[1] for field, entries in figures.items()}

    def L(field: str) -> float | None:  # noqa: N802 - short helper
        return _value(latest.get(field))

    def P(field: str) -> float | None:  # noqa: N802
        return _value(previous.get(field))

    def per(field: str) -> str | None:
        entry = latest.get(field)
        return entry.get("period") if entry else None

    def div(numerator: float, denominator: float) -> float:
        """Plain division that raises ZeroDivisionError (caught in _ratio)."""
        if denominator == 0:
            raise ZeroDivisionError
        return numerator / denominator

    def growth(new: float, old: float) -> float:
        return div(new - old, abs(old)) * 100

    rows = [
        _ratio("net_margin", "Net profit margin", "net_profit / revenue x 100",
               {"net_profit": L("net_profit"), "revenue": L("revenue")},
               lambda: div(L("net_profit"), L("revenue")) * 100,
               "%", per("revenue")),
        _ratio("operating_margin", "Operating margin", "operating_profit / revenue x 100",
               {"operating_profit": L("operating_profit"), "revenue": L("revenue")},
               lambda: div(L("operating_profit"), L("revenue")) * 100,
               "%", per("revenue")),
        _ratio("revenue_growth", "Revenue growth", "(revenue_latest - revenue_previous) / |revenue_previous| x 100",
               {"revenue_latest": L("revenue"), "revenue_previous": P("revenue")},
               lambda: growth(L("revenue"), P("revenue")), "%",
               f"{per('revenue')} vs {previous['revenue']['period']}" if previous.get("revenue") else per("revenue")),
        _ratio("profit_growth", "Net profit growth", "(profit_latest - profit_previous) / |profit_previous| x 100",
               {"net_profit_latest": L("net_profit"), "net_profit_previous": P("net_profit")},
               lambda: growth(L("net_profit"), P("net_profit")), "%",
               f"{per('net_profit')} vs {previous['net_profit']['period']}" if previous.get("net_profit") else per("net_profit")),
        _ratio("roe", "Return on equity (ROE)", "net_profit / equity x 100",
               {"net_profit": L("net_profit"), "equity": L("equity")},
               lambda: div(L("net_profit"), L("equity")) * 100,
               "%", per("net_profit")),
        _ratio("debt_to_equity", "Debt-to-equity", "total_debt / equity",
               {"total_debt": L("total_debt"), "equity": L("equity")},
               lambda: div(L("total_debt"), L("equity")),
               "x", per("total_debt")),
        _ratio("current_ratio", "Current ratio", "current_assets / current_liabilities",
               {"current_assets": L("current_assets"), "current_liabilities": L("current_liabilities")},
               lambda: div(L("current_assets"), L("current_liabilities")),
               "x", per("current_assets")),
        _ratio("free_cash_flow", "Free cash flow", "operating_cash_flow - |capex|",
               {"operating_cash_flow": L("operating_cash_flow"), "capex": L("capex")},
               lambda: L("operating_cash_flow") - abs(L("capex")), "currency", per("operating_cash_flow")),
        _ratio("eps_trend", "EPS trend", "(eps_latest - eps_previous) / |eps_previous| x 100",
               {"eps_latest": L("eps"), "eps_previous": P("eps")},
               lambda: growth(L("eps"), P("eps")), "%",
               f"{per('eps')} vs {previous['eps']['period']}" if previous.get("eps") else per("eps")),
    ]
    for row in rows:
        if row["value"] is not None:
            row["value"] = round(row["value"], 2)
    return rows


def trend_table(figures: dict[str, list[Entry]], fields: tuple[str, ...] = ("revenue", "net_profit", "operating_profit")) -> list[dict[str, Any]]:
    """Rows of {period, field, value} for charting, oldest period first."""
    rows = []
    for field in fields:
        for entry in figures.get(field, []):
            if isinstance(entry.get("value"), (int, float)):
                rows.append({"period": entry["period"], "field": field, "value": float(entry["value"])})
    rows.sort(key=lambda row: period_sort_key(row["period"]))
    return rows


def format_ratio_value(row: dict[str, Any], units: str | None = None, currency: str | None = None) -> str:
    """Pretty string for a ratio row."""
    if row["value"] is None:
        return "Skipped"
    if row["unit"] == "%":
        return f"{row['value']:.2f}%"
    if row["unit"] == "x":
        return f"{row['value']:.2f}x"
    label = " ".join(part for part in (currency, units) if part)
    return f"{row['value']:,.2f} {label}".strip()
