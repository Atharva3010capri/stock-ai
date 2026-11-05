"""Side-by-side comparison of two companies from their extracted figures.

Pure Python/pandas; no LLM call. Only compares what was actually extracted.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from extraction import FIELD_LABELS, NOT_FOUND, load_figures
from fundamentals import compute_ratios, format_ratio_value, latest_and_previous


def _units(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return " ".join(part for part in (record.get("currency"), record.get("units")) if part)


def compare_companies(company_a: str, company_b: str) -> dict[str, Any]:
    """Return {"figures": DataFrame, "ratios": DataFrame, "warning": str}."""
    record_a, record_b = load_figures(company_a), load_figures(company_b)
    warning = ""
    if _units(record_a) and _units(record_b) and _units(record_a) != _units(record_b):
        warning = (
            f"Units differ ({company_a}: {_units(record_a)} vs {company_b}: {_units(record_b)}). "
            "Absolute figures are not directly comparable; ratios are."
        )

    figure_rows = []
    for field, label in FIELD_LABELS.items():
        row = {"Figure": label}
        for name, record in ((company_a, record_a), (company_b, record_b)):
            latest, _ = latest_and_previous((record or {}).get("figures", {}).get(field, []))
            row[name] = f"{latest['value']:,} ({latest['period']})" if latest else NOT_FOUND
        figure_rows.append(row)

    ratio_rows = []
    ratios_a = {r["key"]: r for r in compute_ratios((record_a or {}).get("figures", {}))}
    ratios_b = {r["key"]: r for r in compute_ratios((record_b or {}).get("figures", {}))}
    for key, row_a in ratios_a.items():
        row_b = ratios_b[key]
        ratio_rows.append(
            {
                "Ratio": row_a["name"],
                company_a: format_ratio_value(row_a, (record_a or {}).get("units"), (record_a or {}).get("currency")),
                company_b: format_ratio_value(row_b, (record_b or {}).get("units"), (record_b or {}).get("currency")),
                "Formula": row_a["formula"],
            }
        )
    return {"figures": pd.DataFrame(figure_rows), "ratios": pd.DataFrame(ratio_rows), "warning": warning}
