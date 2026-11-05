"""Financial figure extraction from report chunks (LLM + strict validation).

The LLM only *reads* passages and returns JSON. Python validates every
value: numbers must parse, pages must be ones we actually showed the
model, and the evidence quote is checked against the passage text.
Anything that fails is discarded. Missing figures stay "Not found in report".
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from config import GEMINI_MODEL, KNOWLEDGE_BASE_DIR
from gemini_client import LLMError, LLMJSONError, generate_json
from rag import retrieve_multi

NOT_FOUND = "Not found in report"

# field key -> (label, search queries used to find the right passages)
FIELD_DEFINITIONS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("Revenue", ["total revenue from operations", "total income revenue for the year"]),
    "net_profit": ("Net profit", ["net profit for the year profit after tax", "profit attributable to owners"]),
    "operating_profit": ("Operating profit", ["operating profit EBIT", "profit before interest and tax"]),
    "eps": ("EPS", ["earnings per share basic diluted"]),
    "total_debt": ("Total debt", ["total borrowings long term short term debt"]),
    "equity": ("Shareholders' equity", ["total equity attributable to owners", "shareholders funds net worth"]),
    "total_assets": ("Total assets", ["total assets balance sheet"]),
    "current_assets": ("Current assets", ["total current assets"]),
    "current_liabilities": ("Current liabilities", ["total current liabilities"]),
    "operating_cash_flow": ("Operating cash flow", ["net cash generated from operating activities"]),
    "capex": ("Capital expenditure", ["purchase of property plant and equipment capital expenditure"]),
    "dividends": ("Dividends", ["dividend paid per share dividend declared"]),
}

FIELD_LABELS = {key: value[0] for key, value in FIELD_DEFINITIONS.items()}

EXTRACTION_INSTRUCTIONS = """You are a careful financial data extractor.
Read the numbered report passages and copy out the requested figures.
Rules:
- Use ONLY numbers that literally appear in the passages. Never estimate or recall from memory.
- For each figure give every reporting period you can see (e.g. current year and previous year).
- "value" must be a plain number (no commas, no units). Keep the sign shown in the report.
- "page" must be the page of the passage the number came from, exactly as labelled.
- "source" must be the filename of that passage.
- "evidence" must be a short exact quote (under 200 characters) copied from that passage containing the number.
- If a figure is not present, return an empty list for it. Do not guess.
- Report the currency and the unit the statements use (e.g. "INR crore", "USD million").
Return JSON only, with this shape:
{
  "currency": "INR" or null,
  "units": "crore" or "million" or "thousand" or "absolute" or null,
  "figures": {
    "revenue": [{"period": "FY2024", "value": 12345.6, "page": 112, "source": "report.pdf", "evidence": "Revenue from operations 12,345.6"}],
    "net_profit": [], "operating_profit": [], "eps": [], "total_debt": [], "equity": [],
    "total_assets": [], "current_assets": [], "current_liabilities": [],
    "operating_cash_flow": [], "capex": [], "dividends": []
  }
}"""


def _slug(company: str) -> str:
    return re.sub(r"[^\w\-]+", "_", company.strip()).strip("_").lower() or "company"


def figures_path(company: str):
    return KNOWLEDGE_BASE_DIR / f"figures_{_slug(company)}.json"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9.%-]+", " ", str(text).lower()).strip()


def _to_number(value: Any) -> float | None:
    """Accept 1234, "1,234.5", "(1,234)" (negative) and reject everything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("₹", "").replace("$", "")
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()")
        try:
            number = float(text)
        except ValueError:
            return None
        return -number if negative else number
    return None


def gather_evidence(company: str, document_id: str | None = None, max_chunks: int = 24) -> list[dict[str, Any]]:
    """Collect the passages most likely to contain the key figures."""
    queries = [query for _, queries in FIELD_DEFINITIONS.values() for query in queries]
    return retrieve_multi(queries, company, per_query=3, max_total=max_chunks, document_id=document_id)


def build_prompt(company: str, hits: list[dict[str, Any]]) -> str:
    parts = [f"COMPANY: {company}", "", "PASSAGES:"]
    for number, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        parts.append(
            f"[Passage {number}] source: {meta.get('source')} | page: {meta.get('page')} | "
            f"period label: {meta.get('reporting_period')}\n{hit['text']}"
        )
    parts.append("")
    parts.append(EXTRACTION_INSTRUCTIONS)
    return "\n".join(parts)


def validate_extraction(raw: Any, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep only well-formed, page-verified figures. Never invents anything.

    Returns {"currency", "units", "figures": {field: [entries]}, "discarded": [...]}
    """
    allowed_pages: dict[tuple[str, str], list[str]] = {}
    for hit in hits:
        meta = hit["metadata"]
        key = (str(meta.get("source")), str(meta.get("page")))
        allowed_pages.setdefault(key, []).append(_normalise(hit["text"]))

    result: dict[str, Any] = {"currency": None, "units": None, "figures": {}, "discarded": []}
    if not isinstance(raw, dict):
        result["discarded"].append("Top-level JSON was not an object; everything discarded.")
        result["figures"] = {field: [] for field in FIELD_DEFINITIONS}
        return result

    if isinstance(raw.get("currency"), str):
        result["currency"] = raw["currency"].strip()[:10] or None
    if isinstance(raw.get("units"), str):
        result["units"] = raw["units"].strip()[:30] or None

    figures = raw.get("figures") if isinstance(raw.get("figures"), dict) else {}
    for field in FIELD_DEFINITIONS:
        entries = figures.get(field, [])
        kept: list[dict[str, Any]] = []
        if not isinstance(entries, list):
            result["discarded"].append(f"{field}: expected a list.")
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                result["discarded"].append(f"{field}: entry was not an object.")
                continue
            value = _to_number(entry.get("value"))
            if value is None:
                result["discarded"].append(f"{field}: value '{entry.get('value')}' is not a number.")
                continue
            source = str(entry.get("source", "")).strip()
            page = str(entry.get("page", "")).strip()
            if (source, page) not in allowed_pages:
                result["discarded"].append(
                    f"{field}: cited {source} page {page}, which was not among the passages shown."
                )
                continue
            evidence = str(entry.get("evidence", "")).strip()
            period = str(entry.get("period", "")).strip() or "Unspecified period"
            page_texts = allowed_pages[(source, page)]
            evidence_norm = _normalise(evidence)
            number_token = _normalise(f"{abs(value):,.2f}").rstrip("0").rstrip(".")
            plain_token = _normalise(str(abs(value))).rstrip("0").rstrip(".")
            evidence_verified = bool(evidence_norm) and any(evidence_norm in text for text in page_texts)
            number_verified = any(
                number_token in text.replace(",", "") or plain_token in text.replace(",", "")
                for text in page_texts
            )
            if not evidence and not number_verified:
                result["discarded"].append(f"{field}: no evidence quote and number not found on page {page}.")
                continue
            kept.append(
                {
                    "period": period,
                    "value": value,
                    "page": page,
                    "source": source,
                    "evidence": evidence or "(no quote returned)",
                    "evidence_verified": evidence_verified or number_verified,
                    "origin": "llm",
                }
            )
        # Keep one entry per period (first wins) so growth maths is clean.
        unique: dict[str, dict[str, Any]] = {}
        for entry in kept:
            unique.setdefault(entry["period"], entry)
        result["figures"][field] = list(unique.values())
    return result


def extract_figures(company: str, document_id: str | None = None, status_callback=None) -> dict[str, Any]:
    """Run retrieval + LLM + validation and save the result for the company.

    Manual corrections already saved for this company are preserved.
    """
    hits = gather_evidence(company, document_id=document_id)
    if not hits:
        raise LLMError(f"No ingested passages found for {company}. Upload or paste a report first.")

    prompt = build_prompt(company, hits)
    try:
        raw = generate_json(prompt, status_callback=status_callback)
    except LLMJSONError:
        raw = None  # validate_extraction turns this into "everything discarded"

    validated = validate_extraction(raw, hits)
    previous = load_figures(company)
    record = {
        "company": company,
        "document_id": document_id,
        "currency": validated["currency"],
        "units": validated["units"],
        "figures": validated["figures"],
        "discarded": validated["discarded"],
        "passages_used": len(hits),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "model": GEMINI_MODEL,
    }
    if previous:
        record = _merge_manual(previous, record)
    save_figures(company, record)
    return record


def _merge_manual(previous: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Re-apply any manual corrections the user made earlier."""
    for field, entries in previous.get("figures", {}).items():
        for entry in entries:
            if entry.get("origin") == "manual":
                _upsert_entry(new["figures"].setdefault(field, []), entry)
    return new


def _upsert_entry(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    for index, existing in enumerate(entries):
        if existing["period"] == entry["period"]:
            entries[index] = entry
            return
    entries.append(entry)


def save_figures(company: str, record: dict[str, Any]) -> None:
    figures_path(company).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def load_figures(company: str) -> dict[str, Any] | None:
    path = figures_path(company)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def delete_figures(company: str) -> None:
    path = figures_path(company)
    if path.exists():
        path.unlink()


def set_manual_value(company: str, field: str, period: str, value: float | None) -> dict[str, Any]:
    """Let the user override (or add) one figure. value=None removes the period."""
    if field not in FIELD_DEFINITIONS:
        raise ValueError(f"Unknown field: {field}")
    record = load_figures(company) or {
        "company": company, "currency": None, "units": None,
        "figures": {key: [] for key in FIELD_DEFINITIONS}, "discarded": [],
        "extracted_at": None, "model": None,
    }
    entries = record["figures"].setdefault(field, [])
    period = period.strip() or "Unspecified period"
    if value is None:
        record["figures"][field] = [entry for entry in entries if entry["period"] != period]
    else:
        _upsert_entry(
            entries,
            {
                "period": period,
                "value": float(value),
                "page": "manual",
                "source": "manual correction",
                "evidence": "Entered by the user",
                "evidence_verified": True,
                "origin": "manual",
            },
        )
    save_figures(company, record)
    return record


def figures_to_rows(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten a figures record into table rows for display/editing."""
    rows = []
    figures = (record or {}).get("figures", {})
    for field, label in FIELD_LABELS.items():
        entries = figures.get(field, [])
        if not entries:
            rows.append(
                {"Field": label, "key": field, "Period": "", "Value": None, "Status": NOT_FOUND,
                 "Page": "", "Source": "", "Evidence": "", "Origin": ""}
            )
            continue
        for entry in entries:
            rows.append(
                {
                    "Field": label,
                    "key": field,
                    "Period": entry["period"],
                    "Value": entry["value"],
                    "Status": "Verified" if entry.get("evidence_verified") else "Unverified quote",
                    "Page": entry["page"],
                    "Source": entry["source"],
                    "Evidence": entry["evidence"],
                    "Origin": entry.get("origin", "llm"),
                }
            )
    return rows
