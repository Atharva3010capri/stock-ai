"""Company analysis report: Markdown built in Python, narrative parts by Gemini.

Sections: Overview, Key Financials, Ratio Analysis, Trends, Business Segments,
Graph Insights, Key Findings, Risks, Limitations, Conclusion.
The conclusion carries only a fundamental label: strong / weak / mixed /
insufficient information. Never a buy/sell call.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import knowledge_graph as kg
from config import DISCLAIMER, GEMINI_MODEL, REPORTS_DIR
from extraction import FIELD_LABELS, NOT_FOUND, load_figures
from fundamentals import compute_ratios, format_ratio_value, period_sort_key, trend_table
from gemini_client import LLMJSONError, generate_json
from ingest import list_documents
from rag import SYSTEM_PROMPT, build_context, retrieve_multi

LABELS = ("strong", "weak", "mixed", "insufficient information")

NARRATIVE_QUERIES = [
    "company overview business description what the company does",
    "management discussion and analysis performance highlights",
    "business segments revenue by segment",
    "principal risks and uncertainties risk factors",
    "outlook and strategy going forward",
]

NARRATIVE_INSTRUCTIONS = """Write the narrative parts of a company analysis report from the evidence above.
Return JSON only:
{
  "overview": "2-4 sentences describing the business, each with a (filename, page N) citation",
  "segments": "bullet-style text on business segments from the passages, with citations; or 'Not described in the ingested reports.'",
  "key_findings": ["finding with citation", "..."],
  "risks": ["risk taken from the report's own risk disclosures, with citation", "..."],
  "conclusion_label": "strong" | "weak" | "mixed" | "insufficient information",
  "conclusion": "3-5 sentences: the evidence supporting the label, and what the evidence does NOT establish",
  "not_covered": ["important things the reports do not say"]
}
Rules: only use the passages, figures and ratios given. Cite filename and page for every factual claim.
No share price, no buy/sell language, no outside knowledge."""


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text.strip()).strip("_") or "company"


def _figures_table(record: dict[str, Any] | None) -> str:
    if not record:
        return f"_No figures extracted yet. Every field is '{NOT_FOUND}'._"
    units = " ".join(part for part in (record.get("currency"), record.get("units")) if part) or "units not stated"
    lines = [f"Units: **{units}**", "", "| Figure | Period | Value | Source | Page | Verified |", "|---|---|---:|---|---|---|"]
    for field, label in FIELD_LABELS.items():
        entries = record.get("figures", {}).get(field, [])
        if not entries:
            lines.append(f"| {label} | - | {NOT_FOUND} | - | - | - |")
            continue
        for entry in sorted(entries, key=lambda e: period_sort_key(e["period"]), reverse=True):
            lines.append(
                f"| {label} | {entry['period']} | {entry['value']:,} | {entry['source']} | {entry['page']} | "
                f"{'yes' if entry.get('evidence_verified') else 'no'} |"
            )
    return "\n".join(lines)


def _ratio_table(record: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    if not record:
        return "_Ratios need extracted figures first._", []
    rows = compute_ratios(record.get("figures", {}))
    lines = ["| Ratio | Value | Formula | Inputs | Period | Meaning |", "|---|---:|---|---|---|---|"]
    for row in rows:
        inputs = ", ".join(f"{k}={v:,}" if isinstance(v, (int, float)) else f"{k}=missing" for k, v in row["inputs"].items())
        value = format_ratio_value(row, record.get("units"), record.get("currency")) if row["status"] == "ok" else f"Skipped ({row['reason']})"
        lines.append(f"| {row['name']} | {value} | `{row['formula']}` | {inputs} | {row['period']} | {row['explanation']} |")
    lines.append("")
    lines.append("_No universal good/bad thresholds are applied; interpret against the company's sector._")
    return "\n".join(lines), rows


def _trends_text(record: dict[str, Any] | None) -> str:
    if not record:
        return "_No figures available for trends._"
    rows = trend_table(record.get("figures", {}))
    if not rows:
        return "_Not enough period data in the extracted figures to describe a trend._"
    by_field: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_field.setdefault(row["field"], []).append(row)
    lines = []
    for field, items in by_field.items():
        series = " -> ".join(f"{item['period']}: {item['value']:,}" for item in items)
        if len(items) >= 2:
            first, last = items[0]["value"], items[-1]["value"]
            direction = "rose" if last > first else "fell" if last < first else "was flat"
            lines.append(f"- **{FIELD_LABELS.get(field, field)}** {direction} across the periods found: {series}")
        else:
            lines.append(f"- **{FIELD_LABELS.get(field, field)}**: only one period found ({series}); no trend can be stated")
    return "\n".join(lines)


def _graph_section(company: str) -> str:
    graph = kg.load_graph(company)
    stats = kg.statistics(graph)
    if stats["relationships"] == 0:
        return "_Knowledge graph has not been built for this company._"
    lines = [
        f"{stats['entities']} entities and {stats['relationships']} relationships "
        f"({stats['confirmed']} confirmed, {stats['inferred']} inferred).",
        "",
    ]
    for title, rows in (
        ("Subsidiaries / ownership", kg.get_subsidiaries(graph, company)),
        ("Segments and products", kg.get_segments_and_products(graph, company)),
        ("Competitors", kg.get_competitors(graph, company)),
        ("Factors the report says affect the company", kg.get_risk_factors(graph, company)),
    ):
        if rows:
            lines.append(f"**{title}**")
            for row in rows[:8]:
                lines.append(
                    f"- {row['source_name']} {row['relationship']} {row['target_name']} "
                    f"({row['status']}, {row['source_file']}, page {row['page']})"
                )
            lines.append("")
    lines.append("_Graph links describe relationships stated in the reports; they are not evidence of financial performance._")
    return "\n".join(lines)


def _narrative(company: str, record: dict[str, Any] | None, ratio_rows: list[dict[str, Any]], status_callback=None) -> dict[str, Any]:
    hits = retrieve_multi(NARRATIVE_QUERIES, company, per_query=4, max_total=16)
    graph = kg.load_graph(company)
    graph_lines = [
        f"- {r['source_name']} {r['relationship']} {r['target_name']} ({r['status']}; {r['source_file']}, page {r['page']})"
        for r in kg.relationships(graph)[:25]
    ]
    figures_lines = []
    if record:
        for field, entries in record.get("figures", {}).items():
            for entry in entries:
                figures_lines.append(f"- {FIELD_LABELS[field]} {entry['period']}: {entry['value']:,} ({entry['source']}, page {entry['page']})")
    ratio_lines = [
        f"- {r['name']}: {format_ratio_value(r, (record or {}).get('units'), (record or {}).get('currency'))} ({r['formula']})"
        for r in ratio_rows if r["status"] == "ok"
    ]
    prompt = (
        f"COMPANY: {company}\n\n"
        f"REPORT PASSAGES:\n{build_context(hits) if hits else '(none)'}\n\n"
        f"EXTRACTED FIGURES:\n" + ("\n".join(figures_lines) or "(none extracted)") + "\n\n"
        f"COMPUTED RATIOS:\n" + ("\n".join(ratio_lines) or "(none computed)") + "\n\n"
        f"KNOWLEDGE GRAPH:\n" + ("\n".join(graph_lines) or "(empty)") + "\n\n"
        + NARRATIVE_INSTRUCTIONS
    )
    fallback = {
        "overview": "_The model did not return usable narrative text._",
        "segments": "_Not available._", "key_findings": [], "risks": [],
        "conclusion_label": "insufficient information",
        "conclusion": "The narrative step failed, so no fundamental label can be supported.",
        "not_covered": [],
    }
    if not hits and not figures_lines:
        fallback["conclusion"] = "No report passages or figures are available for this company."
        return fallback
    try:
        raw = generate_json(prompt, system_instruction=SYSTEM_PROMPT, status_callback=status_callback)
    except LLMJSONError:
        return fallback
    if not isinstance(raw, dict):
        return fallback
    result = {**fallback, **{k: v for k, v in raw.items() if k in fallback}}
    label = str(result.get("conclusion_label", "")).strip().lower()
    result["conclusion_label"] = label if label in LABELS else "insufficient information"
    for key in ("key_findings", "risks", "not_covered"):
        if not isinstance(result[key], list):
            result[key] = [str(result[key])]
        result[key] = [str(item) for item in result[key]]
    return result


def generate_report(company: str, status_callback=None) -> dict[str, Any]:
    """Build the Markdown report, save it in reports/, return {markdown, path, label}."""
    record = load_figures(company)
    ratio_md, ratio_rows = _ratio_table(record)
    if status_callback:
        status_callback("Writing narrative sections with Gemini...")
    narrative = _narrative(company, record, ratio_rows, status_callback=status_callback)

    documents = list_documents(company)
    doc_lines = [
        f"- {d['source']} ({d['document_type']}, {d['reporting_period']}, {d['pages']} pages, ingested {d['ingested_at'][:10]})"
        for d in documents
    ] or ["- none"]
    periods = sorted({d["reporting_period"] for d in documents})

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None stated in the ingested reports."

    markdown = f"""# {company}: Company Analysis Report

_Generated by StockGraph AI on {datetime.now():%Y-%m-%d %H:%M} from the documents listed below. {DISCLAIMER}_

**Documents analysed**
{chr(10).join(doc_lines)}

**Reporting periods covered:** {", ".join(periods) if periods else "not specified"}

## 1. Overview
{narrative['overview']}

## 2. Key Financials
{_figures_table(record)}

## 3. Ratio Analysis
{ratio_md}

## 4. Trends
{_trends_text(record)}

## 5. Business Segments
{narrative['segments']}

## 6. Graph Insights
{_graph_section(company)}

## 7. Key Findings
{bullets(narrative['key_findings'])}

## 8. Risks (from the report's own disclosures)
{bullets(narrative['risks'])}

## 9. Limitations
- Analysis is limited to the ingested documents; nothing was fetched from markets or the web.
- Figures were extracted by an LLM and validated against page text; entries marked "Verified = no" could not be matched to a quote and should be checked by hand.
- Ratios are computed only where inputs exist; missing inputs are skipped, never estimated.
- Knowledge graph relationships marked "inferred" are the model's reading, not explicit statements.
- What the reports do not cover:
{chr(10).join(f"  - {item}" for item in narrative['not_covered']) or "  - (none listed)"}

## 10. Conclusion
**Fundamental assessment: {narrative['conclusion_label'].upper()}**

{narrative['conclusion']}

_{DISCLAIMER} No price targets or buy/sell recommendations are given._
"""
    path = REPORTS_DIR / f"{_slug(company)}_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(markdown, encoding="utf-8")
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps({"company": company, "model": GEMINI_MODEL, "narrative": narrative}, indent=2), encoding="utf-8")
    return {"markdown": markdown, "path": str(path), "label": narrative["conclusion_label"]}
