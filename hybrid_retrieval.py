"""Hybrid chatbot: RAG passages + knowledge graph + computed ratios -> one answer.

Steps for every question:
  1. resolve the company (explicit selection, or name match in the question)
  2. RAG retrieval over that company's reports
  3. graph lookup for related entities/relationships
  4. include ratios computed from extracted figures
  5. build one context, add chat history, call Gemini
"""

from __future__ import annotations

from typing import Any

import knowledge_graph as kg
from config import DISCLAIMER, TOP_K
from extraction import FIELD_LABELS, load_figures
from fundamentals import compute_ratios, format_ratio_value
from gemini_client import LLMError, generate_text
from ingest import list_companies
from rag import SYSTEM_PROMPT, build_context, retrieve, sources_from_hits
from vector_store import VectorStoreError

MAX_HISTORY_TURNS = 6


def resolve_company(question: str, selected: str | None) -> str | None:
    """Prefer the workspace selection; otherwise look for a known company name."""
    if selected:
        return selected
    lowered = question.lower()
    for info in list_companies():
        name = info["company"].lower()
        ticker = (info.get("ticker") or "").lower()
        if name in lowered or (ticker and f" {ticker} " in f" {lowered} "):
            return info["company"]
    return None


def figures_context(company: str) -> str:
    """Extracted figures and computed ratios as plain text (or a note if absent)."""
    record = load_figures(company)
    if not record:
        return "No figures have been extracted for this company yet."
    units = " ".join(part for part in (record.get("currency"), record.get("units")) if part) or "units not stated"
    lines = [f"Units: {units}"]
    for field, entries in record.get("figures", {}).items():
        if not entries:
            lines.append(f"- {FIELD_LABELS.get(field, field)}: Not found in report")
            continue
        for entry in entries:
            lines.append(
                f"- {FIELD_LABELS.get(field, field)} ({entry['period']}): {entry['value']:,} "
                f"[{entry['source']}, page {entry['page']}]"
            )
    lines.append("Computed ratios (Python, from the figures above):")
    for row in compute_ratios(record.get("figures", {})):
        if row["status"] == "ok":
            lines.append(f"- {row['name']} = {format_ratio_value(row, record.get('units'), record.get('currency'))} "
                         f"(formula: {row['formula']}; period: {row['period']})")
        else:
            lines.append(f"- {row['name']}: skipped ({row['reason']})")
    return "\n".join(lines)


def graph_context(company: str, question: str) -> tuple[str, list[dict[str, Any]]]:
    """Relevant graph relationships as text plus the raw rows used."""
    graph = kg.load_graph(company)
    if graph.number_of_edges() == 0:
        return "Knowledge graph has not been built for this company yet.", []
    rows = kg.search_relationships(graph, question, limit=8)
    if not rows:
        rows = kg.get_segments_and_products(graph, company)[:4] + kg.get_risk_factors(graph, company)[:4]
    if not rows:
        return "No graph relationships matched the question.", []
    lines = []
    for row in rows:
        lines.append(
            f"- {row['source_name']} --{row['relationship']}--> {row['target_name']} "
            f"[{row['status']}, confidence {row['confidence']}] "
            f"evidence: \"{row['evidence']}\" ({row['source_file']}, page {row['page']})"
        )
    return "\n".join(lines), rows


def history_text(history: list[dict[str, str]]) -> str:
    recent = history[-MAX_HISTORY_TURNS * 2 :]
    if not recent:
        return "(no previous messages)"
    return "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in recent)


def chat(
    question: str,
    selected_company: str | None,
    history: list[dict[str, str]] | None = None,
    top_k: int = TOP_K,
    status_callback=None,
) -> dict[str, Any]:
    """Answer one chat turn. history = [{"role": "user"|"assistant", "content": str}]."""
    history = history or []
    question = (question or "").strip()
    if not question:
        return {"status": "error", "answer": "Please type a question.", "sources": [], "graph_rows": []}

    company = resolve_company(question, selected_company)
    if not company:
        known = ", ".join(info["company"] for info in list_companies()) or "none yet"
        return {
            "status": "error",
            "answer": f"I could not tell which company you mean. Select one in the sidebar. Known companies: {known}.",
            "sources": [], "graph_rows": [],
        }

    # Follow-up questions ("what about its debt?") are short; add the previous
    # user question to the retrieval query so the search has context.
    retrieval_query = question
    previous_user = [turn["content"] for turn in history if turn["role"] == "user"]
    if len(question.split()) < 6 and previous_user:
        retrieval_query = f"{previous_user[-1]} {question}"

    try:
        hits = retrieve(retrieval_query, company, top_k=top_k)
    except VectorStoreError as error:
        return {"status": "error", "answer": str(error), "sources": [], "graph_rows": []}

    graph_text, graph_rows = graph_context(company, retrieval_query)
    figures_text = figures_context(company)

    if not hits and not graph_rows:
        return {
            "status": "no_evidence",
            "answer": (
                f"The reports ingested for {company} do not contain passages relevant to that "
                "question, and the knowledge graph has nothing matching either. I will not guess."
            ),
            "sources": [], "graph_rows": [],
        }

    prompt = (
        f"COMPANY: {company}\n\n"
        f"CONVERSATION SO FAR:\n{history_text(history)}\n\n"
        f"REPORT PASSAGES (primary evidence):\n{build_context(hits) if hits else '(no relevant passages found)'}\n\n"
        f"KNOWLEDGE GRAPH RELATIONSHIPS (context only; a graph link is never proof of financial performance):\n{graph_text}\n\n"
        f"EXTRACTED FIGURES AND COMPUTED RATIOS:\n{figures_text}\n\n"
        f"NEW QUESTION: {question}\n\n"
        "Answer in this structure:\n"
        "1. What the reports state (with (filename, page N) citations).\n"
        "2. Relevant connections from the knowledge graph, each labelled confirmed or inferred.\n"
        "3. What the reports do NOT say about this question.\n"
        "Keep it concise and beginner-friendly."
    )
    try:
        answer = generate_text(prompt, system_instruction=SYSTEM_PROMPT, status_callback=status_callback)
    except LLMError as error:
        return {"status": "error", "answer": str(error), "sources": sources_from_hits(hits), "graph_rows": graph_rows}

    if DISCLAIMER.lower() not in answer.lower():
        answer = f"{answer}\n\n_{DISCLAIMER}_"
    return {"status": "ok", "company": company, "answer": answer, "sources": sources_from_hits(hits), "graph_rows": graph_rows}
