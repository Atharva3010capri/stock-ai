"""Build the knowledge graph from report chunks with the LLM.

For each batch of passages Gemini returns entities and relationships as
JSON with provenance. Python then enforces:
  - allowed entity and relationship types only,
  - the evidence quote must actually appear in the cited passage,
  - confidence in [0, 1] and status in {confirmed, inferred}.
Anything else is dropped. The result is saved per company as JSON.
"""

from __future__ import annotations

import re
from typing import Any

import knowledge_graph as kg
from gemini_client import LLMError, LLMJSONError, generate_json
from rag import retrieve_multi

GRAPH_QUERIES = [
    "subsidiaries and joint ventures of the company",
    "business segments and industries the company operates in",
    "main products and services offered",
    "competitors and competitive landscape",
    "key customers and markets the company sells to",
    "suppliers and raw material sourcing",
    "key risks: commodity prices, regulation, currency, demand",
    "board of directors managing director chief executive officer chief financial officer",
    "acquisitions, mergers and major events during the year",
]

GRAPH_INSTRUCTIONS = """You are an information-extraction system for financial reports.
From the numbered passages, extract entities and relationships about the company.

Entity types (use exactly): Company, Industry, Sector, Subsidiary, Product, Executive, Financial Event
Relationship types (use exactly): OPERATES_IN, COMPETES_WITH, SUPPLIES_TO, SELLS_TO, OWNS, SUBSIDIARY_OF, PRODUCES, AFFECTED_BY

Rules:
- Only extract what the passages state. No outside knowledge.
- Every relationship needs "evidence": an exact quote (under 200 characters) copied from ONE passage,
  plus that passage's "source" filename and "page" exactly as labelled.
- "status" is "confirmed" when the passage states the relationship directly,
  "inferred" when it is a reasonable reading but not explicit.
- "confidence" is a number from 0 to 1.
- Use AFFECTED_BY for risk factors (e.g. Company AFFECTED_BY "Crude oil prices" as a Financial Event).
- Skip generic entities ("the Company's customers") and anything without evidence.
Return JSON only:
{
  "entities": [{"name": "Reliance Retail Ventures Limited", "type": "Subsidiary"}],
  "relationships": [
    {"source": "Reliance Industries Limited", "source_type": "Company",
     "target": "Reliance Retail Ventures Limited", "target_type": "Subsidiary",
     "relationship": "OWNS", "evidence": "exact quote", "source_file": "report.pdf", "page": 45,
     "confidence": 0.95, "status": "confirmed"}
  ]
}"""


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def gather_passages(company: str, max_chunks: int = 18, document_id: str | None = None) -> list[dict[str, Any]]:
    return retrieve_multi(GRAPH_QUERIES, company, per_query=3, max_total=max_chunks, document_id=document_id)


def build_prompt(company: str, hits: list[dict[str, Any]]) -> str:
    parts = [f"COMPANY OF INTEREST: {company}", "", "PASSAGES:"]
    for number, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        parts.append(f"[Passage {number}] source_file: {meta.get('source')} | page: {meta.get('page')}\n{hit['text']}")
    parts.append("")
    parts.append(GRAPH_INSTRUCTIONS)
    return "\n".join(parts)


def merge_extraction(
    graph, raw: Any, hits: list[dict[str, Any]], company: str
) -> dict[str, int]:
    """Validate one LLM response and add the good parts to the graph."""
    counts = {"entities_added": 0, "relationships_added": 0, "rejected": 0}
    if not isinstance(raw, dict):
        counts["rejected"] += 1
        return counts

    passages: dict[tuple[str, str], list[str]] = {}
    for hit in hits:
        meta = hit["metadata"]
        passages.setdefault((str(meta.get("source")), str(meta.get("page"))), []).append(_normalise(hit["text"]))

    declared_types: dict[str, str] = {}
    for entity in raw.get("entities", []) if isinstance(raw.get("entities"), list) else []:
        if isinstance(entity, dict):
            name, etype = str(entity.get("name", "")).strip(), str(entity.get("type", "")).strip()
            if etype in kg.ENTITY_TYPES and name:
                declared_types[name.lower()] = etype

    for rel in raw.get("relationships", []) if isinstance(raw.get("relationships"), list) else []:
        if not isinstance(rel, dict):
            counts["rejected"] += 1
            continue
        src_name = str(rel.get("source", "")).strip()
        tgt_name = str(rel.get("target", "")).strip()
        src_type = str(rel.get("source_type") or declared_types.get(src_name.lower(), "")).strip()
        tgt_type = str(rel.get("target_type") or declared_types.get(tgt_name.lower(), "")).strip()
        rel_type = str(rel.get("relationship", "")).strip().upper()
        source_file = str(rel.get("source_file", "")).strip()
        page = str(rel.get("page", "")).strip()
        evidence = str(rel.get("evidence", "")).strip()
        status = str(rel.get("status", "")).strip().lower()

        if (source_file, page) not in passages:
            counts["rejected"] += 1
            continue
        if not evidence or not any(_normalise(evidence) in text for text in passages[(source_file, page)]):
            counts["rejected"] += 1
            continue

        before = graph.number_of_nodes()
        src_id = kg.add_entity(graph, src_name, src_type)
        tgt_id = kg.add_entity(graph, tgt_name, tgt_type)
        counts["entities_added"] += graph.number_of_nodes() - before
        if not src_id or not tgt_id:
            counts["rejected"] += 1
            continue
        added = kg.add_relationship(
            graph, src_id, tgt_id, rel_type, source_file, page, evidence,
            rel.get("confidence", 0.5), status,
        )
        if added:
            counts["relationships_added"] += 1
        else:
            counts["rejected"] += 1

    # Remove entities left without any relationship.
    orphans = [n for n in graph.nodes if graph.degree(n) == 0]
    graph.remove_nodes_from(orphans)
    return counts


def build_graph(
    company: str,
    document_id: str | None = None,
    batch_size: int = 6,
    status_callback=None,
    reset: bool = False,
) -> dict[str, Any]:
    """Extract the graph for a company and save it. Returns a summary."""
    hits = gather_passages(company, document_id=document_id)
    if not hits:
        raise LLMError(f"No ingested passages found for {company}. Upload or paste a report first.")

    graph = kg.new_graph(company) if reset else kg.load_graph(company)
    totals = {"entities_added": 0, "relationships_added": 0, "rejected": 0, "batches": 0, "failed_batches": 0}
    batches = [hits[i : i + batch_size] for i in range(0, len(hits), batch_size)]
    for index, batch in enumerate(batches, start=1):
        if status_callback:
            status_callback(f"Extracting entities from passage batch {index}/{len(batches)}...")
        try:
            raw = generate_json(build_prompt(company, batch), status_callback=status_callback)
        except LLMJSONError:
            totals["failed_batches"] += 1
            continue
        counts = merge_extraction(graph, raw, batch, company)
        for key in ("entities_added", "relationships_added", "rejected"):
            totals[key] += counts[key]
        totals["batches"] += 1

    path = kg.save_graph(graph, company)
    return {**totals, "saved_path": path, "statistics": kg.statistics(graph)}
