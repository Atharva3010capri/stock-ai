"""Knowledge graph per company: NetworkX in memory, JSON on disk.

Nodes  = entities (Company, Industry, Sector, Subsidiary, Product, Executive,
         Financial Event) with a name and type.
Edges  = relationships with provenance: source file, page, evidence quote,
         confidence (0-1) and status ("confirmed" or "inferred").
"""

from __future__ import annotations

import json
import re
from typing import Any

import networkx as nx

from config import KNOWLEDGE_BASE_DIR

ENTITY_TYPES = ("Company", "Industry", "Sector", "Subsidiary", "Product", "Executive", "Financial Event")
RELATIONSHIP_TYPES = (
    "OPERATES_IN", "COMPETES_WITH", "SUPPLIES_TO", "SELLS_TO", "OWNS",
    "SUBSIDIARY_OF", "PRODUCES", "AFFECTED_BY",
)
STATUSES = ("confirmed", "inferred")

ENTITY_COLORS = {
    "Company": "#1f77b4", "Subsidiary": "#17becf", "Industry": "#2ca02c", "Sector": "#98df8a",
    "Product": "#ff7f0e", "Executive": "#9467bd", "Financial Event": "#d62728",
}


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(text).strip()).strip("_").lower() or "x"


def graph_path(company: str):
    return KNOWLEDGE_BASE_DIR / f"graph_{_slug(company)}.json"


def entity_id(name: str, entity_type: str) -> str:
    """Stable node id from type + lowercase name."""
    return f"{_slug(entity_type)}:{_slug(name)}"


def new_graph(company: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(company=company)
    return graph


def add_entity(graph: nx.MultiDiGraph, name: str, entity_type: str) -> str | None:
    """Add a node (idempotent). Returns the node id or None if invalid."""
    name = (name or "").strip()
    if not name or entity_type not in ENTITY_TYPES:
        return None
    node_id = entity_id(name, entity_type)
    if node_id not in graph:
        graph.add_node(node_id, name=name, entity_type=entity_type)
    return node_id


def add_relationship(
    graph: nx.MultiDiGraph,
    source_id: str,
    target_id: str,
    relationship: str,
    source_file: str,
    page: Any,
    evidence: str,
    confidence: float,
    status: str,
) -> bool:
    """Add an edge with provenance. Returns False if invalid or duplicate."""
    if source_id not in graph or target_id not in graph or source_id == target_id:
        return False
    if relationship not in RELATIONSHIP_TYPES or status not in STATUSES:
        return False
    evidence = (evidence or "").strip()
    if not evidence or not source_file:
        return False
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        return False
    page = str(page)
    existing = graph.get_edge_data(source_id, target_id) or {}
    for data in existing.values():
        if data.get("relationship") == relationship and data.get("source_file") == source_file and data.get("page") == page:
            return False  # same fact from the same page already stored
    graph.add_edge(
        source_id, target_id,
        relationship=relationship, source_file=source_file, page=page,
        evidence=evidence, confidence=round(confidence, 2), status=status,
    )
    return True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_graph(graph: nx.MultiDiGraph, company: str) -> str:
    data = {
        "company": company,
        "entities": [{"id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)],
        "relationships": [
            {"source": source, "target": target, **attrs}
            for source, target, attrs in graph.edges(data=True)
        ],
    }
    path = graph_path(company)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_graph(company: str) -> nx.MultiDiGraph:
    """Load the saved graph for a company, or an empty one."""
    graph = new_graph(company)
    path = graph_path(company)
    if not path.exists():
        return graph
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return graph
    for entity in data.get("entities", []):
        node_id = entity.get("id")
        if node_id:
            graph.add_node(node_id, name=entity.get("name", node_id), entity_type=entity.get("entity_type", "Company"))
    for rel in data.get("relationships", []):
        add_relationship(
            graph, rel.get("source"), rel.get("target"), rel.get("relationship"),
            rel.get("source_file"), rel.get("page"), rel.get("evidence"),
            rel.get("confidence", 0), rel.get("status", "inferred"),
        )
    return graph


def delete_graph(company: str) -> None:
    path = graph_path(company)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def relationships(graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """All edges as flat dicts with readable names."""
    rows = []
    for source, target, attrs in graph.edges(data=True):
        rows.append(
            {
                "source_id": source,
                "source_name": graph.nodes[source].get("name", source),
                "source_type": graph.nodes[source].get("entity_type", ""),
                "relationship": attrs.get("relationship"),
                "target_id": target,
                "target_name": graph.nodes[target].get("name", target),
                "target_type": graph.nodes[target].get("entity_type", ""),
                "status": attrs.get("status"),
                "confidence": attrs.get("confidence"),
                "source_file": attrs.get("source_file"),
                "page": attrs.get("page"),
                "evidence": attrs.get("evidence"),
            }
        )
    return rows


def find_company_node(graph: nx.MultiDiGraph, company: str) -> str | None:
    """The node that represents the workspace company.

    The user may type "Reliance Industries" while the report says "Reliance
    Industries Limited", so we score name overlap and prefer the node with the
    most relationships.
    """
    target = _slug(company)
    best, best_score = None, (-1, -1)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("entity_type") != "Company":
            continue
        name = _slug(attrs.get("name", ""))
        if name == target:
            match = 3
        elif target in name or name in target:
            match = 2
        else:
            match = 0
        score = (match, graph.degree(node))
        if score > best_score:
            best, best_score = node, score
    return best


def _related(graph: nx.MultiDiGraph, node: str | None, rel_types: tuple[str, ...], direction: str) -> list[dict[str, Any]]:
    if node is None:
        return []
    rows = []
    for row in relationships(graph):
        if row["relationship"] not in rel_types:
            continue
        if direction in ("out", "both") and row["source_id"] == node:
            rows.append(row)
        elif direction in ("in", "both") and row["target_id"] == node:
            rows.append(row)
    return rows


def get_subsidiaries(graph: nx.MultiDiGraph, company: str) -> list[dict[str, Any]]:
    node = find_company_node(graph, company)
    rows = _related(graph, node, ("OWNS",), "out") + _related(graph, node, ("SUBSIDIARY_OF",), "in")
    return rows


def get_competitors(graph: nx.MultiDiGraph, company: str) -> list[dict[str, Any]]:
    node = find_company_node(graph, company)
    return _related(graph, node, ("COMPETES_WITH",), "both")


def get_segments_and_products(graph: nx.MultiDiGraph, company: str) -> list[dict[str, Any]]:
    node = find_company_node(graph, company)
    return _related(graph, node, ("OPERATES_IN", "PRODUCES"), "out")


def get_risk_factors(graph: nx.MultiDiGraph, company: str) -> list[dict[str, Any]]:
    node = find_company_node(graph, company)
    return _related(graph, node, ("AFFECTED_BY",), "out")


def get_connected(graph: nx.MultiDiGraph, entity_name: str) -> list[dict[str, Any]]:
    """Every relationship touching an entity whose name contains the text."""
    needle = _slug(entity_name)
    matches = {n for n, a in graph.nodes(data=True) if needle in _slug(a.get("name", ""))}
    return [row for row in relationships(graph) if row["source_id"] in matches or row["target_id"] in matches]


def search_relationships(graph: nx.MultiDiGraph, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Keyword match of a question against entity names, relationship types and evidence."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    stop = {"the", "and", "for", "with", "what", "how", "does", "this", "that", "company", "its", "can", "are", "about"}
    words -= stop
    if not words:
        return []
    scored = []
    for row in relationships(graph):
        name_text = f"{row['source_name']} {row['target_name']} {row['relationship'].replace('_', ' ')}".lower()
        evidence_text = (row["evidence"] or "").lower()
        score = sum(3 for w in words if w in name_text) + sum(1 for w in words if w in evidence_text)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["source_name"]))
    return [row for _, row in scored[:limit]]


def statistics(graph: nx.MultiDiGraph) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        by_type[attrs.get("entity_type", "?")] = by_type.get(attrs.get("entity_type", "?"), 0) + 1
    confirmed = sum(1 for _, _, a in graph.edges(data=True) if a.get("status") == "confirmed")
    return {
        "entities": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "confirmed": confirmed,
        "inferred": graph.number_of_edges() - confirmed,
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# Visualisation (Plotly figure for Streamlit)
# ---------------------------------------------------------------------------

def to_plotly_figure(graph: nx.MultiDiGraph, title: str = "Knowledge graph"):
    """Draw the graph: solid edges = confirmed, dashed = inferred."""
    import plotly.graph_objects as go

    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Knowledge graph is empty", xaxis_visible=False, yaxis_visible=False)
        return fig

    positions = nx.spring_layout(graph, seed=42, k=1.2 / max(1, graph.number_of_nodes() ** 0.5))
    traces = []
    for status, dash in (("confirmed", "solid"), ("inferred", "dash")):
        xs, ys = [], []
        for source, target, attrs in graph.edges(data=True):
            if attrs.get("status") != status:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        if xs:
            traces.append(
                go.Scatter(x=xs, y=ys, mode="lines", name=f"{status} relationship",
                           line=dict(width=1.5, dash=dash, color="#888"), hoverinfo="none")
            )

    # Edge labels at the midpoint
    label_x, label_y, label_text = [], [], []
    for source, target, attrs in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        label_x.append((x0 + x1) / 2)
        label_y.append((y0 + y1) / 2)
        label_text.append(
            f"{graph.nodes[source]['name']} {attrs.get('relationship')} {graph.nodes[target]['name']}<br>"
            f"{attrs.get('status')} (confidence {attrs.get('confidence')})<br>"
            f"{attrs.get('source_file')}, page {attrs.get('page')}<br>"
            f"<i>{(attrs.get('evidence') or '')[:160]}</i>"
        )
    traces.append(
        go.Scatter(x=label_x, y=label_y, mode="markers", marker=dict(size=6, color="#bbb", symbol="diamond"),
                   hovertext=label_text, hoverinfo="text", name="relationship details", showlegend=False)
    )

    for entity_type in ENTITY_TYPES:
        nodes = [n for n, a in graph.nodes(data=True) if a.get("entity_type") == entity_type]
        if not nodes:
            continue
        traces.append(
            go.Scatter(
                x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
                mode="markers+text", text=[graph.nodes[n]["name"] for n in nodes], textposition="top center",
                marker=dict(size=[10 + 3 * graph.degree(n) for n in nodes], color=ENTITY_COLORS.get(entity_type, "#333")),
                hovertext=[f"{graph.nodes[n]['name']} ({entity_type})" for n in nodes], hoverinfo="text",
                name=entity_type,
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, showlegend=True, hovermode="closest", height=600,
        xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
