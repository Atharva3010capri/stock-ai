"""Knowledge graph validation, persistence, queries and evidence enforcement."""

import knowledge_graph as kg
from graph_extraction import merge_extraction


def build_sample():
    g = kg.new_graph("Reliance Industries")
    company = kg.add_entity(g, "Reliance Industries", "Company")
    retail = kg.add_entity(g, "Reliance Retail", "Subsidiary")
    oil = kg.add_entity(g, "Crude oil prices", "Financial Event")
    assert kg.add_relationship(g, company, retail, "OWNS", "ar.pdf", 45, "owns Reliance Retail", 0.95, "confirmed")
    assert kg.add_relationship(g, company, oil, "AFFECTED_BY", "ar.pdf", 60, "exposed to crude oil prices", 0.7, "inferred")
    return g


def test_invalid_entities_and_relationships_rejected():
    g = kg.new_graph("X")
    assert kg.add_entity(g, "", "Company") is None
    assert kg.add_entity(g, "Thing", "Alien") is None
    a = kg.add_entity(g, "A", "Company")
    b = kg.add_entity(g, "B", "Company")
    assert not kg.add_relationship(g, a, b, "LOVES", "f.pdf", 1, "ev", 0.5, "confirmed")       # bad type
    assert not kg.add_relationship(g, a, b, "OWNS", "f.pdf", 1, "", 0.5, "confirmed")          # no evidence
    assert not kg.add_relationship(g, a, b, "OWNS", "f.pdf", 1, "ev", 0.5, "maybe")           # bad status
    assert not kg.add_relationship(g, a, a, "OWNS", "f.pdf", 1, "ev", 0.5, "confirmed")       # self loop
    assert kg.add_relationship(g, a, b, "OWNS", "f.pdf", 1, "ev", 0.5, "confirmed")
    assert not kg.add_relationship(g, a, b, "OWNS", "f.pdf", 1, "ev again", 0.9, "confirmed")  # duplicate


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "KNOWLEDGE_BASE_DIR", tmp_path)
    g = build_sample()
    kg.save_graph(g, "Reliance Industries")
    loaded = kg.load_graph("Reliance Industries")
    assert kg.statistics(loaded) == kg.statistics(g)
    assert kg.statistics(loaded)["confirmed"] == 1 and kg.statistics(loaded)["inferred"] == 1


def test_queries():
    g = build_sample()
    assert [r["target_name"] for r in kg.get_subsidiaries(g, "Reliance Industries")] == ["Reliance Retail"]
    assert [r["target_name"] for r in kg.get_risk_factors(g, "Reliance Industries")] == ["Crude oil prices"]
    assert kg.get_competitors(g, "Reliance Industries") == []
    assert len(kg.get_connected(g, "retail")) == 1
    assert kg.search_relationships(g, "How can crude oil prices affect this company?")[0]["target_name"] == "Crude oil prices"


def test_merge_extraction_requires_evidence_in_passage():
    g = kg.new_graph("Acme")
    hits = [{"text": "Acme Limited owns 100% of Acme Retail Limited.", "metadata": {"source": "ar.pdf", "page": 7}}]
    raw = {
        "entities": [{"name": "Acme Limited", "type": "Company"}, {"name": "Acme Retail Limited", "type": "Subsidiary"}],
        "relationships": [
            {"source": "Acme Limited", "target": "Acme Retail Limited", "relationship": "OWNS",
             "evidence": "owns 100% of Acme Retail Limited", "source_file": "ar.pdf", "page": 7,
             "confidence": 0.9, "status": "confirmed"},
            {"source": "Acme Limited", "target": "Globex", "relationship": "COMPETES_WITH", "target_type": "Company",
             "evidence": "competes fiercely with Globex", "source_file": "ar.pdf", "page": 7,
             "confidence": 0.9, "status": "confirmed"},   # quote not in passage -> rejected
            {"source": "Acme Limited", "target": "Acme Retail Limited", "relationship": "OWNS",
             "evidence": "owns 100%", "source_file": "ar.pdf", "page": 99,
             "confidence": 0.9, "status": "confirmed"},   # page not shown -> rejected
        ],
    }
    counts = merge_extraction(g, raw, hits, "Acme")
    assert counts["relationships_added"] == 1
    assert counts["rejected"] == 2
    assert "company:globex" not in g  # orphan never added
    assert kg.statistics(g)["relationships"] == 1
