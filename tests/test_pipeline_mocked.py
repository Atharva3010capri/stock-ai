"""End-to-end glue test: extraction -> ratios -> graph -> chat -> report, with the
vector store and Gemini replaced by fakes. Checks that the modules fit together
and that nothing is invented when the model output is wrong."""

import json
from pathlib import Path

import pytest

import extraction
import graph_extraction
import hybrid_retrieval
import ingest
import knowledge_graph as kg
import report_generator
import vector_store

PASSAGE = (
    "Revenue from operations for FY2024 was 4,250.0 crore compared with 3,700.0 crore in FY2023. "
    "Net profit for the year (profit after tax) was 425.0 crore in FY2024 compared with 352.0 crore in FY2023. "
    "Nimbus Foods Limited owns 100% of Nimbus Beverages Private Limited. "
    "The Company's margins are affected by palm oil prices."
)
META = {"source": "nimbus.txt", "page": "pasted", "company": "Nimbus", "chunk_id": "c0",
        "reporting_period": "FY2024", "document_type": "Annual Report"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "KNOWLEDGE_BASE_DIR", tmp_path)
    monkeypatch.setattr(kg, "KNOWLEDGE_BASE_DIR", tmp_path)
    monkeypatch.setattr(report_generator, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "DOCUMENT_REGISTRY_FILE", tmp_path / "documents.json")
    ingest.save_registry([{"document_id": "d1", "hash": "h", "source": "nimbus.txt", "company": "Nimbus",
                           "ticker": "NIM", "document_type": "Annual Report", "reporting_period": "FY2024",
                           "pages": 1, "chunks": 1, "ingested_at": "2026-11-03T10:00:00"}])
    monkeypatch.setattr(vector_store, "query", lambda *a, **k: [{"text": PASSAGE, "metadata": META, "similarity": 0.6}])
    return tmp_path


def test_extraction_to_report(world, monkeypatch):
    # 1. extraction: good values plus one with an unknown page (must be dropped)
    fake_figures = {
        "currency": "INR", "units": "crore",
        "figures": {
            "revenue": [
                {"period": "FY2024", "value": 4250.0, "page": "pasted", "source": "nimbus.txt", "evidence": "Revenue from operations for FY2024 was 4,250.0 crore"},
                {"period": "FY2023", "value": 3700.0, "page": "pasted", "source": "nimbus.txt", "evidence": "3,700.0 crore in FY2023"},
            ],
            "net_profit": [{"period": "FY2024", "value": 425.0, "page": "pasted", "source": "nimbus.txt", "evidence": "425.0 crore in FY2024"}],
            "equity": [{"period": "FY2024", "value": 2500.0, "page": 77, "source": "nimbus.txt", "evidence": "made up"}],
        },
    }
    monkeypatch.setattr(extraction, "generate_json", lambda *a, **k: fake_figures)
    record = extraction.extract_figures("Nimbus")
    assert record["figures"]["revenue"][0]["value"] == 4250.0
    assert record["figures"]["equity"] == []           # page 77 was never shown -> discarded
    assert any("equity" in d for d in record["discarded"])
    assert extraction.load_figures("Nimbus")["units"] == "crore"

    # 2. graph: two relationships with real quotes, one fabricated
    fake_graph = {
        "entities": [{"name": "Nimbus Foods Limited", "type": "Company"}, {"name": "Nimbus Beverages Private Limited", "type": "Subsidiary"}],
        "relationships": [
            {"source": "Nimbus Foods Limited", "source_type": "Company", "target": "Nimbus Beverages Private Limited", "target_type": "Subsidiary",
             "relationship": "OWNS", "evidence": "owns 100% of Nimbus Beverages Private Limited", "source_file": "nimbus.txt", "page": "pasted",
             "confidence": 0.95, "status": "confirmed"},
            {"source": "Nimbus Foods Limited", "source_type": "Company", "target": "Palm oil prices", "target_type": "Financial Event",
             "relationship": "AFFECTED_BY", "evidence": "palm oil prices", "source_file": "nimbus.txt", "page": "pasted",
             "confidence": 0.7, "status": "inferred"},
            {"source": "Nimbus Foods Limited", "source_type": "Company", "target": "Zenith", "target_type": "Company",
             "relationship": "COMPETES_WITH", "evidence": "competes with Zenith", "source_file": "nimbus.txt", "page": "pasted",
             "confidence": 0.9, "status": "confirmed"},
        ],
    }
    monkeypatch.setattr(graph_extraction, "generate_json", lambda *a, **k: fake_graph)
    summary = graph_extraction.build_graph("Nimbus", reset=True)
    assert summary["relationships_added"] == 2 and summary["rejected"] == 1
    graph = kg.load_graph("Nimbus")
    assert kg.statistics(graph) == {"entities": 3, "relationships": 2, "confirmed": 1, "inferred": 1,
                                    "by_type": {"Company": 1, "Subsidiary": 1, "Financial Event": 1}}
    assert kg.find_company_node(graph, "Nimbus") == "company:nimbus_foods_limited"

    # 3. hybrid chat: the prompt must carry passages, graph rows and ratios
    captured = {}

    def fake_text(prompt, system_instruction=None, **kwargs):
        captured["prompt"] = prompt
        return "Answer (nimbus.txt, page pasted)."

    monkeypatch.setattr(hybrid_retrieval, "generate_text", fake_text)
    result = hybrid_retrieval.chat("How can palm oil prices affect this company?", "Nimbus",
                                   history=[{"role": "user", "content": "Tell me about revenue"}])
    assert result["status"] == "ok"
    assert "AFFECTED_BY" in captured["prompt"]
    assert "Net profit margin = 10.00%" in captured["prompt"]      # 425/4250
    assert "Revenue growth = 14.86%" in captured["prompt"]        # (4250-3700)/3700
    assert "graph link is never proof" in captured["prompt"]
    assert result["graph_rows"][0]["target_name"] == "Palm oil prices"

    # company resolution from the question text, and a clear error when unknown
    assert hybrid_retrieval.resolve_company("what about nimbus debt?", None) == "Nimbus"
    assert hybrid_retrieval.chat("what about its debt?", None)["status"] == "error"

    # 4. report: a label outside the allowed set becomes "insufficient information"
    fake_narrative = {"overview": "Packaged foods company (nimbus.txt, page pasted).", "segments": "Snacks and Beverages.",
                      "key_findings": ["Revenue rose 14.9% (nimbus.txt, page pasted)"], "risks": ["Palm oil prices"],
                      "conclusion_label": "BUY NOW", "conclusion": "text", "not_covered": ["cash flows"]}
    monkeypatch.setattr(report_generator, "generate_json", lambda *a, **k: fake_narrative)
    report = report_generator.generate_report("Nimbus")
    assert report["label"] == "insufficient information"
    md = report["markdown"]
    for heading in ("## 1. Overview", "## 2. Key Financials", "## 3. Ratio Analysis", "## 4. Trends", "## 5. Business Segments",
                    "## 6. Graph Insights", "## 7. Key Findings", "## 8. Risks", "## 9. Limitations", "## 10. Conclusion"):
        assert heading in md
    assert "Not found in report" in md                     # e.g. the EPS row
    assert "4,250.0" in md and "net_profit / revenue x 100" in md
    assert "rose" in md                                    # revenue trend FY2023 -> FY2024
    assert "OWNS Nimbus Beverages Private Limited" in md
    assert "not investment advice" in md.lower()
    assert Path(report["path"]).exists() and Path(report["path"]).parent == world
    meta = json.loads(Path(report["path"]).with_suffix(".json").read_text())
    assert meta["narrative"]["conclusion_label"] == "insufficient information"
