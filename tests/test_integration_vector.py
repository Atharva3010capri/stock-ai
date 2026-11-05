"""Integration tests: real sentence-transformers embeddings + a real ChromaDB
collection in a temporary folder. No LLM, no network (model is cached after
the first run). Marked 'integration' so they can be skipped with
`pytest -m "not integration"` when you only want the fast unit tests.
"""

from pathlib import Path

import pytest

import ingest
import rag
import vector_store

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "demo_report_text.txt"


@pytest.fixture
def real_store(tmp_path, monkeypatch):
    """Fresh ChromaDB in tmp_path; the embedding model is shared."""
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(vector_store, "CHROMA_COLLECTION", "test_reports")
    monkeypatch.setattr(vector_store, "_collection", None)
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ingest, "DOCUMENT_REGISTRY_FILE", tmp_path / "documents.json")
    yield tmp_path
    vector_store._collection = None


def test_ingest_then_retrieve_with_real_embeddings(real_store):
    text = SAMPLE.read_text(encoding="utf-8")
    summary = ingest.ingest_document("Nimbus Foods Limited", "NIMBUS", "Annual Report", "FY2024",
                                     pasted_text=text, pasted_title="nimbus")
    assert summary["chunks"] >= 3
    assert vector_store.count_chunks("Nimbus Foods Limited") == summary["chunks"]

    hits = rag.retrieve("How can crude oil prices affect this company?", "Nimbus Foods Limited")
    assert hits, "a relevant question must clear the relevance threshold"
    assert "crude oil" in hits[0]["text"].lower()
    assert hits[0]["metadata"]["page"] == "pasted"
    assert rag.format_citation(hits[0]["metadata"]) == "nimbus.txt, pasted"

    assert rag.retrieve("Who won the cricket world cup?", "Nimbus Foods Limited") == []
    assert rag.retrieve("revenue", "Some Other Company") == []   # company isolation

    with pytest.raises(ingest.DuplicateDocumentError):
        ingest.ingest_document("Nimbus Foods Limited", pasted_text=text, pasted_title="again")


def test_delete_document_removes_chunks(real_store):
    summary = ingest.ingest_document("Temp Co", pasted_text=SAMPLE.read_text(encoding="utf-8"), pasted_title="t")
    assert vector_store.count_chunks("Temp Co") > 0
    ingest.delete_document(summary["document_id"])
    assert vector_store.count_chunks("Temp Co") == 0
    assert ingest.list_documents("Temp Co") == []


def test_embedding_batches_report_progress(real_store):
    chunks = [{"chunk_id": f"c{i}", "text": f"Paragraph number {i} about revenue and profit.",
               "metadata": {"document_id": "d", "source": "s.txt", "page": 1, "company": "P", "chunk_id": f"c{i}"}}
              for i in range(150)]
    seen = []
    stored = vector_store.add_chunks(chunks, batch_size=64, progress_callback=lambda done, total: seen.append((done, total)))
    assert stored == 150
    assert seen == [(64, 150), (128, 150), (150, 150)]
