"""Chunking, hashing, duplicate detection and metadata (no embeddings needed)."""

import pytest

import ingest


def test_clean_text_collapses_whitespace_and_joins_hyphenation():
    assert ingest.clean_text("Reve-\nnue   grew\n\n\n\nfast") == "Revenue grew\n\nfast"


def test_chunks_do_not_cross_pages_and_overlap():
    pages = [{"page": 1, "text": "a" * 1500}, {"page": 2, "text": "b" * 300}]
    chunks = ingest.chunk_pages(pages, chunk_size=1000, overlap=200)
    assert all(set(c["text"]) == {"a"} for c in chunks if c["page"] == 1)
    assert [c["page"] for c in chunks] == [1, 1, 2]
    assert chunks[0]["text"][-200:] == chunks[1]["text"][:200]
    assert [c["index"] for c in chunks] == [0, 1, 2]


def test_chunk_validation():
    with pytest.raises(ValueError):
        ingest.chunk_pages([{"page": 1, "text": "x"}], chunk_size=100, overlap=100)


def test_hash_is_stable():
    assert ingest.compute_hash(b"abc") == ingest.compute_hash(b"abc")
    assert ingest.compute_hash(b"abc") != ingest.compute_hash(b"abd")


def test_pasted_text_uses_pasted_page():
    pages = ingest.pasted_text_pages("Revenue for the year was 1,000 crore, up from 800 crore last year.")
    assert pages[0]["page"] == "pasted"


def test_short_pasted_text_rejected():
    with pytest.raises(ingest.IngestionError):
        ingest.pasted_text_pages("too short")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point ingestion at temp folders and skip real embeddings."""
    import vector_store

    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ingest, "DOCUMENT_REGISTRY_FILE", tmp_path / "documents.json")
    stored = []
    monkeypatch.setattr(vector_store, "add_chunks", lambda chunks, **kwargs: stored.extend(chunks) or len(chunks))
    monkeypatch.setattr(vector_store, "delete_document", lambda document_id: None)
    return stored


def test_ingest_pasted_text_sets_metadata_and_registry(isolated):
    text = "Revenue from operations was 1,000 crore in FY2024 compared with 800 crore in FY2023. " * 3
    summary = ingest.ingest_document(
        company="Demo Ltd", ticker="demo", document_type="Annual Report",
        reporting_period="FY2024", pasted_text=text, pasted_title="demo notes",
    )
    assert summary["chunks"] == len(isolated) >= 1
    meta = isolated[0]["metadata"]
    for key in ("document_id", "source", "page", "company", "ticker", "document_type", "reporting_period", "chunk_id"):
        assert key in meta
    assert meta["company"] == "Demo Ltd" and meta["ticker"] == "DEMO" and meta["page"] == "pasted"
    assert ingest.list_companies()[0]["company"] == "Demo Ltd"
    assert ingest.list_documents("Demo Ltd")[0]["document_id"] == summary["document_id"]


def test_duplicate_ingestion_is_rejected(isolated):
    text = "Net profit for the year was 150 crore against 120 crore in the previous year. " * 3
    ingest.ingest_document(company="Demo Ltd", pasted_text=text, pasted_title="n1")
    with pytest.raises(ingest.DuplicateDocumentError):
        ingest.ingest_document(company="Other Ltd", pasted_text=text, pasted_title="n2")
    assert len(ingest.load_registry()) == 1


def test_companies_stay_separate(isolated):
    ingest.ingest_document(company="A Ltd", pasted_text="Alpha revenue 100 crore for FY2024 as reported by management. " * 2)
    ingest.ingest_document(company="B Ltd", pasted_text="Beta revenue 200 crore for FY2024 as reported by management. " * 2)
    companies = {c["company"] for c in ingest.list_companies()}
    assert companies == {"A Ltd", "B Ltd"}
    assert {c["metadata"]["company"] for c in isolated} == {"A Ltd", "B Ltd"}
    assert ingest.delete_company("A Ltd") == 1
    assert {c["company"] for c in ingest.list_companies()} == {"B Ltd"}
