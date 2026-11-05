"""TDD: progress reporting, the ingestion lock, and registry hygiene.

Written before the implementation. They describe how ingestion must behave
so a slow 150-page PDF cannot be started three times in parallel and the
user always sees what stage is running.
"""

import threading

import pytest

import ingest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import vector_store

    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ingest, "DOCUMENT_REGISTRY_FILE", tmp_path / "documents.json")
    stored = []

    def fake_add(chunks, progress_callback=None, **kwargs):
        stored.extend(chunks)
        if progress_callback:
            progress_callback(len(chunks), len(chunks))
        return len(chunks)

    monkeypatch.setattr(vector_store, "add_chunks", fake_add)
    monkeypatch.setattr(vector_store, "delete_document", lambda document_id: None)
    return stored


TEXT = "Revenue from operations was 1,000 crore in FY2024 compared with 800 crore in FY2023. " * 4


def test_progress_callback_reports_each_stage(isolated):
    events = []
    ingest.ingest_document(
        company="Demo Ltd", pasted_text=TEXT, pasted_title="n",
        progress_callback=lambda stage, done, total: events.append((stage, done, total)),
    )
    stages = [stage for stage, _, _ in events]
    assert stages[0] == "reading"
    assert "embedding" in stages
    assert stages[-1] == "done"
    embedding = [e for e in events if e[0] == "embedding"][-1]
    assert embedding[1] == embedding[2] >= 1  # finished: done == total chunks


def test_second_ingest_of_same_file_while_first_is_running_is_rejected(isolated, monkeypatch):
    """Simulates Streamlit re-running the script while ingestion is still busy."""
    import vector_store

    started = threading.Event()
    release = threading.Event()
    original = vector_store.add_chunks

    def slow_add(chunks, **kwargs):
        started.set()
        release.wait(timeout=5)
        return original(chunks, **kwargs)

    monkeypatch.setattr(vector_store, "add_chunks", slow_add)
    results = {}

    def first():
        results["first"] = ingest.ingest_document(company="Demo Ltd", pasted_text=TEXT, pasted_title="n")

    worker = threading.Thread(target=first)
    worker.start()
    assert started.wait(timeout=5)
    with pytest.raises(ingest.DuplicateDocumentError) as error:
        ingest.ingest_document(company="Demo Ltd", pasted_text=TEXT, pasted_title="n")
    assert "being ingested" in str(error.value).lower()
    release.set()
    worker.join(timeout=10)
    assert results["first"]["chunks"] >= 1
    assert len(ingest.load_registry()) == 1
    assert len(ingest.list_documents("Demo Ltd")) == 1


def test_failed_ingestion_leaves_no_claim_behind(isolated, monkeypatch):
    import vector_store

    def boom(chunks, **kwargs):
        raise vector_store.VectorStoreError("disk full")

    monkeypatch.setattr(vector_store, "add_chunks", boom)
    with pytest.raises(vector_store.VectorStoreError):
        ingest.ingest_document(company="Demo Ltd", pasted_text=TEXT, pasted_title="n")
    assert ingest.load_registry() == []          # claim removed
    assert ingest.find_duplicate(ingest.compute_hash(TEXT.encode())) is None


def test_in_progress_documents_are_not_listed_as_ready(isolated):
    ingest.save_registry([{"document_id": "x", "hash": "h", "source": "s.pdf", "company": "Demo Ltd",
                           "ticker": "", "document_type": "Annual Report", "reporting_period": "FY2024",
                           "pages": 0, "chunks": 0, "ingested_at": "2026-11-03T10:00:00", "status": "in_progress"}])
    assert ingest.list_documents("Demo Ltd") == []
    assert ingest.list_companies() == []
    assert ingest.find_duplicate("h") is not None      # but it still blocks a re-upload


def test_registry_duplicates_are_collapsed_on_load(isolated):
    row = {"document_id": "same", "hash": "h", "source": "s.pdf", "company": "Demo Ltd", "ticker": "",
           "document_type": "Annual Report", "reporting_period": "FY2024", "pages": 1, "chunks": 1,
           "ingested_at": "2026-11-03T10:00:00", "status": "ready"}
    ingest.save_registry([row, dict(row), dict(row)])
    assert len(ingest.load_registry()) == 1
