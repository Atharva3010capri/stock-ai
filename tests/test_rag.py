"""RAG behaviour with the vector store and LLM mocked (no network, no model download)."""

import rag
import vector_store


def fake_hits(similarities):
    return [
        {"text": f"passage {i}", "similarity": s,
         "metadata": {"source": "ar.pdf", "page": 10 + i, "company": "Demo", "chunk_id": f"c{i}"}}
        for i, s in enumerate(similarities)
    ]


def test_irrelevant_question_returns_not_found(monkeypatch):
    monkeypatch.setattr(vector_store, "query", lambda *a, **k: fake_hits([0.05, 0.02]))
    called = []
    monkeypatch.setattr(rag, "generate_text", lambda *a, **k: called.append(1) or "should not run")
    result = rag.answer_question("What is the capital of France?", "Demo")
    assert result["status"] == "no_evidence"
    assert "could not find" in result["answer"].lower()
    assert called == []  # the LLM was never asked to guess


def test_relevant_question_cites_sources(monkeypatch):
    monkeypatch.setattr(vector_store, "query", lambda *a, **k: fake_hits([0.62, 0.41, 0.1]))
    captured = {}

    def fake_generate(prompt, system_instruction=None, **kwargs):
        captured["prompt"] = prompt
        captured["system"] = system_instruction
        return "Revenue grew (ar.pdf, page 10)."

    monkeypatch.setattr(rag, "generate_text", fake_generate)
    result = rag.answer_question("How did revenue change?", "Demo")
    assert result["status"] == "ok"
    assert [s["page"] for s in result["sources"]] == [10, 11]  # the 0.1 hit was filtered out
    assert "passage 0" in captured["prompt"] and "passage 2" not in captured["prompt"]
    assert "Never invent figures" in captured["system"]
    assert "not investment advice" in result["answer"].lower()


def test_retrieve_multi_deduplicates(monkeypatch):
    monkeypatch.setattr(vector_store, "query", lambda *a, **k: fake_hits([0.5, 0.4]))
    merged = rag.retrieve_multi(["q1", "q2", "q3"], "Demo", per_query=2)
    assert len(merged) == 2
    assert merged[0]["similarity"] >= merged[1]["similarity"]
