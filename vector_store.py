"""Local embeddings (sentence-transformers) + persistent ChromaDB storage.

This is the only module that talks to ChromaDB. Every chunk carries the
metadata needed for citations: document_id, source, page, company,
ticker, document_type, reporting_period, chunk_id.
"""

from __future__ import annotations

from typing import Any

from config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL


class VectorStoreError(Exception):
    """Raised when embeddings or ChromaDB fail."""


_embedder = None
_collection = None


def get_embedder():
    """Load the local embedding model once (first call downloads it)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as error:  # noqa: BLE001
            raise VectorStoreError(
                f"Could not load embedding model '{EMBEDDING_MODEL}': {error}"
            ) from error
    return _embedder


def get_collection():
    """Open (once) the persistent ChromaDB collection with cosine distance."""
    global _collection
    if _collection is None:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:  # noqa: BLE001
            raise VectorStoreError(f"Could not open ChromaDB: {error}") from error
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into normalised embedding vectors."""
    if not texts:
        return []
    try:
        vectors = get_embedder().encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Embedding failed: {error}") from error


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB only accepts str/int/float/bool values; drop None."""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def add_chunks(chunks: list[dict[str, Any]], batch_size: int = 128, progress_callback=None) -> int:
    """Embed and store chunks. Each chunk: {"chunk_id", "text", "metadata"}.

    progress_callback(done, total) is called after every batch.
    """
    if not chunks:
        return 0
    collection = get_collection()
    stored = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        try:
            collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[_clean_metadata(chunk["metadata"]) for chunk in batch],
                embeddings=embed_texts([chunk["text"] for chunk in batch]),
            )
        except VectorStoreError:
            raise
        except Exception as error:  # noqa: BLE001
            raise VectorStoreError(f"Storing chunks in ChromaDB failed: {error}") from error
        stored += len(batch)
        if progress_callback:
            progress_callback(stored, len(chunks))
    return stored


def query(
    text: str,
    top_k: int = 5,
    company: str | None = None,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search. Returns hits sorted by similarity (1 = identical).

    Each hit: {"text", "metadata", "similarity"}.
    """
    if not text or not text.strip():
        return []
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    filters = []
    if company:
        filters.append({"company": company})
    if document_id:
        filters.append({"document_id": document_id})
    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    try:
        result = collection.query(
            query_embeddings=embed_texts([text]),
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except VectorStoreError:
        raise
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"ChromaDB query failed: {error}") from error

    hits = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        # cosine distance = 1 - cosine similarity
        hits.append({"text": doc, "metadata": meta, "similarity": 1.0 - float(distance)})
    hits.sort(key=lambda hit: hit["similarity"], reverse=True)
    return hits


def get_chunks_for_document(document_id: str) -> list[dict[str, Any]]:
    """Return every stored chunk of one document, ordered by page."""
    collection = get_collection()
    try:
        result = collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"ChromaDB read failed: {error}") from error
    chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(result.get("documents") or [], result.get("metadatas") or [])
    ]
    chunks.sort(key=lambda c: (str(c["metadata"].get("page")), c["metadata"].get("chunk_id", "")))
    return chunks


def delete_document(document_id: str) -> None:
    """Remove all chunks of a document."""
    try:
        get_collection().delete(where={"document_id": document_id})
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"ChromaDB delete failed: {error}") from error


def count_chunks(company: str | None = None) -> int:
    """How many chunks are stored (optionally for one company)."""
    collection = get_collection()
    if company is None:
        return collection.count()
    result = collection.get(where={"company": company}, include=[])
    return len(result.get("ids") or [])
