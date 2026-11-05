"""Report ingestion: PDF or pasted text -> cleaned pages -> chunks -> ChromaDB.

Also keeps a small JSON registry (knowledge_base/documents.json) of every
ingested document so we can list companies, reject duplicates (by file
hash) and survive restarts.

Concurrency: Streamlit cannot stop a running ingestion when the user clicks
again, so a document is "claimed" in the registry (status in_progress) under
a lock before the slow work starts. A second attempt for the same file is
rejected immediately instead of running in parallel.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DOCUMENT_REGISTRY_FILE

ProgressCallback = Callable[[str, int, int], None]  # (stage, done, total)

_REGISTRY_LOCK = threading.Lock()


class IngestionError(Exception):
    """Raised for user-facing ingestion problems (bad file, scanned PDF...)."""


class DuplicateDocumentError(IngestionError):
    """Raised when the same file/text was already ingested (or is being ingested)."""


# ---------------------------------------------------------------------------
# Text extraction and cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace while keeping line breaks that separate rows."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces/tabs
    text = re.sub(r"\n\s*\n+", "\n\n", text)       # collapse blank lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # join hyphenated line breaks
    return text.strip()


def extract_pdf_pages(pdf_path: str | Path, progress: ProgressCallback | None = None) -> list[dict[str, Any]]:
    """Return [{"page": 1, "text": "..."}, ...] for a PDF.

    Raises IngestionError if the file is not a readable PDF or contains no
    extractable text (scanned image PDFs need OCR, which is out of scope).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise IngestionError(f"File not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise IngestionError("Only PDF files are supported (or paste the text instead).")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as error:  # noqa: BLE001
                raise IngestionError("This PDF is password protected.") from error
        total_pages = len(reader.pages)
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"page": number, "text": text})
            if progress and (number % 5 == 0 or number == total_pages):
                progress("reading", number, total_pages)
    except IngestionError:
        raise
    except Exception as error:  # noqa: BLE001
        raise IngestionError(f"Could not read the PDF: {error}") from error

    if not pages:
        raise IngestionError(
            "No extractable text was found in this PDF. It is probably a scanned "
            "image. OCR is needed (not supported here); try pasting the text instead."
        )
    if total_pages and len(pages) < total_pages * 0.3:
        pages[0]["warning"] = (
            f"Only {len(pages)} of {total_pages} pages contain extractable text; "
            "the rest may be scanned images."
        )
    return pages


def pasted_text_pages(text: str) -> list[dict[str, Any]]:
    """Wrap pasted text as a single 'page' called 'pasted'."""
    cleaned = clean_text(text or "")
    if len(cleaned) < 50:
        raise IngestionError("Pasted text is too short to analyse (need at least 50 characters).")
    return [{"page": "pasted", "text": cleaned}]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split each page into overlapping character windows.

    Chunks never cross page boundaries so the page citation stays exact.
    Returns [{"page", "text", "index"}] where index counts chunks per document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    index = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page["page"], "text": piece, "index": index})
                index += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def compute_hash(content: bytes) -> str:
    """SHA-256 of the raw file or text; used as the duplicate check."""
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Document registry (JSON file)
# ---------------------------------------------------------------------------

def load_registry() -> list[dict[str, Any]]:
    """Read the list of ingested documents (duplicates by id are collapsed)."""
    if not DOCUMENT_REGISTRY_FILE.exists():
        return []
    try:
        entries = json.loads(DOCUMENT_REGISTRY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    seen: set[str] = set()
    unique = []
    for entry in entries:
        if entry.get("document_id") in seen:
            continue
        seen.add(entry.get("document_id"))
        unique.append(entry)
    return unique


def save_registry(entries: list[dict[str, Any]]) -> None:
    """Write the registry to disk."""
    DOCUMENT_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_REGISTRY_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _is_ready(entry: dict[str, Any]) -> bool:
    return entry.get("status", "ready") == "ready"


def list_companies() -> list[dict[str, Any]]:
    """Distinct companies with at least one fully ingested document."""
    companies: dict[str, dict[str, Any]] = {}
    for entry in load_registry():
        if not _is_ready(entry):
            continue
        info = companies.setdefault(
            entry["company"], {"company": entry["company"], "ticker": entry.get("ticker", ""), "documents": 0}
        )
        info["documents"] += 1
        if not info["ticker"] and entry.get("ticker"):
            info["ticker"] = entry["ticker"]
    return sorted(companies.values(), key=lambda item: item["company"].lower())


def list_documents(company: str | None = None) -> list[dict[str, Any]]:
    """Fully ingested documents, optionally filtered by company."""
    entries = [entry for entry in load_registry() if _is_ready(entry)]
    if company:
        entries = [entry for entry in entries if entry["company"] == company]
    return entries


def find_duplicate(file_hash: str) -> dict[str, Any] | None:
    """Return the registry entry with the same hash (ready or in progress)."""
    for entry in load_registry():
        if entry.get("hash") == file_hash:
            return entry
    return None


def _claim(entry: dict[str, Any]) -> None:
    """Atomically reject a duplicate or record the document as in progress."""
    with _REGISTRY_LOCK:
        duplicate = find_duplicate(entry["hash"])
        if duplicate:
            if _is_ready(duplicate):
                raise DuplicateDocumentError(
                    f"This document was already ingested as '{duplicate['source']}' for "
                    f"{duplicate['company']} on {duplicate['ingested_at'][:10]}."
                )
            raise DuplicateDocumentError(
                f"'{duplicate['source']}' is already being ingested (started "
                f"{duplicate['ingested_at'][11:19]}). Please wait for it to finish."
            )
        entries = load_registry()
        entries.append(entry)
        save_registry(entries)


def _release(document_id: str) -> None:
    """Remove a claim after a failed ingestion."""
    with _REGISTRY_LOCK:
        save_registry([entry for entry in load_registry() if entry["document_id"] != document_id])


def _finalize(document_id: str, **updates: Any) -> None:
    """Mark a claimed document as ready and store its page/chunk counts."""
    with _REGISTRY_LOCK:
        entries = load_registry()
        for entry in entries:
            if entry["document_id"] == document_id:
                entry.update(updates)
                entry["status"] = "ready"
        save_registry(entries)


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_document(
    company: str,
    ticker: str = "",
    document_type: str = "Annual Report",
    reporting_period: str = "",
    pdf_bytes: bytes | None = None,
    pdf_name: str | None = None,
    pasted_text: str | None = None,
    pasted_title: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Ingest one PDF (bytes + name) or one pasted text for a company.

    progress_callback(stage, done, total) is called with stages
    "reading", "chunking", "embedding" and "done" so a UI can show a bar.
    Returns a summary dict with document_id, pages, chunks and any warning.
    Raises DuplicateDocumentError if the exact file/text was ingested before
    or is being ingested right now.
    """
    import vector_store  # local import keeps this module importable without chromadb

    def progress(stage: str, done: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, done, total)

    company = (company or "").strip()
    if not company:
        raise IngestionError("Company name is required.")
    ticker = (ticker or "").strip().upper()
    reporting_period = (reporting_period or "").strip() or "Not specified"

    if pdf_bytes is not None:
        if not pdf_name:
            raise IngestionError("PDF file name is missing.")
        raw = pdf_bytes
        source = Path(pdf_name).name
    elif pasted_text is not None:
        raw = pasted_text.encode("utf-8")
        source = (pasted_title or f"pasted_{company}").strip()
        if not source.lower().endswith(".txt"):
            source += ".txt"
        source = re.sub(r"[^\w.\- ]", "_", source)
    else:
        raise IngestionError("Provide either a PDF or pasted text.")

    file_hash = compute_hash(raw)
    document_id = file_hash[:16]
    company_dir = DATA_DIR / re.sub(r"[^\w\- ]", "_", company)
    saved_path = company_dir / f"{document_id}_{source}"

    _claim(
        {
            "document_id": document_id,
            "hash": file_hash,
            "source": source,
            "stored_path": str(saved_path),
            "company": company,
            "ticker": ticker,
            "document_type": document_type,
            "reporting_period": reporting_period,
            "pages": 0,
            "chunks": 0,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            "status": "in_progress",
        }
    )

    try:
        progress("reading", 0, 1)
        company_dir.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(raw)

        if pdf_bytes is not None:
            pages = extract_pdf_pages(saved_path, progress=progress)
        else:
            pages = pasted_text_pages(pasted_text or "")
        warning = pages[0].pop("warning", None) if pages else None

        progress("chunking", 0, 1)
        pieces = chunk_pages(pages)
        chunks = []
        for piece in pieces:
            chunk_id = f"{document_id}_{piece['index']:05d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece["text"],
                    "metadata": {
                        "document_id": document_id,
                        "source": source,
                        "page": piece["page"],
                        "company": company,
                        "ticker": ticker,
                        "document_type": document_type,
                        "reporting_period": reporting_period,
                        "chunk_id": chunk_id,
                    },
                }
            )
        progress("chunking", 1, 1)

        progress("embedding", 0, max(len(chunks), 1))
        stored = vector_store.add_chunks(
            chunks, progress_callback=lambda done, total: progress("embedding", done, total)
        )
    except Exception:
        # Undo everything so the user can simply try again.
        _release(document_id)
        if saved_path.exists():
            saved_path.unlink()
        try:
            vector_store.delete_document(document_id)
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass
        raise

    _finalize(document_id, pages=len(pages), chunks=stored)
    progress("done", 1, 1)
    return {
        "document_id": document_id,
        "source": source,
        "company": company,
        "pages": len(pages),
        "chunks": stored,
        "warning": warning,
    }


def delete_document(document_id: str) -> None:
    """Remove a document from ChromaDB, the registry and the data folder."""
    import vector_store

    with _REGISTRY_LOCK:
        entries = load_registry()
        keep = []
        for entry in entries:
            if entry["document_id"] == document_id:
                stored = Path(entry.get("stored_path", ""))
                if stored.exists():
                    stored.unlink()
            else:
                keep.append(entry)
        save_registry(keep)
    vector_store.delete_document(document_id)


def delete_company(company: str) -> int:
    """Remove every document of a company. Returns how many were removed."""
    docs = [entry for entry in load_registry() if entry["company"] == company]
    for entry in docs:
        delete_document(entry["document_id"])
    return len(docs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ingest.py <company> <path-to-pdf> [period] [doc type]")
        sys.exit(1)
    path = Path(sys.argv[2])
    summary = ingest_document(
        company=sys.argv[1],
        pdf_bytes=path.read_bytes(),
        pdf_name=path.name,
        reporting_period=sys.argv[3] if len(sys.argv) > 3 else "",
        document_type=sys.argv[4] if len(sys.argv) > 4 else "Annual Report",
        progress_callback=lambda stage, done, total: print(f"  {stage}: {done}/{total}"),
    )
    print(json.dumps(summary, indent=2))
