"""Central configuration for StockGraph AI.

All settings come from the .env file (loaded with python-dotenv) with
sensible defaults. Nothing secret is hardcoded here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env from the project folder (if it exists).
load_dotenv(BASE_DIR / ".env")

# Quieter first-run download of the embedding model on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"                      # uploaded PDFs / pasted text
CHROMA_DIR = BASE_DIR / "chroma_db"               # vector database
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"  # graphs, figures, registry
REPORTS_DIR = BASE_DIR / "reports"                # exported reports

for _folder in (DATA_DIR, CHROMA_DIR, KNOWLEDGE_BASE_DIR, REPORTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

DOCUMENT_REGISTRY_FILE = KNOWLEDGE_BASE_DIR / "documents.json"


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back to a default."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to a default."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# LLM (Google Gemini, free tier)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
# Used automatically when the main model keeps answering "busy" (503).
GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3-flash-preview").strip()
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", 4)
LLM_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# Embeddings and vector store
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION: str = "stockgraph_reports"

# ---------------------------------------------------------------------------
# Chunking and retrieval
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 200)
TOP_K: int = _env_int("TOP_K", 5)
# Minimum cosine similarity (0-1) for a chunk to count as relevant.
RELEVANCE_THRESHOLD: float = _env_float("RELEVANCE_THRESHOLD", 0.25)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
DOCUMENT_TYPES = (
    "Annual Report",
    "Quarterly Results",
    "Investor Presentation",
    "Earnings Call Transcript",
    "Other",
)

DISCLAIMER = "Educational research only, not investment advice."
