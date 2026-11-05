"""Smoke test: verifies the environment before you run the app.

Run:  python check_setup.py
It checks imports, folders, the embedding model, ChromaDB, and (if a key is
set) that Gemini answers. Nothing here changes your data.
"""

from __future__ import annotations

import sys


def ok(message: str) -> None:
    print(f"  [OK]   {message}")


def fail(message: str) -> None:
    print(f"  [FAIL] {message}")


def main() -> int:
    problems = 0
    print(f"Python {sys.version.split()[0]}")

    print("\n1. Imports")
    for module in ("streamlit", "google.genai", "sentence_transformers", "chromadb", "networkx", "pypdf", "dotenv", "plotly", "pandas"):
        try:
            __import__(module)
            ok(module)
        except Exception as error:  # noqa: BLE001
            fail(f"{module}: {error}")
            problems += 1

    print("\n2. Configuration")
    import config

    for name in ("DATA_DIR", "CHROMA_DIR", "KNOWLEDGE_BASE_DIR", "REPORTS_DIR"):
        path = getattr(config, name)
        (ok if path.exists() else fail)(f"{name} -> {path}")
    ok(f"CHUNK_SIZE={config.CHUNK_SIZE} CHUNK_OVERLAP={config.CHUNK_OVERLAP} TOP_K={config.TOP_K} RELEVANCE_THRESHOLD={config.RELEVANCE_THRESHOLD}")
    if config.GEMINI_API_KEY:
        ok(f"GEMINI_API_KEY is set (model: {config.GEMINI_MODEL})")
    else:
        fail("GEMINI_API_KEY is empty. Copy .env.example to .env and add your key from https://aistudio.google.com/apikey")
        problems += 1

    print("\n3. Local embeddings (first run downloads ~90 MB)")
    try:
        import vector_store

        vectors = vector_store.embed_texts(["revenue grew 12% this year", "the cat sat on the mat"])
        ok(f"embedding model '{config.EMBEDDING_MODEL}' loaded; vector size {len(vectors[0])}")
    except Exception as error:  # noqa: BLE001
        fail(f"embeddings: {error}")
        problems += 1

    print("\n4. ChromaDB")
    try:
        import vector_store

        count = vector_store.count_chunks()
        ok(f"collection '{config.CHROMA_COLLECTION}' opened; {count} chunks stored")
    except Exception as error:  # noqa: BLE001
        fail(f"ChromaDB: {error}")
        problems += 1

    print("\n5. Gemini")
    if config.GEMINI_API_KEY:
        try:
            from gemini_client import generate_text, get_client

            reply = generate_text("Reply with the single word OK.", max_retries=1)
            ok(f"model '{config.GEMINI_MODEL}' replied: {reply[:40]!r}")
        except Exception as error:  # noqa: BLE001
            fail(f"Gemini call failed: {error}")
            problems += 1
            try:
                names = [m.name for m in get_client().models.list() if "flash" in m.name.lower()]
                print("       Flash models visible to your key (set GEMINI_MODEL in .env):")
                for name in names[:15]:
                    print("         -", name.replace("models/", ""))
            except Exception:  # noqa: BLE001
                pass
    else:
        print("  [SKIP] no API key; LLM features will show a clear error in the app until it is set")

    print("\nResult:", "ALL GOOD" if problems == 0 else f"{problems} problem(s) to fix")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
