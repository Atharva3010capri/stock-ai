# StockGraph AI

A local Streamlit app that analyses **one company at a time from the financial
reports you give it** (annual reports, quarterly results, investor
presentations, transcripts). Upload a PDF or paste text; the app extracts key
figures with page citations, computes fundamental ratios in Python, answers
questions with citations (RAG), builds a knowledge graph of entities and
relationships, chats across all of that, and exports a Markdown report.

There is **no live market data, no price charts, no news and no web access**.
It is an educational research tool, not investment advice. No buy/sell calls,
no price targets.

## How the parts connect

1. **Ingestion** (`ingest.py`, `vector_store.py`): PDF pages or pasted text are
   cleaned, split into overlapping chunks (page boundaries are never crossed),
   embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB with
   metadata (`document_id, source, page, company, ticker, document_type,
   reporting_period, chunk_id`). A SHA-256 file hash blocks duplicate uploads.
2. **RAG** (`rag.py`): a question is embedded, the closest chunks of the
   selected company are fetched, chunks under `RELEVANCE_THRESHOLD` are dropped,
   and Gemini answers using only the remaining passages, citing file and page.
3. **Figure extraction** (`extraction.py`): the chunks most likely to hold
   revenue, profit, debt, etc. are shown to Gemini, which returns JSON. Python
   validates every value (numeric, page actually shown, quote matched to the
   page) and discards the rest. Missing values stay "Not found in report". You
   can correct values by hand.
4. **Fundamentals** (`fundamentals.py`): ratios are computed in plain Python
   from the validated figures, each with its formula, inputs and a one-line
   explanation. Missing inputs mean the ratio is skipped, never estimated.
5. **Knowledge graph** (`graph_extraction.py`, `knowledge_graph.py`): Gemini
   extracts entities and relationships with an evidence quote; Python keeps only
   relationships whose quote appears in the cited passage. Stored as JSON in
   `knowledge_base/`, drawn with Plotly (solid = confirmed, dashed = inferred).
6. **Hybrid chat** (`hybrid_retrieval.py`) and **report**
   (`report_generator.py`): combine passages, graph links and ratios into one
   grounded context for Gemini, with chat history for follow-ups.

All LLM calls go through `gemini_client.py`, which retries on free-tier rate
limits (HTTP 429) and turns errors into readable messages.

## Setup (Windows PowerShell)

Requirements: Python 3.11, 3.12 or 3.13 and a free Gemini API key from
<https://aistudio.google.com/apikey>.

```powershell
cd D:\code\StockGraph_AI_Source
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env        # paste your GEMINI_API_KEY
python check_setup.py
```

If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass` and try
again. The first run of `check_setup.py` downloads the embedding model (~90 MB).

If `check_setup.py` says the model was not found, it prints the Flash models
your key can use; put one of them in `GEMINI_MODEL` in `.env`. As of November
2026 `gemini-2.5-flash` is retired for new keys; `gemini-3.5-flash` works, with
`gemini-3-flash-preview` as the fallback.

## Run

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## Tests

Unit tests run in about three seconds with the vector store and Gemini
replaced by fakes; integration tests use the real embedding model and a real
ChromaDB in a temporary folder (about 20 seconds). Neither needs an API key.

```powershell
python -m pytest -q -m "not integration"   # unit tests only (fast)
python -m pytest -q -m integration         # real embeddings + ChromaDB
python -m pytest -q                        # everything
```

The speed and reliability fixes were written test-first (see
`tests/test_ingest_progress.py` and `tests/test_gemini_client.py`): the
tests describe the behaviour, then the code was changed until they passed.

## Project layout

```
app.py                 Streamlit UI (tabs: Documents, Financials, Ask, Graph, Chat, Report, Compare)
config.py              settings from .env, folder paths
gemini_client.py       Gemini wrapper with retry/backoff and JSON parsing
vector_store.py        local embeddings + ChromaDB
ingest.py              PDF/pasted text -> chunks -> ChromaDB, document registry
rag.py                 retrieval, system prompt, cited answers
extraction.py          financial figure extraction + validation + manual corrections
fundamentals.py        ratio formulas (pure Python)
graph_extraction.py    LLM entity/relationship extraction with evidence checks
knowledge_graph.py     NetworkX graph, JSON persistence, queries, Plotly drawing
hybrid_retrieval.py    chatbot combining RAG + graph + ratios + history
report_generator.py    Markdown company report -> reports/
comparison.py          two-company side-by-side (nice to have)
check_setup.py         environment smoke test
tests/                 pytest suite (fundamentals, ingest, extraction, graph, rag)
data/                  copies of uploaded files      chroma_db/       vector store
knowledge_base/        documents.json, figures_*.json, graph_*.json
reports/               exported reports
legacy_ollama_version/ the earlier Ollama/yfinance prototype, kept for reference
```

## Demo script (5 minutes)

1. **Add a company.** Sidebar -> "Add new company" -> name and ticker -> Create
   workspace. In *Documents*, upload the annual report PDF with its period
   (e.g. FY2024). Show the chunk count. Upload the same file again to show the
   duplicate being rejected.
2. **Figures and ratios.** *Financials* -> "Extract figures with Gemini". Point
   out the page and evidence columns, a "Not found in report" row, and the
   ratio cards with formula and inputs. Correct one value manually and watch the
   ratio update. Show the by-period chart.
3. **Ask a question.** *Ask the reports* -> e.g. "What were the main drivers of
   revenue growth?" Show the citations and the retrieved passages. Then ask
   something unrelated ("Who won the cricket world cup?") to show the
   "not found in the reports" response.
4. **Knowledge graph.** *Knowledge graph* -> "Build / extend graph". Show
   confirmed (solid) vs inferred (dashed) edges, then the Subsidiaries and
   Risk factors queries with their page evidence.
5. **Hybrid chat and report.** *Chat* -> "How can crude oil prices affect this
   company?" and the follow-up "What about its debt?". Then *Report* ->
   Generate -> show the fundamental label and download the Markdown file from
   `reports/`.

## How long things take

- Ingesting a 150-page annual report takes about 1-2 minutes on a laptop CPU:
  roughly 25 seconds reading pages with pypdf, then about a minute embedding
  1,300 chunks. The Documents tab shows a progress bar for each stage.
- Do not click Ingest twice. Streamlit cannot stop a running ingestion, so a
  second click used to start a parallel run; the app now rejects it with
  "already being ingested".
- Gemini calls take 5-30 seconds each. If the free-tier model is busy (HTTP
  503) the app retries after 3, 6, 12... seconds and, after two failures,
  switches to `GEMINI_FALLBACK_MODEL` for the rest of that call.
- Figure extraction is one call; graph building is one call per batch of six
  passages (about three calls); a report is one call.

## Limitations

- Scanned PDFs (images only) are rejected with a message; OCR is out of scope.
- Figures are only as good as the report text; always check rows marked
  "Unverified quote".
- Ratios are shown without good/bad thresholds because those depend on sector.
