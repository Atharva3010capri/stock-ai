"""StockGraph AI - Streamlit UI.

Run with:  streamlit run app.py
Everything here is presentation; the logic lives in the other modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from packaging.version import Version

import comparison
import extraction
import fundamentals
import graph_extraction
import hybrid_retrieval
import ingest
import knowledge_graph as kg
import rag
import report_generator
import vector_store
from config import DISCLAIMER, DOCUMENT_TYPES, GEMINI_API_KEY, GEMINI_MODEL, REPORTS_DIR
from gemini_client import LLMError

# Streamlit >= 1.49 replaced use_container_width with width="stretch".
WIDE = {"width": "stretch"} if Version(st.__version__) >= Version("1.49") else {"use_container_width": True}

st.set_page_config(page_title="StockGraph AI", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("company", None)
st.session_state.setdefault("pending_companies", {})   # name -> ticker, before first upload
st.session_state.setdefault("chat_histories", {})      # company -> [{"role","content"}]


def status_writer(placeholder):
    """Return a callback the backend can use to show progress messages."""
    return lambda message: placeholder.info(message)


def show_error(error: Exception) -> None:
    st.error(str(error))


# ---------------------------------------------------------------------------
# Sidebar: company workspace
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 StockGraph AI")
    st.caption("Report-based company analysis. No market data, no price targets.")

    known = {info["company"]: info for info in ingest.list_companies()}
    names = sorted(set(known) | set(st.session_state.pending_companies), key=str.lower)
    options = names + ["➕ Add new company"]
    current = st.session_state.company
    index = options.index(current) if current in options else len(options) - 1
    choice = st.selectbox("Company workspace", options, index=index)

    if choice == "➕ Add new company":
        with st.form("new_company"):
            new_name = st.text_input("Company name", placeholder="e.g. Reliance Industries Limited")
            new_ticker = st.text_input("Ticker (optional)", placeholder="e.g. RELIANCE")
            if st.form_submit_button("Create workspace", type="primary"):
                if new_name.strip():
                    st.session_state.pending_companies[new_name.strip()] = new_ticker.strip().upper()
                    st.session_state.company = new_name.strip()
                    st.rerun()
                else:
                    st.warning("Enter a company name.")
        st.session_state.company = None if current == "➕ Add new company" else current
    else:
        st.session_state.company = choice

    company = st.session_state.company
    if company:
        ticker = known.get(company, {}).get("ticker") or st.session_state.pending_companies.get(company, "")
        docs = ingest.list_documents(company)
        st.markdown(f"**{company}**" + (f" ({ticker})" if ticker else ""))
        st.caption(f"{len(docs)} document(s) ingested")
        with st.expander("Danger zone"):
            confirm = st.checkbox("I want to delete this company's data", key="confirm_delete")
            if st.button("Delete company workspace", disabled=not confirm):
                ingest.delete_company(company)
                extraction.delete_figures(company)
                kg.delete_graph(company)
                st.session_state.pending_companies.pop(company, None)
                st.session_state.chat_histories.pop(company, None)
                st.session_state.company = None
                st.rerun()

    st.divider()
    if GEMINI_API_KEY:
        st.success(f"Gemini ready: {GEMINI_MODEL}", icon="✅")
    else:
        st.error("GEMINI_API_KEY missing. Copy .env.example to .env and add your key.", icon="🔑")
    st.caption(DISCLAIMER)

if not company:
    st.info("Create or select a company workspace in the sidebar to begin.")
    st.stop()

ticker = known.get(company, {}).get("ticker") or st.session_state.pending_companies.get(company, "")
documents = ingest.list_documents(company)
has_documents = bool(documents)

tab_docs, tab_fin, tab_ask, tab_graph, tab_chat, tab_report, tab_compare = st.tabs(
    ["📄 Documents", "📈 Financials", "❓ Ask the reports", "🕸️ Knowledge graph", "💬 Chat", "📝 Report", "⚖️ Compare"]
)

# ---------------------------------------------------------------------------
# Tab 1: Documents
# ---------------------------------------------------------------------------
with tab_docs:
    st.subheader(f"Documents for {company}")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    col_form, col_list = st.columns([1, 1])
    with col_form:
        mode = st.radio("Add a report", ["Upload PDF", "Paste text"], horizontal=True)
        with st.form("ingest_form", clear_on_submit=False):
            doc_type = st.selectbox("Document type", DOCUMENT_TYPES)
            period = st.text_input("Reporting period", placeholder="e.g. FY2024 or Q2 FY2025")
            uploaded = st.file_uploader("PDF file", type=["pdf"]) if mode == "Upload PDF" else None
            pasted = st.text_area("Report text", height=220) if mode == "Paste text" else None
            pasted_title = st.text_input("Name for this text", placeholder="e.g. Q2 press release") if mode == "Paste text" else None
            submitted = st.form_submit_button("Ingest", type="primary")
        st.caption("A 150-page annual report takes about 1-2 minutes: reading pages, then embedding every chunk on your CPU.")
        if submitted:
            status_box = st.status("Ingesting report...", expanded=True)
            bar = status_box.progress(0, text="Starting")
            labels = {"reading": "Reading pages", "chunking": "Splitting into chunks",
                      "embedding": "Embedding chunks locally", "done": "Finished"}
            spans = {"reading": (0.0, 0.3), "chunking": (0.3, 0.35), "embedding": (0.35, 1.0), "done": (1.0, 1.0)}

            def on_progress(stage: str, done: int, total: int) -> None:
                low, high = spans.get(stage, (0.0, 1.0))
                fraction = done / total if total else 0.0
                bar.progress(min(1.0, low + (high - low) * fraction), text=f"{labels.get(stage, stage)} ({done}/{total})")

            try:
                if mode == "Upload PDF":
                    if uploaded is None:
                        raise ingest.IngestionError("Choose a PDF first.")
                    summary = ingest.ingest_document(
                        company, ticker, doc_type, period, pdf_bytes=uploaded.getvalue(), pdf_name=uploaded.name,
                        progress_callback=on_progress,
                    )
                else:
                    summary = ingest.ingest_document(
                        company, ticker, doc_type, period, pasted_text=pasted or "", pasted_title=pasted_title,
                        progress_callback=on_progress,
                    )
                status_box.update(label="Ingestion complete", state="complete", expanded=False)
                st.session_state["flash"] = (
                    f"Ingested {summary['source']}: {summary['pages']} page(s), {summary['chunks']} chunks stored."
                    + (f" Warning: {summary['warning']}" if summary.get("warning") else "")
                )
                st.session_state.pending_companies.pop(company, None)
                st.rerun()
            except ingest.DuplicateDocumentError as error:
                status_box.update(label="Not ingested", state="error", expanded=False)
                st.warning(f"Duplicate upload rejected. {error}")
            except (ingest.IngestionError, vector_store.VectorStoreError) as error:
                status_box.update(label="Ingestion failed", state="error", expanded=False)
                show_error(error)
    with col_list:
        st.markdown("**Ingested documents**")
        if not documents:
            st.info("Nothing ingested yet. Upload an annual report PDF or paste report text.")
        for doc in documents:
            with st.container(border=True):
                st.markdown(f"**{doc['source']}**  \n{doc['document_type']} | {doc['reporting_period']} | "
                            f"{doc['pages']} page(s) | {doc['chunks']} chunks | added {doc['ingested_at'][:10]}")
                if st.button("Remove", key=f"rm_{doc['document_id']}"):
                    ingest.delete_document(doc["document_id"])
                    st.rerun()
        try:
            st.caption(f"Total chunks in vector store for this company: {vector_store.count_chunks(company)}")
        except vector_store.VectorStoreError as error:
            show_error(error)

# ---------------------------------------------------------------------------
# Tab 2: Financials
# ---------------------------------------------------------------------------
with tab_fin:
    st.subheader("Key figures and fundamental ratios")
    if not has_documents:
        st.info("Ingest a report first.")
    else:
        doc_options = {"All documents": None} | {f"{d['source']} ({d['reporting_period']})": d["document_id"] for d in documents}
        col_a, col_b = st.columns([2, 1])
        selected_doc = col_a.selectbox("Extract from", list(doc_options))
        if col_b.button("Extract figures with Gemini", type="primary"):
            status = st.empty()
            try:
                with st.spinner("Finding financial passages and asking Gemini for figures..."):
                    record = extraction.extract_figures(company, doc_options[selected_doc], status_callback=status_writer(status))
                status.empty()
                found = sum(1 for entries in record["figures"].values() if entries)
                st.success(f"Extraction done: {found} of {len(record['figures'])} figures found across {record['passages_used']} passages.")
                if record["discarded"]:
                    with st.expander(f"{len(record['discarded'])} value(s) discarded during validation"):
                        for item in record["discarded"]:
                            st.write("-", item)
            except (LLMError, vector_store.VectorStoreError) as error:
                status.empty()
                show_error(error)

    record = extraction.load_figures(company)
    if record:
        units = " ".join(p for p in (record.get("currency"), record.get("units")) if p) or "units not stated in extraction"
        st.caption(f"Units: **{units}** | extracted {record.get('extracted_at') or 'manually'} | model {record.get('model') or '-'}")
        rows = extraction.figures_to_rows(record)
        table = pd.DataFrame(rows).drop(columns=["key"])
        st.dataframe(table, hide_index=True, **WIDE)
        st.caption("Status 'Unverified quote' means the quote could not be matched to the page text; check it against the report.")

        with st.expander("Manually correct or add a value"):
            with st.form("manual_fix"):
                c1, c2, c3 = st.columns(3)
                field_label = c1.selectbox("Figure", list(extraction.FIELD_LABELS.values()))
                fix_period = c2.text_input("Period", placeholder="FY2024")
                fix_value = c3.text_input("Value (blank to remove)", placeholder="12345.6")
                if st.form_submit_button("Save correction"):
                    field_key = next(k for k, v in extraction.FIELD_LABELS.items() if v == field_label)
                    try:
                        value = float(fix_value.replace(",", "")) if fix_value.strip() else None
                        extraction.set_manual_value(company, field_key, fix_period, value)
                        st.rerun()
                    except ValueError:
                        st.error("Value must be a number.")

        st.markdown("### Ratios (computed in Python from the figures above)")
        ratio_rows = fundamentals.compute_ratios(record["figures"])
        cols = st.columns(3)
        for i, row in enumerate(ratio_rows):
            with cols[i % 3]:
                with st.container(border=True):
                    value = fundamentals.format_ratio_value(row, record.get("units"), record.get("currency"))
                    st.metric(row["name"], value if row["status"] == "ok" else "Skipped")
                    st.caption(row["explanation"])
                    st.code(row["formula"], language=None)
                    if row["status"] == "ok":
                        st.caption("Inputs: " + ", ".join(f"{k} = {v:,}" for k, v in row["inputs"].items()) + f" | {row['period']}")
                    else:
                        st.caption(row["reason"])
        st.caption("No universal good/bad thresholds are shown; a ratio only makes sense against the company's sector.")

        trend = fundamentals.trend_table(record["figures"])
        if trend:
            df = pd.DataFrame(trend)
            df["field"] = df["field"].map(extraction.FIELD_LABELS)
            fig = px.bar(df, x="period", y="value", color="field", barmode="group",
                         title=f"Figures by period ({units})", labels={"value": units, "period": "Period", "field": ""})
            st.plotly_chart(fig, **WIDE)
    elif has_documents:
        st.info("No figures yet. Click 'Extract figures with Gemini'.")

# ---------------------------------------------------------------------------
# Tab 3: Ask the reports (RAG)
# ---------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a question about the reports")
    question = st.text_input("Question", placeholder="What were the main drivers of revenue growth?")
    if st.button("Answer from the reports", type="primary", disabled=not has_documents):
        status = st.empty()
        with st.spinner("Searching the reports..."):
            result = rag.answer_question(question, company, status_callback=status_writer(status))
        status.empty()
        if result["status"] == "ok":
            st.markdown(result["answer"])
        elif result["status"] == "no_evidence":
            st.warning(result["answer"])
        else:
            st.error(result["answer"])
        if result["sources"]:
            st.markdown("**Sources**")
            for source in result["sources"]:
                st.write(f"- {source['citation']} (similarity {source['similarity']})")
        if result["hits"]:
            with st.expander("Retrieved passages"):
                for hit in result["hits"]:
                    st.markdown(f"**{rag.format_citation(hit['metadata'])}** (similarity {hit['similarity']:.2f})")
                    st.text(hit["text"][:1200])
    if not has_documents:
        st.info("Ingest a report first.")

# ---------------------------------------------------------------------------
# Tab 4: Knowledge graph
# ---------------------------------------------------------------------------
with tab_graph:
    st.subheader("Knowledge graph built from the reports")
    graph = kg.load_graph(company)
    stats = kg.statistics(graph)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entities", stats["entities"])
    c2.metric("Relationships", stats["relationships"])
    c3.metric("Confirmed", stats["confirmed"])
    c4.metric("Inferred", stats["inferred"])

    b1, b2 = st.columns([1, 1])
    build = b1.button("Build / extend graph with Gemini", type="primary", disabled=not has_documents)
    rebuild = b2.button("Rebuild from scratch", disabled=not has_documents)
    if build or rebuild:
        status = st.empty()
        try:
            with st.spinner("Extracting entities and relationships..."):
                summary = graph_extraction.build_graph(company, reset=rebuild, status_callback=status_writer(status))
            status.empty()
            st.success(
                f"Added {summary['relationships_added']} relationship(s) and {summary['entities_added']} entity(ies); "
                f"rejected {summary['rejected']} without matching evidence; {summary['failed_batches']} batch(es) returned bad JSON."
            )
            st.rerun()
        except (LLMError, vector_store.VectorStoreError) as error:
            status.empty()
            show_error(error)

    if stats["relationships"]:
        st.plotly_chart(kg.to_plotly_figure(graph, f"{company}: solid = confirmed, dashed = inferred"), **WIDE)

        st.markdown("### Graph queries")
        q1, q2 = st.columns([1, 2])
        query_kind = q1.radio("Show", ["Subsidiaries", "Competitors", "Segments & products", "Risk factors", "Connected to..."])
        entity_text = q2.text_input("Entity name (for 'Connected to...')", placeholder="e.g. Retail")
        if query_kind == "Subsidiaries":
            rows = kg.get_subsidiaries(graph, company)
        elif query_kind == "Competitors":
            rows = kg.get_competitors(graph, company)
        elif query_kind == "Segments & products":
            rows = kg.get_segments_and_products(graph, company)
        elif query_kind == "Risk factors":
            rows = kg.get_risk_factors(graph, company)
        else:
            rows = kg.get_connected(graph, entity_text) if entity_text.strip() else []
        if rows:
            st.dataframe(
                pd.DataFrame(rows)[["source_name", "relationship", "target_name", "target_type", "status", "confidence", "source_file", "page", "evidence"]],
                hide_index=True, **WIDE,
            )
        else:
            st.info("No matching relationships in the graph.")

        with st.expander("All relationships"):
            show = st.multiselect("Status", ["confirmed", "inferred"], default=["confirmed", "inferred"])
            all_rows = [r for r in kg.relationships(graph) if r["status"] in show]
            if all_rows:
                st.dataframe(
                    pd.DataFrame(all_rows)[["source_name", "relationship", "target_name", "status", "confidence", "source_file", "page", "evidence"]],
                    hide_index=True, **WIDE,
                )
    elif has_documents:
        st.info("Graph is empty. Click 'Build / extend graph with Gemini'.")
    else:
        st.info("Ingest a report first.")

# ---------------------------------------------------------------------------
# Tab 5: Hybrid chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader(f"Chat about {company} (reports + graph + ratios)")
    history = st.session_state.chat_histories.setdefault(company, [])
    if st.button("Clear chat"):
        st.session_state.chat_histories[company] = []
        st.rerun()
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    user_message = st.chat_input("e.g. How can crude oil prices affect this company?", disabled=not has_documents)
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            status = st.empty()
            with st.spinner("Retrieving passages, graph links and ratios..."):
                result = hybrid_retrieval.chat(user_message, company, history, status_callback=status_writer(status))
            status.empty()
            st.markdown(result["answer"])
            if result.get("sources"):
                st.caption("Sources: " + "; ".join(s["citation"] for s in result["sources"]))
            if result.get("graph_rows"):
                with st.expander("Graph evidence used"):
                    st.dataframe(pd.DataFrame(result["graph_rows"])[["source_name", "relationship", "target_name", "status", "confidence", "source_file", "page"]],
                                 hide_index=True, **WIDE)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": result["answer"]})
    if not has_documents:
        st.info("Ingest a report first.")

# ---------------------------------------------------------------------------
# Tab 6: Report
# ---------------------------------------------------------------------------
with tab_report:
    st.subheader("Company analysis report")
    st.caption("Overview, key financials, ratios, trends, segments, graph insights, findings, risks, limitations and a fundamental label (strong / weak / mixed / insufficient information).")
    if st.button("Generate report", type="primary", disabled=not has_documents):
        status = st.empty()
        try:
            with st.spinner("Assembling the report..."):
                result = report_generator.generate_report(company, status_callback=status_writer(status))
            status.empty()
            st.success(f"Saved to {result['path']} (label: {result['label']})")
            st.session_state["last_report"] = result
        except (LLMError, vector_store.VectorStoreError) as error:
            status.empty()
            show_error(error)
    last = st.session_state.get("last_report")
    if last and company.lower().replace(" ", "_") in Path(last["path"]).name.lower():
        st.download_button("Download Markdown", last["markdown"], file_name=Path(last["path"]).name, mime="text/markdown")
        st.markdown(last["markdown"])
    previous = sorted(REPORTS_DIR.glob(f"{report_generator._slug(company)}_*.md"), reverse=True)
    if previous:
        with st.expander(f"Previous reports ({len(previous)})"):
            for path in previous[:10]:
                st.download_button(path.name, path.read_text(encoding="utf-8"), file_name=path.name, key=f"dl_{path.name}")
    if not has_documents:
        st.info("Ingest a report first.")

# ---------------------------------------------------------------------------
# Tab 7: Compare
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Compare two companies (extracted figures only)")
    if len(known) < 2:
        st.info("Add and extract figures for at least two companies to compare them.")
    else:
        c1, c2 = st.columns(2)
        first = c1.selectbox("Company A", sorted(known), index=sorted(known).index(company) if company in known else 0)
        second = c2.selectbox("Company B", [n for n in sorted(known) if n != first])
        result = comparison.compare_companies(first, second)
        if result["warning"]:
            st.warning(result["warning"])
        st.markdown("**Latest figures**")
        st.dataframe(result["figures"], hide_index=True, **WIDE)
        st.markdown("**Ratios**")
        st.dataframe(result["ratios"], hide_index=True, **WIDE)

st.divider()
st.caption(f"StockGraph AI | {DISCLAIMER} | Analysis is limited to the documents you provide.")
