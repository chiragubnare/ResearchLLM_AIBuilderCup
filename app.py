import hashlib
import html
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import pymupdf

from ingestion.pipeline import build_chunks
from retrieval.indexing import index_chunks, restore_vector_index
from retrieval.retriever import retrieve
from retrieval.answering import stream_answer, validate_citations
from shared.storage import get_storage


load_dotenv()

st.set_page_config(
    page_title="PaperPilot | Research Paper Chat",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#1b2635; --muted:#6f7a87; --teal:#147d77; --teal-soft:#e8f5f2; --paper:#fbfaf7; --line:#e7e8e4; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink) !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#f3f5f2; border-right:1px solid var(--line); color:var(--ink) !important; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
.block-container { max-width:1120px; padding-top:2.4rem; padding-bottom:5rem; }
h1,h2,h3, .stMarkdown, label, p { font-family:'Manrope', 'DM Sans', sans-serif; letter-spacing:-.015em; color:var(--ink) !important; }
h1 { font-size:2.45rem !important; font-weight:800 !important; margin-bottom:.1rem !important; }
.stSidebar h3, .stSidebar p, .stSidebar label, .stSidebar [data-testid="stCaptionContainer"] { color:var(--ink) !important; }
.eyebrow { color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.65rem; }
.subhead { color:var(--muted); font-size:1rem; margin-top:.25rem; }
.brand-mark { width:46px; height:46px; border-radius:15px; display:grid; place-items:center; background:var(--teal); color:white; font-size:1.35rem; box-shadow:0 8px 22px #147d7729; }
.hero { border:1px solid var(--line); border-radius:22px; padding:2rem 2.2rem; background:linear-gradient(120deg,#fff 5%,#f0f7f4 100%); margin:1.6rem 0 1.25rem; }
.hero-kicker { color:var(--teal); font-weight:700; font-size:.76rem; letter-spacing:.1em; text-transform:uppercase; }
.hero-title { font-family:'Manrope', sans-serif; color:var(--ink) !important; font-size:1.65rem; font-weight:800; letter-spacing:-.04em; margin:.55rem 0; }
.hero-copy { max-width:680px; color:var(--muted); line-height:1.65; }
.steps-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; }
.step-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }
.step-num { color:var(--teal); font-size:.75rem; font-weight:700; letter-spacing:.1em; }
.step-label { color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.05rem; font-weight:700; margin-top:.5rem; }
.status-pill { display:inline-flex; align-items:center; gap:.45rem; background:var(--teal-soft); color:#126e69; border-radius:999px; padding:.45rem .75rem; font-size:.78rem; font-weight:700; }
.stButton > button { border-radius:10px; font-weight:600; min-height:2.7rem; }
.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
.stButton > button[kind="primary"]:hover { background:#106d68; border-color:#106d68; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:16px; padding:.75rem 1rem; background:#fff; }
[data-testid="stChatInput"] { border-color:#d8dedb; border-radius:14px; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:var(--ink) !important; }
[data-testid="stChatInput"] textarea { color:var(--ink) !important; background:#fff !important; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:white; }
.sidebar-note { color:var(--muted); font-size:.82rem; line-height:1.5; }
</style>
""", unsafe_allow_html=True)


def render_pdf_page(pdf_bytes: bytes, page_number: int) -> bytes:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = document[page_number - 1]
        return page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False).tobytes("png")
    finally:
        document.close()


def show_sources(evidence: list, storage) -> None:
    with st.expander("Sources from the paper"):
        for item in evidence:
            chunk = item.chunk
            title = f"Page {chunk.page_number} · {chunk.chunk_type.replace('_', ' ').title()}"
            if chunk.source_ref:
                title += f" · {chunk.source_ref}"
            st.markdown(f"**{title}**")
            if chunk.chunk_type == "table_json":
                st.json(json.loads(chunk.content))
            else:
                st.write(chunk.content)
            if chunk.image_key:
                try:
                    st.image(storage.get_bytes(chunk.image_key), caption=chunk.source_ref or title)
                except Exception as exc:
                    st.caption(f"Figure preview unavailable: {exc}")


header_icon, header_copy, header_status = st.columns([0.7, 5.2, 2.1], vertical_alignment="center")
with header_icon:
    st.markdown('<div class="brand-mark">✳</div>', unsafe_allow_html=True)
with header_copy:
    st.markdown('<div class="eyebrow">Research paper workspace</div>', unsafe_allow_html=True)
    st.markdown("# PaperPilot")
    st.markdown('<div class="subhead">Read papers faster. Get answers with evidence and page citations.</div>', unsafe_allow_html=True)
with header_status:
    if st.session_state.get("document_name"):
        st.markdown('<div class="status-pill">● Paper indexed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill">● Ready for a paper</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Your library")
    st.markdown('<div class="sidebar-note">Add a research PDF to create a searchable index scoped to that document.</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Research paper (PDF)", type=["pdf"])
    if st.button("✦  Index this paper", type="primary", disabled=uploaded_pdf is None, use_container_width=True):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Set GEMINI_API_KEY in a .env file before indexing.")
        else:
            pdf_bytes = uploaded_pdf.getvalue()
            doc_id = hashlib.sha256(pdf_bytes).hexdigest()
            storage = get_storage()
            pdf_key = f"raw_pdfs/{doc_id}/{Path(uploaded_pdf.name).name}"
            try:
                with st.spinner("Reading the paper and creating its search index…"):
                    storage.put_bytes(pdf_key, pdf_bytes, content_type="application/pdf")
                    pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    try:
                        page_count = len(pdf_document)
                    finally:
                        pdf_document.close()
                    if restore_vector_index(doc_id, storage=storage):
                        st.session_state.doc_id = doc_id
                        st.session_state.document_key = pdf_key
                        st.session_state.document_name = uploaded_pdf.name
                        st.session_state.page_count = page_count
                        st.session_state.messages = []
                        st.success(f"Restored the existing index for {uploaded_pdf.name}.")
                    else:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
                            pdf_file.write(pdf_bytes)
                            pdf_path = pdf_file.name
                        try:
                            chunks = build_chunks(pdf_path, doc_id=doc_id, storage=storage)
                        finally:
                            os.unlink(pdf_path)
                        if not chunks:
                            st.error("No searchable text, tables, or figures were found in this PDF.")
                        else:
                            index_chunks(chunks, doc_id=doc_id, replace=True, storage=storage)
                            st.session_state.doc_id = doc_id
                            st.session_state.document_key = pdf_key
                            st.session_state.document_name = uploaded_pdf.name
                            st.session_state.page_count = page_count
                            st.session_state.messages = []
                            st.success(f"Indexed {len(chunks)} sections from {uploaded_pdf.name}.")
            except Exception as exc:
                st.error(f"Could not index the PDF: {exc}")

    if st.session_state.get("document_name"):
        st.markdown(f"**Current paper**  \n{st.session_state.document_name}")
    st.divider()
    storage_label = "Google Cloud Storage" if os.getenv("GCS_BUCKET") else "local storage"
    st.markdown(f'<div class="sidebar-note">Files and vector snapshots use {storage_label}. Configure GCS_BUCKET to enable GCS.</div>', unsafe_allow_html=True)

if not os.getenv("GEMINI_API_KEY"):
    st.warning("Add GEMINI_API_KEY to your .env file to enable indexing and chat.")

if not st.session_state.get("doc_id"):
    st.markdown('''
    <section class="hero">
      <div class="hero-kicker">Your AI reading companion</div>
      <div class="hero-title">Make your next paper easier to explore.</div>
      <div class="hero-copy">Upload a PDF from the sidebar. PaperPilot will organize its text, tables, and figures, then answer questions with references back to the source pages.</div>
    </section>
    ''', unsafe_allow_html=True)
    st.markdown('''
    <div class="steps-grid">
      <div class="step-card"><div class="step-num">01</div><div class="step-label">Upload a paper</div></div>
      <div class="step-card"><div class="step-num">02</div><div class="step-label">Build the index</div></div>
      <div class="step-card"><div class="step-num">03</div><div class="step-label">Ask with citations</div></div>
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    current_name = html.escape(st.session_state.get("document_name", "Research paper"))
    st.markdown(f'<div class="status-pill">Currently exploring · {current_name}</div>', unsafe_allow_html=True)
    with st.expander("Preview paper pages", expanded=False):
        page_count = st.session_state.get("page_count", 1)
        preview_page = st.slider("Page", min_value=1, max_value=page_count, value=1, key=f"page_preview_{st.session_state.doc_id}")
        try:
            pdf_bytes = get_storage().get_bytes(st.session_state.document_key)
            st.image(render_pdf_page(pdf_bytes, preview_page), caption=f"Page {preview_page} of {page_count}", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render this PDF page: {exc}")

messages = st.session_state.setdefault("messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"], get_storage())

question = st.chat_input(
    "Ask a question about the indexed paper…",
    disabled=not (st.session_state.get("doc_id") and os.getenv("GEMINI_API_KEY")),
)
if question:
    st.chat_message("user").markdown(question)
    messages.append({"role": "user", "content": question})
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the paper…"):
                evidence = retrieve(question, doc_id=st.session_state.doc_id, top_k=5)
            if evidence:
                output = st.empty()
                with output.container():
                    draft = st.write_stream(stream_answer(question, evidence, messages[:-1]))
                response, cited_evidence = validate_citations(draft, evidence)
                output.markdown(response)
                if cited_evidence:
                    show_sources(cited_evidence, get_storage())
            else:
                response = "I couldn't find relevant information in the indexed paper."
                cited_evidence = []
                st.markdown(response)
        messages.append({"role": "assistant", "content": response, "sources": cited_evidence})
    except Exception as exc:
        st.error(f"I couldn't answer that question: {exc}")
