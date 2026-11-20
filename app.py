"""
app.py  –  DocMind RAG System  |  Modern Intelligent Document Q&A
"""

import hashlib
import os
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import process_pdf
from vector_store import VectorStore
from rag_engine import RAGEngine

load_dotenv()

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI • RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme & UI Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global resets & base theme ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.06) 0%, transparent 60%),
                #090d16 !important;
    color: #f1f5f9 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 1060px !important;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.2rem 0.5rem 1.6rem 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.4rem;
}

.brand-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35);
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 40%, #c7d2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

.brand-sub {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.panel-heading {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.badge-active {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}
.badge-missing {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
}

/* ── Hero Header ── */
.hero-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 50%, rgba(14, 165, 233, 0.05) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

.hero-box::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #38bdf8);
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #ffffff 40%, #c7d2fe 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
}

.hero-desc {
    color: #94a3b8;
    font-size: 0.96rem;
    max-width: 680px;
    line-height: 1.6;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.2rem;
}

.tech-chip {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.74rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 1.8rem;
}

.metric-tile {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-tile:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.metric-num {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
    font-weight: 600;
}

/* ── Section Titles ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.6rem 0 1rem 0;
}

.section-bullet {
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
}

.section-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.3px;
}

/* ── Chat Messages Styling ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.08)) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}

/* Source citations card */
.source-box {
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.82rem;
    line-height: 1.55;
    color: #cbd5e1;
}

.source-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #a5b4fc;
    margin-bottom: 0.4rem;
}

.score-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
}

/* Empty state banner */
.empty-hero {
    text-align: center;
    padding: 3.5rem 2rem;
    border: 1.5px dashed rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    margin: 1.2rem 0;
    background: rgba(255, 255, 255, 0.015);
}

.empty-icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
    opacity: 0.85;
}

.empty-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}

.empty-sub {
    font-size: 0.88rem;
    color: #64748b;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Custom buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.2s ease !important;
}

/* File uploader polish */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(99, 102, 241, 0.04) !important;
    border: 1.5px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99, 102, 241, 0.6) !important;
    background: rgba(99, 102, 241, 0.08) !important;
}

/* Expander custom styling */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
}

/* Chat input sticky bar */
[data-testid="stChatInput"] {
    background: rgba(13, 19, 34, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
}

[data-testid="stChatInput"] textarea {
    color: #f1f5f9 !important;
    font-size: 0.94rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── State Initialization ─────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "vs": None,
        "rag": None,
        "doc_id": None,
        "doc_name": None,
        "num_chunks": 0,
        "raw_text": "",
        "chat_history": [],
        "init_error": None,
        "quick_prompt": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_session()


# ─── Cached Vector Store & RAG Engine ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_vs():
    return VectorStore()

@st.cache_resource(show_spinner=False)
def get_cached_rag(_vs):
    return RAGEngine(_vs)


def _compute_doc_id(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Branding
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-title">DocMind AI</div>
            <div class="brand-sub">RAG QA Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API Keys Health
    cohere_key_set   = bool(os.getenv("COHERE_API_KEY"))
    pinecone_key_set = bool(os.getenv("PINECONE_API_KEY"))

    st.markdown("""
    <div class="glass-panel">
        <div class="panel-heading">⚡ Connected Services</div>
    """, unsafe_allow_html=True)

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if cohere_key_set:
            st.markdown('<div class="badge-pill badge-active">✓ Cohere</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-pill badge-missing">✗ Cohere</div>', unsafe_allow_html=True)
    with col_k2:
        if pinecone_key_set:
            st.markdown('<div class="badge-pill badge-active">✓ Pinecone</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-pill badge-missing">✗ Pinecone</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Active Document Details
    if st.session_state.doc_name:
        wc = len(st.session_state.raw_text.split())
        st.markdown(f"""
        <div class="glass-panel">
            <div class="panel-heading">📄 Active Document</div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.6rem; word-break: break-all;">
                📎 {st.session_state.doc_name}
            </div>
            <div style="display: flex; gap: 8px;">
                <div style="flex: 1; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #818cf8;">{st.session_state.num_chunks}</div>
                    <div style="font-size: 0.62rem; color: #64748b; text-transform: uppercase;">Chunks</div>
                </div>
                <div style="flex: 1; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #c084fc;">{wc:,}</div>
                    <div style="font-size: 0.62rem; color: #64748b; text-transform: uppercase;">Words</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Unload Document", use_container_width=True):
            try:
                st.session_state.vs.delete_namespace(st.session_state.doc_id)
            except Exception:
                pass
            for field in ["doc_id", "doc_name", "num_chunks", "raw_text", "chat_history"]:
                st.session_state[field] = [] if field == "chat_history" else (0 if field == "num_chunks" else None)
            st.rerun()

    # RAG Hyperparameters
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-heading">🎛️ Retrieval Settings</div>
    """, unsafe_allow_html=True)

    top_k = st.slider(
        "Top-K Excerpts",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of relevant chunks passed to Cohere for answer generation",
    )

    temperature = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower values produce more deterministic, factual answers",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Actions: Export & Clear
    if st.session_state.chat_history:
        st.markdown('<div class="panel-heading" style="padding: 0 0.5rem;">📁 Session Tools</div>', unsafe_allow_html=True)
        # Format chat history as markdown for export
        export_text = f"# QA Session: {st.session_state.doc_name or 'Document'}\n\n"
        for turn in st.session_state.chat_history:
            sender = "User" if turn["role"] == "user" else "DocMind AI"
            export_text += f"### {sender}\n{turn['content']}\n\n"

        st.download_button(
            label="📥 Export Chat (Markdown)",
            data=export_text,
            file_name=f"DocMind_Chat_{st.session_state.doc_id or 'export'}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        '<div style="margin-top: 1.5rem; padding: 0 0.5rem; font-size: 0.7rem; color: #475569; line-height: 1.5;">'
        'Embeddings: <code>embed-english-v3.0</code><br>'
        'LLM: <code>command-a-03-2025</code><br>'
        'Vectors: Pinecone Serverless'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

# Hero Header
st.markdown("""
<div class="hero-box">
    <div class="hero-tag">✦ High-Precision Document Intelligence</div>
    <div class="hero-title">RAG-Based Document Q&amp;A</div>
    <div class="hero-desc">
        Upload your PDF documents and ask context-aware questions. Every answer is grounded in your document's text using Cohere semantic embeddings and Pinecone vector retrieval.
    </div>
    <div class="hero-pills">
        <span class="tech-chip">⚡ Cohere Embeddings v3</span>
        <span class="tech-chip">🌲 Pinecone Vector DB</span>
        <span class="tech-chip">🎯 Zero Hallucinations</span>
        <span class="tech-chip">📄 Source Citations</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Step 1: Document Upload ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-bullet">1</div>
    <div class="section-name">Upload Document</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your PDF file here",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload research papers, technical reports, contracts, manuals or books",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    doc_id = _compute_doc_id(file_bytes)

    if doc_id != st.session_state.doc_id:
        st.session_state.chat_history = []
        with st.spinner("⚡ Extracting text, computing Cohere embeddings, and indexing in Pinecone..."):
            try:
                vs = get_cached_vs()
                rag = get_cached_rag(vs)
                st.session_state.vs = vs
                st.session_state.rag = rag

                raw_text, chunks = process_pdf(file_bytes)

                if vs.namespace_exists(doc_id):
                    vs.delete_namespace(doc_id)

                n_upserted = vs.upsert_chunks(chunks, doc_id)

                st.session_state.doc_id = doc_id
                st.session_state.doc_name = uploaded_file.name
                st.session_state.num_chunks = n_upserted
                st.session_state.raw_text = raw_text
                st.session_state.init_error = None

            except Exception as e:
                st.session_state.init_error = str(e)

        if st.session_state.init_error:
            st.error(f"❌ Failed to process document: {st.session_state.init_error}")
        else:
            st.success(f"🎉 **{uploaded_file.name}** processed successfully! Stored **{st.session_state.num_chunks} vector chunks** in Pinecone.")


# ── Analytics Tiles ───────────────────────────────────────────────────────────
if st.session_state.doc_name:
    total_words = len(st.session_state.raw_text.split())
    num_questions = len([m for m in st.session_state.chat_history if m["role"] == "user"])

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="metric-num">{st.session_state.num_chunks}</div>
            <div class="metric-label">Chunks Indexed</div>
        </div>
        <div class="metric-tile">
            <div class="metric-num">{total_words:,}</div>
            <div class="metric-label">Words Extracted</div>
        </div>
        <div class="metric-tile">
            <div class="metric-num">{num_questions}</div>
            <div class="metric-label">Questions Answered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Step 2: Interactive Q&A ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-bullet">2</div>
    <div class="section-name">Ask Document Questions</div>
</div>
""", unsafe_allow_html=True)

# Quick Prompts if document is loaded
if st.session_state.doc_id:
    st.markdown('<div style="font-size: 0.74rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Suggested Queries:</div>', unsafe_allow_html=True)
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        if st.button("📋 Summarize key findings", use_container_width=True):
            st.session_state.quick_prompt = "Summarize the key findings, conclusions, and takeaways from this document."
    with c_p2:
        if st.button("🎯 What is the main objective?", use_container_width=True):
            st.session_state.quick_prompt = "What is the main problem statement and primary objective discussed in this document?"
    with c_p3:
        if st.button("📊 List critical metrics & data", use_container_width=True):
            st.session_state.quick_prompt = "List all important experimental metrics, quantitative results, and data values mentioned in the document."

# Render Chat History
if not st.session_state.doc_id:
    st.markdown("""
    <div class="empty-hero">
        <div class="empty-icon">📁</div>
        <div class="empty-title">No Document Loaded</div>
        <div class="empty-sub">Please upload a PDF document above to start semantic question answering.</div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-hero">
        <div class="empty-icon">💬</div>
        <div class="empty-title">Ready for Your Questions</div>
        <div class="empty-sub">Ask any question or click one of the suggested query buttons above to begin exploring your document.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(turn["content"])

                # Source Citations Expander
                sources = turn.get("sources", [])
                if sources:
                    with st.expander(f"📌 View Retrieved Source Excerpts ({len(sources)})", expanded=False):
                        for idx, src in enumerate(sources, 1):
                            score_label = f"Match: {src['score'] * 100:.1f}%" if "score" in src else "Relevant"
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-meta">
                                    <span>Chunk #{src.get('chunk_id', idx)}</span>
                                    <span class="score-badge">{score_label}</span>
                                </div>
                                {src.get('text', '')}
                            </div>
                            """, unsafe_allow_html=True)


# ── Question Input Processing ─────────────────────────────────────────────────
active_prompt = None

# Check if a quick-prompt button was pressed
if st.session_state.quick_prompt:
    active_prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None
else:
    # Native chat input
    if st.session_state.doc_id:
        active_prompt = st.chat_input("Ask anything about your document...")

# Handle query execution
if active_prompt:
    # 1. Append user prompt
    st.session_state.chat_history.append({"role": "user", "content": active_prompt})

    # Render immediate turn in UI
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)

    # 2. Answer generation
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🔍 Retrieving excerpts and generating answer with Cohere..."):
            try:
                res = st.session_state.rag.answer(
                    question=active_prompt,
                    doc_id=st.session_state.doc_id,
                    top_k=top_k,
                    temperature=temperature,
                )
                answer_text = res["answer"]
                retrieved_sources = res["source_chunks"]
            except Exception as e:
                answer_text = f"⚠️ An error occurred while generating the response: `{e}`"
                retrieved_sources = []

            st.markdown(answer_text)

            if retrieved_sources:
                with st.expander(f"📌 View Retrieved Source Excerpts ({len(retrieved_sources)})", expanded=False):
                    for idx, src in enumerate(retrieved_sources, 1):
                        score_label = f"Match: {src['score'] * 100:.1f}%" if "score" in src else "Relevant"
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-meta">
                                <span>Chunk #{src.get('chunk_id', idx)}</span>
                                <span class="score-badge">{score_label}</span>
                            </div>
                            {src.get('text', '')}
                        </div>
                        """, unsafe_allow_html=True)

    # 3. Store assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": retrieved_sources,
    })
    st.rerun()
