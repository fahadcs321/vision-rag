"""
streamlit_app.py — Vision RAG demo UI.

Features:
- PDF upload (drag & drop)
- Indexing status tracker
- Query input with plain English
- Retrieved page thumbnails (the visual proof-of-sources)
- Answer display with source citations
- Latency breakdown for demos

Run: streamlit run app/streamlit_app.py
     (FastAPI must be running on localhost:8000)
"""

import base64
import io
import os
import time

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

# DEMO_MODE renders a self-contained scripted walkthrough (no API, no models) so
# the hosted Streamlit Cloud demo works without a GPU. Unset it to run for real
# against a local FastAPI + Qdrant.
DEMO_MODE = os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")
if DEMO_MODE:
    from demo import DEMO_DOCUMENTS, demo_result


def fetch_documents():
    """Indexed documents — canned in demo mode, live from the API otherwise."""
    if DEMO_MODE:
        return DEMO_DOCUMENTS, None
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=5)
        if resp.ok:
            return resp.json().get("documents", []), None
        return [], f"API error {resp.status_code}"
    except Exception:
        return [], "offline"


def run_query(question: str, top_k: int = 5) -> dict:
    """Run a query — scripted in demo mode, live two-stage retrieval otherwise."""
    if DEMO_MODE:
        return demo_result(question)
    resp = requests.post(
        f"{API_URL}/query/", json={"question": question, "top_k": top_k}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


st.set_page_config(
    page_title="Vision RAG — Reads Charts, Not Just Text",
    page_icon="👁️",
    layout="wide",
)

# ── Custom CSS (dark theme matching the portfolio) ─────────────────────────────
st.markdown(
    """
<style>
    .stApp { background-color: #0D1117; color: #C9D1D9; }
    .main-title { color: #A855F7; font-size: 2rem; font-weight: 700; }
    .sub-title { color: #8B949E; font-size: 0.9rem; }
    .metric-box {
        background: #161B22; border: 1px solid #30363D;
        border-radius: 8px; padding: 12px 16px;
    }
    .page-card {
        background: #161B22; border: 1px solid #8B5CF6;
        border-radius: 8px; padding: 8px;
        text-align: center;
    }
    .answer-box {
        background: #161B22; border-left: 3px solid #8B5CF6;
        padding: 16px 20px; border-radius: 8px;
        line-height: 1.7;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">👁️ Vision RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Reads charts, scanned PDFs, and figures that text-RAG cannot. '
    "ColQwen2.5 + Qdrant MAX-SIM + Qwen2-VL. No OCR required.</div>",
    unsafe_allow_html=True,
)
if DEMO_MODE:
    st.info(
        "🎬 **Demo mode** — this hosted version runs a scripted walkthrough "
        "(the full pipeline needs ColQwen2.5 + Qwen2-VL + a GPU). "
        "Run it locally for live PDF indexing and retrieval.",
        icon="ℹ️",
    )
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_query, tab_upload, tab_compare = st.tabs(
    [
        "🔍 Ask a Question",
        "📄 Upload PDF",
        "⚖️ vs Text-RAG",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Query
# ═══════════════════════════════════════════════════════════════════════════════
with tab_query:
    col_input, col_docs = st.columns([3, 1])

    with col_docs:
        st.caption("Indexed documents")
        docs, doc_err = fetch_documents()
        if docs:
            for d in docs:
                st.markdown(
                    f"📄 **{d['filename']}** ({d['pages']} pages)",
                    help=f"doc_id: {d['doc_id']}",
                )
        elif doc_err == "offline":
            st.warning("FastAPI not running.\n`uvicorn src.api.main:app --reload`")
        else:
            st.info("No documents yet. Upload a PDF first.")

    with col_input:
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What was the Q3 revenue shown in the chart?",
        )
        col_btn, col_topk = st.columns([2, 1])
        with col_btn:
            ask_btn = st.button("Ask →", type="primary", use_container_width=True)
        with col_topk:
            top_k = st.slider("Pages to retrieve", 1, 10, 5, label_visibility="collapsed")

        # Sample questions
        st.caption("Try asking:")
        samples = [
            "What is the revenue growth shown in the chart?",
            "Summarize the key findings on page 3",
            "What does the architecture diagram show?",
            "What are the quarterly profit margins?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(samples):
            if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
                question = q
                ask_btn = True

    if ask_btn and question:
        with st.spinner("Stage 1: semantic prefetch → Stage 2: MaxSim rerank → VLM answer..."):
            try:
                result = run_query(question, top_k=top_k)
                if result:
                    # ── Answer ──────────────────────────────────────────────
                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Latency metrics ─────────────────────────────────────
                    lat = result.get("latency_ms", {})
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Retrieval", f"{lat.get('retrieval_ms', 0)}ms")
                    m2.metric("VLM Answer", f"{lat.get('answer_ms', 0)}ms")
                    m3.metric("Total", f"{lat.get('total_ms', 0)}ms")

                    # ── Retrieved pages (the key differentiator) ────────────
                    pages = result.get("retrieved_pages", [])
                    if pages:
                        st.markdown(f"### Source Pages ({len(pages)} retrieved)")
                        st.caption(
                            "These are the actual document pages the system READ "
                            "to answer your question — including charts and figures."
                        )
                        cols = st.columns(min(len(pages), 5))
                        for i, page in enumerate(pages):
                            with cols[i]:
                                img = Image.open(io.BytesIO(base64.b64decode(page["page_b64"])))
                                st.image(img, use_container_width=True)
                                st.caption(
                                    f"**{page['filename']}**\n"
                                    f"Page {page['page_num']} · Score {page['score']:.3f}"
                                )

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Run: `uvicorn src.api.main:app --reload`  (or set DEMO_MODE=true)"
                )
            except Exception as exc:
                st.error(f"Query failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload a PDF to index")
    st.info(
        "Annual reports, research papers, scanned documents, presentations — "
        "anything with charts or figures text-RAG can't read."
    )

    if DEMO_MODE:
        st.warning(
            "Upload is disabled in the hosted demo — indexing needs ColQwen2.5 + a "
            "GPU. Run the project locally (`uvicorn src.api.main:app` + `docker "
            "compose up`) to index your own PDFs. The **Ask a Question** tab shows a "
            "real scripted result."
        )

    uploaded = None if DEMO_MODE else st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        st.markdown(f"**Selected:** {uploaded.name} ({len(uploaded.getvalue()):,} bytes)")

        if st.button("Index PDF", type="primary"):
            with st.spinner("Uploading and starting indexing..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest/",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=30,
                    )

                    if resp.ok:
                        job = resp.json()
                        st.success(f"Indexing started! Job ID: `{job['job_id']}`")

                        # Poll for completion
                        status_placeholder = st.empty()
                        for _ in range(120):  # max 2 min polling
                            time.sleep(3)
                            status_resp = requests.get(f"{API_URL}/ingest/status/{job['job_id']}")
                            if status_resp.ok:
                                s = status_resp.json()
                                status_placeholder.info(
                                    f"Status: **{s['status']}** | Pages: {s.get('pages', '?')}"
                                )
                                if s["status"] == "done":
                                    status_placeholder.success(
                                        f"✅ Indexed {s['pages']} pages from {s['filename']}"
                                    )
                                    break
                                elif s["status"] == "failed":
                                    status_placeholder.error(f"❌ Failed: {s.get('error')}")
                                    break
                    else:
                        st.error(f"Upload failed: {resp.text}")

                except requests.ConnectionError:
                    st.error("Cannot connect to FastAPI.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: vs Text-RAG comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Vision RAG vs Text-RAG: Side-by-Side")
    st.markdown(
        "Ask a question about a chart or figure. "
        "Vision RAG reads it visually; Text-RAG fails silently."
    )

    compare_q = st.text_input(
        "Comparison question:",
        value="What was the Q3 revenue shown in the chart?",
        key="compare_input",
    )

    if st.button("Compare →", key="compare_btn"):
        col_vision, col_text = st.columns(2)

        with col_vision:
            st.markdown("#### 👁️ Vision RAG (ColQwen2.5)")
            with st.spinner("Running visual retrieval..."):
                try:
                    r = run_query(compare_q)
                    st.success(r["answer"])
                    if r.get("retrieved_pages"):
                        img = Image.open(
                            io.BytesIO(base64.b64decode(r["retrieved_pages"][0]["page_b64"]))
                        )
                        st.image(img, caption="Page read visually", use_container_width=True)
                    st.caption(f"Total: {r['latency_ms'].get('total_ms', '?')}ms")
                except Exception as e:
                    st.error(str(e))

        with col_text:
            st.markdown("#### 📄 Text-RAG (OCR-based)")
            st.warning(
                "Text-RAG would fail here. A bar chart with 'Q3 = 210' "
                "exists only as pixels — OCR either misreads it or "
                "produces garbled text that embeds as noise. "
                "The retrieved chunks have no signal about this number."
            )
            st.caption("~35% chart retrieval accuracy (vs ~87% for Vision RAG)")

st.divider()
st.caption("Muhammad Fahad · MSc CS @ ITU Copenhagen · github.com/fahadcs321")
