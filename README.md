# Vision RAG: Visual Document Retrieval Without OCR

> ColQwen2.5 reads charts and scanned PDFs that text-RAG cannot. No OCR. No chunking. Just pixels.

[![CI](https://github.com/fahadcs321/vision-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/fahadcs321/vision-rag/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/live%20demo-streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://vision-rag.streamlit.app)
[![Model](https://img.shields.io/badge/model-ColQwen2.5-8B5CF6)]()
[![VectorDB](https://img.shields.io/badge/vectordb-Qdrant%20MAX--SIM-6366F1)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**▶ Live demo: [vision-rag.streamlit.app](https://vision-rag.streamlit.app)** (scripted DEMO_MODE — the full pipeline needs a GPU)

---

## The Problem

Standard RAG pipelines fail on visually rich documents:

```
PDF → OCR → text chunks → embeddings → retrieval
```

OCR failure modes (all real):
- Revenue bar charts → `Q3 = 210` exists only as pixels
- Scanned reports → 60-80% OCR accuracy on photocopied docs
- Architecture diagrams → spatial structure destroyed when linearized
- Mixed-layout tables → 5 columns × 12 rows becomes unstructured numbers

ColQwen2.5 bypasses OCR entirely. It treats every page as an image and
understands it visually — charts, tables, and figures included.

---

## The Solution

```
PDF → page images (448×448) → ColQwen2.5 → ~1030 patch vectors per page
                                                        ↓
Query → ColQwen2.5 → query token vectors → MAX-SIM late interaction
                                                        ↓
                                       Retrieved pages (with thumbnails)
                                                        ↓
                                       Qwen2-VL reads the images → answer
```

`Score(query, page) = Σ_i max_j cosine(query_i, page_j)` — every query token finds
its best-matching visual patch. That is why "Q3 = 210" inside a bar chart is
retrievable: the query token for "Q3" matches the patch holding that bar.

---

## Architecture

```
Stage 1 — Fast prefetch (~500ms)
  Query → ColQwen2.5 → mean-pooled 128-D vector
  Qdrant ANN search on the "mean_pool" named vector → top 20 candidates

Stage 2 — Accurate rerank (~1s)
  Qdrant MAX-SIM on the "colqwen" multi-vector → top 5 pages
  (binary-quantized: ~32× less memory, <1% nDCG loss)

Stage 3 — VLM answer (~3-5s)
  Top-3 page images → Qwen2-VL-7B (local) or a hosted vision model (Groq/OpenAI)
  → grounded answer with page citations
```

Two-stage matters: single-stage MAX-SIM over 1000 pages × ~1030 vectors is ~1B
inner products per query (10+ s). Mean-pool prefetch then MAX-SIM on 20 candidates
keeps it at ~1.5 s.

---

## Stack (2026 production standard)

| Component | Tool |
|-----------|------|
| Visual embedding | ColQwen2.5-7B (`colpali-engine`) |
| CPU fallback | ColSmol-500M |
| Vector DB | Qdrant — multi-vector, native MAX-SIM, binary quantization |
| Two-stage retrieval | mean-pool prefetch + MAX-SIM rerank |
| VLM answering | Qwen2-VL-7B (local) / Groq Llama-4 vision (API) |
| PDF rendering | PyMuPDF |
| API | FastAPI (async, model loaded once in lifespan) |
| UI | Streamlit (page-thumbnail proof-of-sources) |

---

## Quick Start

```bash
git clone https://github.com/fahadcs321/vision-rag
cd vision-rag

# Base runtime (UI, API skeleton, PDF rendering, Qdrant client, tests)
pip install -r requirements.txt
# Heavy visual-model stack (ColQwen2.5, Qwen2-VL, torch) — for real indexing
pip install -r requirements-models.txt

docker compose up -d            # Qdrant (+ optional Langfuse)
cp .env.example .env            # set GROQ_API_KEY; COLPALI_MODEL=colsmol for CPU

uvicorn src.api.main:app --reload          # API on :8000
streamlit run app/streamlit_app.py         # UI on :8501
```

Run the offline test suite (no GPU, no keys — fakes the model and Qdrant):

```bash
pip install -r requirements-dev.txt
ruff check . && pytest
```

---

## Deploy the demo (Streamlit Cloud)

ColQwen2.5 + Qwen2-VL need a GPU, so the hosted demo runs `DEMO_MODE` — a scripted,
model-free walkthrough that shows exactly what the real system returns (generated
chart, grounded answer, page thumbnails, latency). On
[share.streamlit.io](https://share.streamlit.io): point it at `app/streamlit_app.py`,
and add `DEMO_MODE = "true"` under **Advanced settings → Secrets**. Only the slim
root `requirements.txt` installs — no torch, no models.

---

## Project structure

```
vision-rag/
├── src/
│   ├── models/
│   │   ├── colqwen.py          # ColQwen2.5 / ColSmol wrapper (pages + queries)
│   │   └── answerer.py         # Qwen2-VL-7B local answerer
│   ├── indexing/
│   │   ├── pdf_renderer.py     # PDF → 448×448 page images (PyMuPDF)
│   │   ├── embedder.py         # render → encode → mean-pool → thumbnails
│   │   └── qdrant_client.py    # multi-vector collection + two-stage MAX-SIM
│   ├── retrieval/
│   │   ├── two_stage.py        # prefetch + rerank orchestration
│   │   └── reranker.py         # MAX-SIM late interaction (pure NumPy, tested)
│   ├── api/
│   │   ├── main.py             # FastAPI + lifespan model loading
│   │   └── routes/             # ingest.py (background) · query.py (two-stage + VLM)
│   └── evaluation/
│       ├── benchmark.py        # Recall@k + MRR over the golden set
│       └── vs_text_rag.py      # side-by-side vs text-RAG baseline
├── app/
│   ├── streamlit_app.py        # UI with page-thumbnail results + DEMO_MODE
│   └── demo.py                 # self-contained scripted demo (no models)
├── tests/                      # offline suite — model + Qdrant are faked
├── docker-compose.yml          # Qdrant + Langfuse
└── data/golden/visual_qa.json  # visual QA evaluation set
```

---

## Benchmark results

| Metric | Text-RAG (OCR) | Vision RAG (ColQwen2.5) |
|--------|:--------------:|:-----------------------:|
| Chart page retrieval | ~35% | **~87%** |
| Scanned PDF accuracy | ~45% | **~82%** |
| Clean text pages | ~85% | **~89%** |
| Avg query latency | 0.8s | 1.5s |
| Memory per 1K pages | ~500 MB | **~16 MB** (binary quant) |

*Representative numbers from the ColPali/ViDoRe literature. Run
`python -m src.evaluation.benchmark` and `src/evaluation/vs_text_rag.py` to measure
your own.*

---

## Built by

**Muhammad Fahad** · BSc Computer Science
[GitHub](https://github.com/fahadcs321) · [LinkedIn](https://www.linkedin.com/in/muhammad-fahad-89a1b0358/)
