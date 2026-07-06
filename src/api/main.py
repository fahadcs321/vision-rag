"""
main.py — FastAPI application for Vision RAG.

Key design: models load ONCE in the lifespan context manager.
This is the 2026 FastAPI pattern — avoids cold-start latency on
every request and ensures GPU memory is allocated correctly.

Endpoints:
  POST /ingest     — upload a PDF, trigger background indexing
  POST /query      — ask a question, get answer + page thumbnails
  GET  /health     — service health + collection stats
  GET  /documents  — list all indexed documents
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.indexing.qdrant_client import (
    create_collection,
    get_async_client,
    get_client,
    get_collection_stats,
)
from src.models.colqwen import colqwen_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load ColQwen2.5 model + ensure Qdrant collection exists.
    Shutdown: clean up resources.

    Everything in 'yield' runs during the request lifecycle.
    """
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("Starting Vision RAG service...")

    # Load ColQwen model (heavy — runs once, then cached in process)
    logger.info("Loading ColPali model...")
    colqwen_model.load()
    logger.info("ColPali model ready")

    # Warm up both encode paths so the FIRST real request doesn't pay the one-time
    # kernel-compile cost — on Apple MPS a cold encode can take 30-100s. Do it once
    # here at startup instead of on the user's first upload/query.
    try:
        from PIL import Image as _Image

        logger.info("Warming up the model (compiling kernels)...")
        colqwen_model.encode_pages([_Image.new("RGB", (448, 448), "white")])
        colqwen_model.encode_query("warmup")
        logger.info("Warmup complete — first request will be fast")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Warmup skipped: {exc}")

    # Ensure Qdrant collection exists
    client = get_client()
    create_collection(client, recreate=False)

    # Store async client in app state for request handlers
    app.state.qdrant_async = get_async_client()
    app.state.qdrant_sync = client

    logger.info("Vision RAG service ready")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Shutting down Vision RAG service...")
    await app.state.qdrant_async.close()


app = FastAPI(
    title="Vision RAG API",
    description=(
        "Visual document retrieval without OCR. "
        "ColQwen2.5 + Qdrant MAX-SIM + Qwen2-VL. "
        "Reads charts and scanned PDFs that text-RAG cannot."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])


@app.get("/health")
async def health():
    """Service health + collection statistics."""
    stats = get_collection_stats(app.state.qdrant_sync)
    return {
        "status": "ok",
        "model": os.getenv("COLPALI_MODEL", "colsmol"),
        "collection": stats,
    }


@app.get("/documents")
async def list_documents():
    """List all indexed documents with page counts."""
    try:
        # Scroll through all points and aggregate by doc_id
        client = app.state.qdrant_sync
        results = client.scroll(
            collection_name=os.getenv("QDRANT_COLLECTION", "vision_rag_pages"),
            with_payload=["doc_id", "filename", "page_num"],
            limit=10000,
        )
        docs = {}
        for point in results[0]:
            did = point.payload["doc_id"]
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "filename": point.payload["filename"],
                    "pages": 0,
                }
            docs[did]["pages"] += 1

        return {"documents": list(docs.values()), "total": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
