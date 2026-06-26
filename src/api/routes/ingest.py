"""
ingest.py — POST /ingest endpoint.
Accepts a PDF upload and indexes all pages into Qdrant in the background.

Pattern: accept → return job_id immediately → process async in background.
This prevents HTTP timeout on large PDFs (200 pages takes ~8 min on CPU).
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile

from src.indexing.embedder import embed_pdf_bytes
from src.indexing.qdrant_client import upsert_pages

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple in-memory job tracker (use Redis in production)
_jobs: dict = {}


@router.post("/")
async def ingest_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF and start indexing in the background.

    Returns a job_id you can poll at GET /ingest/status/{job_id}.
    The PDF is fully indexed when status == "done".
    """
    if not file.filename.endswith(".pdf"):
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "status": "queued",
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": 0,
        "error": None,
    }

    background_tasks.add_task(
        _index_pdf,
        job_id=job_id,
        doc_id=doc_id,
        filename=file.filename,
        pdf_bytes=pdf_bytes,
        qdrant_sync=request.app.state.qdrant_sync,
    )

    return {
        "job_id": job_id,
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "queued",
        "message": "Indexing started. Poll /ingest/status/{job_id} for progress.",
    }


@router.get("/status/{job_id}")
async def ingest_status(job_id: str):
    """Poll indexing job status."""
    if job_id not in _jobs:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


async def _index_pdf(
    job_id: str,
    doc_id: str,
    filename: str,
    pdf_bytes: bytes,
    qdrant_sync,
):
    """Background task: render → embed → upsert to Qdrant."""
    _jobs[job_id]["status"] = "embedding"

    try:
        # 1-4. Render → encode (ColQwen) → mean-pool → thumbnails, in one call.
        logger.info(f"[{job_id}] Rendering + embedding {filename}...")
        doc = embed_pdf_bytes(pdf_bytes)
        _jobs[job_id]["pages"] = len(doc)
        logger.info(f"[{job_id}] Embedded {len(doc)} pages")

        # 5. Upsert to Qdrant
        _jobs[job_id]["status"] = "indexing"
        logger.info(f"[{job_id}] Upserting to Qdrant...")

        upsert_pages(
            client=qdrant_sync,
            doc_id=doc_id,
            filename=filename,
            page_embeddings=doc.page_embeddings,
            mean_pool_vecs=doc.mean_pool_vecs,
            page_b64s=doc.page_b64s,
        )

        _jobs[job_id]["status"] = "done"
        logger.info(f"[{job_id}] Done: {filename} ({len(doc)} pages indexed)")

    except Exception as e:
        logger.error(f"[{job_id}] Indexing failed: {e}", exc_info=True)
        _jobs[job_id].update({"status": "failed", "error": str(e)})
