"""
query.py — POST /query endpoint.
Two-stage visual retrieval + Qwen2-VL answer generation.

Flow:
  1. Encode query text with ColQwen2.5 → per-token embeddings
  2. Stage 1: mean-pool prefetch on Qdrant → top 20 pages (~500ms)
  3. Stage 2: MAX-SIM rerank → top 5 pages (~1s)
  4. Qwen2-VL: read the top page images + answer the question (~3-5s)
  5. Return answer + page thumbnails (for visual source verification)
"""

import logging
import os
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.indexing.qdrant_client import two_stage_search
from src.models.colqwen import colqwen_model

router = APIRouter()
logger = logging.getLogger(__name__)

TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
ANSWERER_MODE = os.getenv("ANSWERER_MODE", "api")  # "local" | "api"

# API answerer provider. Defaults to Groq (free, OpenAI-compatible vision endpoint)
# so no OpenAI key is needed. Swap with ANSWER_PROVIDER=openai for GPT-4o-mini.
_ANSWER_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}
ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "groq").lower()


class QueryRequest(BaseModel):
    question: str
    doc_id: str | None = None  # restrict to one document (optional)
    top_k: int = TOP_K_RERANK


class RetrievedPage(BaseModel):
    doc_id: str
    filename: str
    page_num: int
    score: float
    page_b64: str  # base64 JPEG thumbnail


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_pages: list[RetrievedPage]
    latency_ms: dict  # stage1_ms, stage2_ms, answer_ms, total_ms


@router.post("/", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Ask a question about indexed documents.
    Returns a grounded answer + the page images used to answer.

    The page thumbnails let users visually verify the sources —
    this is the key differentiator from text-RAG.
    """
    t_start = time.time()

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ── Stage 1+2: Two-stage visual retrieval ─────────────────────────────────
    logger.info(f"Query: {body.question[:60]}...")

    t_retrieval_start = time.time()

    # Encode query → per-token ColQwen2.5 embeddings
    query_embeddings = colqwen_model.encode_query(body.question)

    # Two-stage search: prefetch(20) → MaxSim rerank(5)
    retrieved = await two_stage_search(
        async_client=request.app.state.qdrant_async,
        query_embeddings=query_embeddings,
        top_k_rerank=body.top_k,
        filter_doc_id=body.doc_id,
    )

    retrieval_ms = int((time.time() - t_retrieval_start) * 1000)
    logger.info(
        f"Retrieved {len(retrieved)} pages in {retrieval_ms}ms. "
        f"Top: {retrieved[0]['filename']} p{retrieved[0]['page_num']} "
        f"(score={retrieved[0]['score']})"
        if retrieved
        else "No pages found"
    )

    if not retrieved:
        return QueryResponse(
            question=body.question,
            answer="No relevant pages found. Please upload documents first.",
            retrieved_pages=[],
            latency_ms={"total_ms": int((time.time() - t_start) * 1000)},
        )

    # ── VLM Answer Generation ──────────────────────────────────────────────────
    t_answer_start = time.time()

    answer = await _generate_answer(
        question=body.question,
        retrieved_pages=retrieved,
    )

    answer_ms = int((time.time() - t_answer_start) * 1000)

    total_ms = int((time.time() - t_start) * 1000)
    logger.info(f"Total latency: {total_ms}ms (retrieval={retrieval_ms}ms, answer={answer_ms}ms)")

    return QueryResponse(
        question=body.question,
        answer=answer,
        retrieved_pages=[
            RetrievedPage(
                doc_id=p["doc_id"],
                filename=p["filename"],
                page_num=p["page_num"],
                score=p["score"],
                page_b64=p["page_b64"],
            )
            for p in retrieved
        ],
        latency_ms={
            "retrieval_ms": retrieval_ms,
            "answer_ms": answer_ms,
            "total_ms": total_ms,
        },
    )


async def _generate_answer(question: str, retrieved_pages: list) -> str:
    """
    Generate a grounded answer from the retrieved page images.
    Uses either local Qwen2-VL or GPT-4o-mini API depending on ANSWERER_MODE.
    """
    if ANSWERER_MODE == "api":
        return await _answer_via_api(question, retrieved_pages)
    else:
        return await _answer_via_local_vlm(question, retrieved_pages)


async def _answer_via_api(question: str, retrieved_pages: list) -> str:
    """
    Use a hosted vision model (Groq Llama-4 by default, OpenAI-compatible) with
    base64 page images. Dev-friendly — no local GPU needed for answering.
    """
    from openai import AsyncOpenAI

    cfg = _ANSWER_PROVIDERS.get(ANSWER_PROVIDER, _ANSWER_PROVIDERS["groq"])
    client = AsyncOpenAI(api_key=os.getenv(cfg["key_env"]), base_url=cfg["base_url"])

    # Build message with page images
    content = [
        {
            "type": "text",
            "text": (
                f"You are a helpful assistant analyzing document pages. "
                f"Answer this question based ONLY on the provided pages. "
                f"Cite specific page numbers. "
                f"If you cannot find the answer in the pages, say so honestly.\n\n"
                f"Question: {question}"
            ),
        }
    ]

    # Add top-3 page images to context (top 3 = enough, controls token cost)
    for page in retrieved_pages[:3]:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{page['page_b64']}",
                    "detail": "high",
                },
            }
        )
        content.append(
            {
                "type": "text",
                "text": f"[Page {page['page_num']} from {page['filename']}]",
            }
        )

    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0.0,
    )

    return response.choices[0].message.content


async def _answer_via_local_vlm(question: str, retrieved_pages: list) -> str:
    """
    Use local Qwen2-VL-7B-Instruct for answering (requires GPU).
    Same quality as the API approach but free after hardware cost.
    """
    import asyncio
    import functools

    # Import here to avoid loading if not needed
    from src.models.answerer import qwen_vl_model

    # Run synchronous VLM inference in a thread (non-blocking)
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None,
        functools.partial(
            qwen_vl_model.answer,
            question=question,
            pages=retrieved_pages[:3],
        ),
    )
    return answer
