"""
two_stage.py — Two-stage visual retrieval orchestration.

    Stage 1 (fast, ~500ms): mean-pool single-vector ANN prefetch → top 20
    Stage 2 (accurate, ~1s): MAX-SIM late interaction on the candidates → top 5

In production both stages run inside Qdrant in a single query (see
``qdrant_client.two_stage_search``) — that is the fast path and the default.
This module is the thin, testable orchestration layer over it, plus a local
NumPy rerank fallback for when candidates are fetched with their raw multi-vectors
(e.g. Qdrant without server-side MAX-SIM, or offline tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.indexing.qdrant_client import TOP_K_PREFETCH, TOP_K_RERANK, two_stage_search
from src.retrieval.reranker import rerank


async def retrieve(
    async_client: Any,
    query_embeddings: np.ndarray,
    top_k_prefetch: int = TOP_K_PREFETCH,
    top_k_rerank: int = TOP_K_RERANK,
    filter_doc_id: str | None = None,
) -> list[dict]:
    """Run server-side two-stage retrieval via Qdrant (the production path)."""
    return await two_stage_search(
        async_client=async_client,
        query_embeddings=query_embeddings,
        top_k_prefetch=top_k_prefetch,
        top_k_rerank=top_k_rerank,
        filter_doc_id=filter_doc_id,
    )


def rerank_locally(
    query_embeddings: np.ndarray,
    candidates: list[dict],
    top_k_rerank: int = TOP_K_RERANK,
    vectors_key: str = "vectors",
) -> list[dict]:
    """Stage 2 in pure NumPy: MAX-SIM rerank candidates carrying their patches.

    Used as a fallback when stage-1 prefetch returns candidates together with
    their full multi-vectors instead of letting Qdrant do the late interaction.
    """
    return rerank(query_embeddings, candidates, vectors_key=vectors_key, top_k=top_k_rerank)
