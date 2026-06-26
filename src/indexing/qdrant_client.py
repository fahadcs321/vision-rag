"""
qdrant_client.py — Qdrant setup and operations for Vision RAG.

Key design decisions (2026 standard):
1. Named vectors: "colqwen" (multi-vector) + "mean_pool" (single vector)
   This is what enables two-stage retrieval in one collection.
2. Binary quantization on "colqwen" → 32× memory reduction
3. Scalar INT8 on "mean_pool" → 4× memory reduction
4. MAX-SIM scoring handled natively by Qdrant (no custom code needed)

Qdrant is the ONLY production vector DB with:
- No 100-vector-per-document limit
- Native MAX-SIM operator
- Named vectors (multi + mean in same point)
"""

import logging
import os
import uuid
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient, models

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "vision_rag_pages")
VECTOR_DIM = 128  # ColPali family always outputs 128-dim vectors
TOP_K_PREFETCH = int(os.getenv("TOP_K_PREFETCH", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))


def get_client() -> QdrantClient:
    """Synchronous client — for indexing."""
    return QdrantClient(url=QDRANT_URL)


def get_async_client() -> AsyncQdrantClient:
    """Async client — for query serving."""
    return AsyncQdrantClient(url=QDRANT_URL)


def create_collection(client: QdrantClient, recreate: bool = False):
    """
    Create the Qdrant collection with two named vector spaces:
    - "colqwen"   → multi-vector, binary quantization, MAX-SIM
    - "mean_pool" → single vector, scalar INT8, fast ANN prefetch
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            logger.info(f"Recreating collection: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info(f"Collection already exists: {COLLECTION_NAME}")
            return

    logger.info(f"Creating collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # Multi-vector for ColQwen patch embeddings + MAX-SIM
            "colqwen": models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True,  # keep binary quantized index in RAM
                    ),
                ),
            ),
            # Single vector for fast mean-pool prefetch
            "mean_pool": models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
        },
    )

    # Create index on payload fields for filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="filename",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    logger.info("Collection created with binary quantization + INT8 scalar quant")


def upsert_pages(
    client: QdrantClient,
    doc_id: str,
    filename: str,
    page_embeddings: list[np.ndarray],  # list of (n_patches, 128) arrays
    mean_pool_vecs: list[np.ndarray],  # list of (128,) arrays
    page_b64s: list[str],  # list of base64 page images
):
    """
    Upsert all pages of a document into Qdrant.

    Each point represents one page and stores:
    - colqwen: all patch embeddings (multi-vector, ~400-1030 vectors)
    - mean_pool: average embedding (single vector, fast search)
    - payload: doc_id, filename, page_num, page_b64 (thumbnail)
    """
    points = []

    for idx, (page_emb, mean_vec, page_b64) in enumerate(
        zip(page_embeddings, mean_pool_vecs, page_b64s, strict=False)
    ):
        page_num = idx + 1

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "colqwen": page_emb.tolist(),  # list of lists (multi-vector)
                    "mean_pool": mean_vec.tolist(),  # flat list (single vector)
                },
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "page_b64": page_b64,  # JPEG thumbnail for UI
                },
            )
        )

    # Upsert in batches of 32 to avoid payload size limits
    batch_size = 32
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Upserted pages {i + 1}–{i + len(batch)} of {filename} ({len(points)} total)")


async def two_stage_search(
    async_client: AsyncQdrantClient,
    query_embeddings: np.ndarray,  # (n_tokens, 128) — from encode_query()
    top_k_prefetch: int = TOP_K_PREFETCH,
    top_k_rerank: int = TOP_K_RERANK,
    filter_doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Two-stage retrieval:
    Stage 1 — fast prefetch using mean-pool single vector (~500ms)
    Stage 2 — accurate MAX-SIM rerank on full multi-vectors (~1s)

    Args:
        async_client: Qdrant async client
        query_embeddings: per-token query vectors (n_tokens, 128)
        top_k_prefetch: how many candidates to retrieve in stage 1
        top_k_rerank: how many to return after stage 2
        filter_doc_id: optionally restrict search to one document

    Returns:
        List of dicts with doc_id, filename, page_num, page_b64, score
    """
    # Mean-pool the query for stage-1 fast search
    query_mean = query_embeddings.mean(axis=0).tolist()

    # Optional filter by document
    search_filter = None
    if filter_doc_id:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=filter_doc_id),
                )
            ]
        )

    # Two-stage query: prefetch on mean_pool, rerank with MAX-SIM on colqwen
    results = await async_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=query_mean,
            using="mean_pool",
            limit=top_k_prefetch,
            filter=search_filter,
        ),
        query=query_embeddings.tolist(),  # multi-vector query → MAX-SIM
        using="colqwen",
        limit=top_k_rerank,
        with_payload=True,
    )

    return [
        {
            "doc_id": r.payload["doc_id"],
            "filename": r.payload["filename"],
            "page_num": r.payload["page_num"],
            "page_b64": r.payload["page_b64"],
            "score": round(r.score, 4),
        }
        for r in results.points
    ]


def get_collection_stats(client: QdrantClient) -> dict[str, Any]:
    """Return collection statistics for the API health endpoint."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}
