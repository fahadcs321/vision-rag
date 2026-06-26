"""
reranker.py — ColBERT-style MAX-SIM late interaction, in pure NumPy.

Qdrant computes MAX-SIM natively in production (see qdrant_client.two_stage_search),
but the same scoring runs here for two reasons:
  1. A local rerank fallback when Qdrant returns multi-vectors directly.
  2. It is dependency-light and fully unit-testable — no GPU, no DB — so the
     core retrieval idea is verified in CI.

MAX-SIM(query, page) = Σ_i  max_j  cosine(query_i, page_j)

Every query token i finds its single best-matching page patch j; the score is the
sum of those best matches. That is exactly why "Q3 = 210" inside a bar chart is
retrievable: the query token for "Q3" matches the visual patch holding that bar.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization so dot products equal cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def max_sim(query_vectors: np.ndarray, page_vectors: np.ndarray) -> float:
    """Late-interaction MAX-SIM score between a query and one page.

    Args:
        query_vectors: (n_query_tokens, dim)
        page_vectors:  (n_patches, dim)

    Returns:
        Sum over query tokens of the best cosine similarity to any page patch.
    """
    q = _l2_normalize(query_vectors)
    p = _l2_normalize(page_vectors)
    # (n_query_tokens, n_patches) cosine similarity matrix.
    sim = q @ p.T
    # Each query token takes its best-matching patch, then sum across tokens.
    return float(sim.max(axis=1).sum())


def rerank(
    query_vectors: np.ndarray,
    pages: list[dict],
    vectors_key: str = "vectors",
    top_k: int = 5,
) -> list[dict]:
    """Re-score candidate pages by MAX-SIM and return the best ``top_k``.

    Args:
        query_vectors: (n_query_tokens, dim) query embeddings.
        pages: candidate dicts, each holding its patch matrix under ``vectors_key``.
        vectors_key: key in each page dict holding an (n_patches, dim) array.
        top_k: how many top pages to return.

    Returns:
        The top_k pages, each with an added ``maxsim_score``, best first.
    """
    scored = []
    for page in pages:
        score = max_sim(query_vectors, np.asarray(page[vectors_key]))
        scored.append({**page, "maxsim_score": round(score, 4)})
    scored.sort(key=lambda p: p["maxsim_score"], reverse=True)
    return scored[:top_k]
