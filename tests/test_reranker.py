"""Tests for MAX-SIM late interaction — the core retrieval idea, in pure NumPy."""

from __future__ import annotations

import numpy as np

from src.retrieval.reranker import max_sim, rerank


def test_identical_vectors_score_is_token_count():
    # Each query token perfectly matches an identical patch → cosine 1.0 each.
    vecs = np.eye(4, dtype=np.float32)  # 4 orthonormal vectors
    score = max_sim(vecs, vecs)
    assert score == 4.0  # 4 query tokens × best cosine 1.0


def test_max_sim_picks_best_patch_per_token():
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # Page has a patch aligned with each query token plus a noise patch.
    page = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    assert max_sim(query, page) == 2.0


def test_orthogonal_query_scores_low():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    page = np.array([[0.0, 1.0]], dtype=np.float32)  # orthogonal → cosine 0
    assert abs(max_sim(query, page)) < 1e-6


def test_rerank_orders_by_score_and_trims():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    pages = [
        {"id": "a", "vectors": np.array([[0.0, 1.0]], dtype=np.float32)},  # orthogonal
        {"id": "b", "vectors": np.array([[1.0, 0.0]], dtype=np.float32)},  # aligned
        {"id": "c", "vectors": np.array([[0.7, 0.7]], dtype=np.float32)},  # partial
    ]
    ranked = rerank(query, pages, top_k=2)
    assert [p["id"] for p in ranked] == ["b", "c"]
    assert "maxsim_score" in ranked[0]
    assert ranked[0]["maxsim_score"] >= ranked[1]["maxsim_score"]


def test_rerank_top_k_caps_output():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    pages = [{"id": str(i), "vectors": np.array([[1.0, 0.0]], dtype=np.float32)} for i in range(10)]
    assert len(rerank(query, pages, top_k=3)) == 3
