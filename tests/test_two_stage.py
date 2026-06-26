"""Tests for the local two-stage rerank fallback (stage 2 in NumPy)."""

from __future__ import annotations

import numpy as np

from src.retrieval.two_stage import rerank_locally


def test_rerank_locally_promotes_best_match():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = [
        {"page_num": 1, "vectors": np.array([[0.0, 1.0]], dtype=np.float32)},
        {"page_num": 2, "vectors": np.array([[1.0, 0.0]], dtype=np.float32)},
    ]
    out = rerank_locally(query, candidates, top_k_rerank=1)
    assert len(out) == 1
    assert out[0]["page_num"] == 2


def test_rerank_locally_preserves_payload():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = [
        {
            "page_num": 7,
            "filename": "report.pdf",
            "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
        }
    ]
    out = rerank_locally(query, candidates, top_k_rerank=5)
    assert out[0]["filename"] == "report.pdf"
    assert out[0]["page_num"] == 7
