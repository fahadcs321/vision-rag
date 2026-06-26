"""Tests for the retrieval-metric math (no network, no API)."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark import aggregate, recall_at_k, reciprocal_rank


def test_recall_at_k_hits_and_misses():
    assert recall_at_k([3, 1, 7], expected_page=1, k=5) == 1.0
    assert recall_at_k([3, 1, 7], expected_page=9, k=5) == 0.0
    # Outside the top-k window counts as a miss.
    assert recall_at_k([3, 1, 7], expected_page=7, k=2) == 0.0


def test_recall_at_k_unlabelled_is_zero():
    assert recall_at_k([1, 2], expected_page=None, k=5) == 0.0


def test_reciprocal_rank_is_one_over_rank():
    assert reciprocal_rank([5, 2, 9], expected_page=5) == 1.0
    assert reciprocal_rank([5, 2, 9], expected_page=2) == 0.5
    assert reciprocal_rank([5, 2, 9], expected_page=9) == 1 / 3
    assert reciprocal_rank([5, 2, 9], expected_page=8) == 0.0


def test_aggregate_excludes_unlabelled():
    per_q = [
        {"retrieved_pages": [1, 2], "expected_page": 1},  # hit, rank 1
        {"retrieved_pages": [3, 4], "expected_page": 4},  # hit, rank 2
        {"retrieved_pages": [9], "expected_page": None},  # unlabelled → ignored
    ]
    out = aggregate(per_q, k=5)
    assert out["n_labelled"] == 2
    assert out["recall_at_k"] == 1.0
    assert out["mrr"] == round((1.0 + 0.5) / 2, 3)


def test_aggregate_handles_no_labels():
    out = aggregate([{"retrieved_pages": [1], "expected_page": None}], k=5)
    assert out["n_labelled"] == 0
    assert out["recall_at_k"] == 0.0


def test_golden_dataset_is_valid():
    golden = json.loads(Path("data/golden/visual_qa.json").read_text(encoding="utf-8"))
    assert isinstance(golden, list) and len(golden) > 0
    for item in golden:
        assert "question" in item and item["question"].strip()
