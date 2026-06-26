"""
benchmark.py — Retrieval-quality benchmark for the Vision RAG pipeline.

Runs the visual-QA golden set against the live API and computes the standard
document-retrieval metrics used by ViDoRe: Recall@k and MRR (mean reciprocal
rank), based on whether the expected page appears in the retrieved set.

The pure scoring functions (recall_at_k, reciprocal_rank, aggregate) take plain
data and have no network/model dependency, so they are unit-tested offline.

Usage:
    python -m src.evaluation.benchmark --golden data/golden/visual_qa.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def recall_at_k(retrieved_pages: list[int], expected_page: int | None, k: int) -> float:
    """1.0 if the expected page is in the top-k retrieved pages, else 0.0.

    With no labelled expected page, the sample is not scoreable → returns 0.0
    and should be excluded from the denominator by the caller.
    """
    if expected_page is None:
        return 0.0
    return 1.0 if expected_page in retrieved_pages[:k] else 0.0


def reciprocal_rank(retrieved_pages: list[int], expected_page: int | None) -> float:
    """1/rank of the expected page (rank is 1-indexed); 0.0 if absent/unlabelled."""
    if expected_page is None:
        return 0.0
    for idx, page in enumerate(retrieved_pages, start=1):
        if page == expected_page:
            return 1.0 / idx
    return 0.0


def aggregate(per_question: list[dict[str, Any]], k: int = 5) -> dict[str, float]:
    """Aggregate Recall@k and MRR over only the labelled questions."""
    labelled = [q for q in per_question if q.get("expected_page") is not None]
    n = len(labelled)
    if n == 0:
        return {"n_labelled": 0, "recall_at_k": 0.0, "mrr": 0.0, "k": k}

    recall = sum(recall_at_k(q["retrieved_pages"], q["expected_page"], k) for q in labelled) / n
    mrr = sum(reciprocal_rank(q["retrieved_pages"], q["expected_page"]) for q in labelled) / n
    return {"n_labelled": n, "recall_at_k": round(recall, 3), "mrr": round(mrr, 3), "k": k}


def _query_api(api_url: str, question: str, top_k: int) -> list[int]:
    import requests

    resp = requests.post(
        f"{api_url}/query/", json={"question": question, "top_k": top_k}, timeout=60
    )
    resp.raise_for_status()
    return [p["page_num"] for p in resp.json().get("retrieved_pages", [])]


def run_benchmark(
    golden_path: str,
    output_path: str = "results/benchmark.json",
    api_url: str = "http://localhost:8000",
    k: int = 5,
) -> dict[str, Any]:
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))

    per_question: list[dict[str, Any]] = []
    print(f"Benchmarking {len(golden)} questions against {api_url} ...")
    for i, item in enumerate(golden, start=1):
        question = item["question"]
        try:
            retrieved = _query_api(api_url, question, top_k=k)
        except Exception as exc:  # noqa: BLE001 - record failures, keep going
            print(f"  [{i}] query failed: {exc}")
            retrieved = []
        per_question.append(
            {
                "question": question,
                "expected_page": item.get("expected_page"),
                "retrieved_pages": retrieved,
            }
        )

    summary = aggregate(per_question, k=k)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "results": per_question}, indent=2), encoding="utf-8"
    )

    print("\n── Retrieval Benchmark ──────────────────────────────")
    print(f"  Labelled questions: {summary['n_labelled']}")
    print(f"  Recall@{summary['k']}:        {summary['recall_at_k']}")
    print(f"  MRR:               {summary['mrr']}")
    print("──────────────────────────────────────────────────────")
    print(f"Written to {output_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision RAG retrieval benchmark.")
    parser.add_argument("--golden", default="data/golden/visual_qa.json")
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(args.golden, args.output, api_url=args.api_url, k=args.k)
