"""
conftest.py — Shared fixtures for the offline test suite.

These tests never load ColQwen, Qwen2-VL, or a real Qdrant: the heavy model is
faked and PDF rendering uses a tiny in-memory document. That keeps CI fast and
keyless while still exercising the real pipeline plumbing (render → embed →
mean-pool → MAX-SIM rerank).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def tiny_pdf_bytes() -> bytes:
    """A minimal 2-page PDF with text, rendered in-memory via PyMuPDF."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for n in range(2):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 100), f"Page {n + 1}: revenue chart Q3 = 210")
    data = doc.tobytes()
    doc.close()
    return data


class FakeColQwen:
    """A stand-in for the ColQwen model — deterministic, no torch.

    encode_pages returns small random patch matrices; mean_pool averages them.
    Enough to exercise the embedder + Qdrant payload path without a real model.
    """

    def __init__(self, dim: int = 128, n_patches: int = 16, seed: int = 0) -> None:
        self.dim = dim
        self.n_patches = n_patches
        self._rng = np.random.default_rng(seed)

    def encode_pages(self, images: list) -> list[np.ndarray]:
        return [
            self._rng.standard_normal((self.n_patches, self.dim)).astype(np.float32) for _ in images
        ]

    def mean_pool(self, page_embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(page_embeddings).mean(axis=0)


@pytest.fixture
def fake_model() -> FakeColQwen:
    return FakeColQwen()
