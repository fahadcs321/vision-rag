"""Tests for the render → encode → mean-pool orchestration (model is faked)."""

from __future__ import annotations

import numpy as np

from src.indexing.embedder import EmbeddedDocument, embed_pdf_bytes


def test_embed_pdf_bytes_produces_aligned_outputs(tiny_pdf_bytes, fake_model):
    doc = embed_pdf_bytes(tiny_pdf_bytes, model=fake_model)

    assert isinstance(doc, EmbeddedDocument)
    assert len(doc) == 2
    # Every page has a multi-vector, a mean-pool vector, a thumbnail, a number.
    assert len(doc.page_embeddings) == 2
    assert len(doc.mean_pool_vecs) == 2
    assert len(doc.page_b64s) == 2
    assert doc.page_nums == [1, 2]


def test_mean_pool_vector_is_128d_and_is_the_mean(tiny_pdf_bytes, fake_model):
    doc = embed_pdf_bytes(tiny_pdf_bytes, model=fake_model)
    for emb, mean_vec in zip(doc.page_embeddings, doc.mean_pool_vecs, strict=True):
        assert mean_vec.shape == (128,)
        assert np.allclose(mean_vec, emb.mean(axis=0))


def test_thumbnails_are_nonempty_base64(tiny_pdf_bytes, fake_model):
    doc = embed_pdf_bytes(tiny_pdf_bytes, model=fake_model)
    for b64 in doc.page_b64s:
        assert isinstance(b64, str) and len(b64) > 50
