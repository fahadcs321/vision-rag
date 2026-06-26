"""
embedder.py — Turn a PDF into everything Qdrant needs, in one call.

Ties together the render → encode → mean-pool → thumbnail steps so both the
ingest route and offline scripts share one code path:

    PDF bytes → page images → ColQwen multi-vectors
                            → mean-pool summary vectors (stage-1 prefetch)
                            → base64 JPEG thumbnails (UI proof-of-sources)

The heavy model is imported lazily so this module imports without torch present
(keeps the test suite light).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.indexing.pdf_renderer import image_to_b64, render_pdf_bytes


@dataclass
class EmbeddedDocument:
    """Everything required to upsert one document's pages into Qdrant."""

    page_nums: list[int] = field(default_factory=list)
    page_embeddings: list[np.ndarray] = field(default_factory=list)  # (n_patches, 128) each
    mean_pool_vecs: list[np.ndarray] = field(default_factory=list)  # (128,) each
    page_b64s: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.page_nums)


def embed_pdf_bytes(pdf_bytes: bytes, model: Any | None = None) -> EmbeddedDocument:
    """Render and encode a PDF (from bytes) into an EmbeddedDocument.

    Args:
        pdf_bytes: raw PDF content (e.g. from an HTTP upload).
        model: a loaded ColQwen-like model exposing ``encode_pages`` and
            ``mean_pool``. Defaults to the process-wide ``colqwen_model`` singleton.
    """
    if model is None:
        from src.models.colqwen import colqwen_model

        model = colqwen_model

    page_tuples = render_pdf_bytes(pdf_bytes)  # [(page_num, PIL.Image), ...]
    page_nums = [num for num, _ in page_tuples]
    images = [img for _, img in page_tuples]

    page_embeddings = model.encode_pages(images)
    mean_pool_vecs = [model.mean_pool(emb) for emb in page_embeddings]
    page_b64s = [image_to_b64(img) for img in images]

    return EmbeddedDocument(
        page_nums=page_nums,
        page_embeddings=page_embeddings,
        mean_pool_vecs=mean_pool_vecs,
        page_b64s=page_b64s,
    )
