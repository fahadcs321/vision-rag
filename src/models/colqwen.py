"""
colqwen.py — ColQwen2.5 / ColSmol wrapper.

Handles model loading (once at startup) and encodes:
  - Document pages (PIL Images → multi-vector embeddings)
  - Query text (str → query embeddings for MAX-SIM scoring)

2026 standard:
  - ColQwen2.5-7B for production (GPU, best accuracy on ViDoRe)
  - ColSmol-500M for dev / CPU-only environments
  - bfloat16 for 2× memory efficiency vs fp32
  - Batch processing to avoid OOM on large PDFs
"""

import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Model selection ────────────────────────────────────────────────────────────
MODEL_MODE = os.getenv("COLPALI_MODEL", "colsmol")  # "colqwen2.5" | "colsmol"

COLQWEN_MODEL_ID = os.getenv("COLQWEN_MODEL_ID", "vidore/colqwen2.5-v0.2")
COLSMOL_MODEL_ID = os.getenv("COLSMOL_MODEL_ID", "vidore/colsmolvlm-v0.1")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
VECTOR_DIM = 128  # ColPali family always outputs 128-dim vectors


class ColQwenModel:
    """
    Wrapper around colpali-engine for ColQwen2.5 or ColSmol.
    Load once; reuse for all ingestion and query encoding.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def load(self):
        """Load model weights. Called once at FastAPI startup."""
        if self._loaded:
            return

        logger.info(f"Loading ColPali model (mode={MODEL_MODE}, device={self.device})")

        if MODEL_MODE == "colqwen2.5":
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

            model_id = COLQWEN_MODEL_ID
            self.model = ColQwen2_5.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(model_id, use_fast=True)

        else:  # colsmol — CPU-friendly default for dev
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor

            model_id = COLSMOL_MODEL_ID
            self.model = ColIdefics3.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            ).eval()
            self.processor = ColIdefics3Processor.from_pretrained(model_id, use_fast=True)

        self._loaded = True
        logger.info(f"ColPali model loaded: {model_id} on {self.device}")

    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Encode a list of page images into multi-vector embeddings.

        Args:
            images: List of PIL Images (one per page)

        Returns:
            List of numpy arrays, shape (n_patches, 128) per page.
            n_patches varies by image size (~400-1030 for ColQwen2.5).
        """
        self._check_loaded()
        all_embeddings = []

        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i : i + BATCH_SIZE]
            logger.debug(f"Encoding pages {i}–{i + len(batch) - 1}")

            with torch.no_grad():
                batch_input = self.processor.process_images(batch).to(self.device)
                embeddings = self.model(**batch_input)  # (batch, n_patches, 128)

            for emb in embeddings:
                all_embeddings.append(emb.cpu().float().numpy())

        return all_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a text query into per-token embeddings for MAX-SIM scoring.

        Args:
            query: User question string

        Returns:
            numpy array shape (n_tokens, 128).
            Typical query: 8-12 tokens → 8-12 × 128 vectors.
        """
        self._check_loaded()

        with torch.no_grad():
            query_input = self.processor.process_queries([query]).to(self.device)
            embeddings = self.model(**query_input)  # (1, n_tokens, 128)

        return embeddings[0].cpu().float().numpy()

    def mean_pool(self, page_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute mean-pooled summary vector from multi-vector page embeddings.
        Used for fast stage-1 prefetch search.

        Args:
            page_embeddings: shape (n_patches, 128)

        Returns:
            shape (128,) — single summary vector
        """
        return page_embeddings.mean(axis=0)

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "ColQwen model not loaded. Call model.load() first "
                "(this happens automatically in FastAPI lifespan)."
            )


# Singleton — one instance per process
colqwen_model = ColQwenModel()
