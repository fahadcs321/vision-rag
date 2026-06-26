"""
answerer.py — Qwen2-VL-7B-Instruct for grounded answer generation (local GPU path).

Given a question and the top retrieved page images, the VLM reads the pixels —
charts, tables, figures included — and writes a grounded answer with page
citations. Same Qwen2-VL backbone family as ColQwen2.5, so the retriever and the
answerer "see" documents the same way.

Loaded once and reused. Everything heavy is imported lazily so this module
imports without transformers/torch present (keeps the test suite light). The
API path (Groq/OpenAI vision) lives in api/routes/query.py; this is the local
no-API-cost alternative selected with ANSWERER_MODE=local.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

QWEN_VL_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "512"))
MAX_ANSWER_PAGES = int(os.getenv("ANSWER_MAX_PAGES", "3"))

_SYSTEM = (
    "You are analyzing document pages provided as images. Answer the question "
    "using ONLY what is visible in the pages, including charts, tables and figures. "
    "Cite the page numbers you used. If the answer is not present, say so honestly."
)


class QwenVLAnswerer:
    """Lazy-loaded Qwen2-VL wrapper for visual question answering."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Qwen2-VL ({QWEN_VL_MODEL_ID}) on {device}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_VL_MODEL_ID,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_ID)
        self._loaded = True
        logger.info("Qwen2-VL ready")

    @staticmethod
    def _b64_to_image(b64: str):
        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def answer(self, question: str, pages: list[dict[str, Any]]) -> str:
        """Answer a question from the top retrieved page images."""
        self.load()
        import torch

        images = [self._b64_to_image(p["page_b64"]) for p in pages[:MAX_ANSWER_PAGES]]

        content: list[dict[str, Any]] = [{"type": "text", "text": _SYSTEM}]
        for page, img in zip(pages, images, strict=False):
            content.append({"type": "image", "image": img})
            content.append(
                {"type": "text", "text": f"[Page {page['page_num']} of {page['filename']}]"}
            )
        content.append({"type": "text", "text": f"Question: {question}"})

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# Process-wide singleton.
qwen_vl_model = QwenVLAnswerer()
