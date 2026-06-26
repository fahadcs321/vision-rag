"""
demo.py — Self-contained demo data for the hosted Streamlit app.

The full pipeline needs ColQwen2.5 + Qwen2-VL + Qdrant + a GPU — far more than a
free Streamlit Cloud container can run. DEMO_MODE renders a faithful, scripted
walkthrough instead: a generated revenue bar chart (drawn with Pillow, no heavy
deps), a grounded answer, and the "retrieved" source page — so the live URL shows
exactly what the real system returns without loading any model.

Run the real pipeline locally (uvicorn + Qdrant) with DEMO_MODE unset.
"""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw

# Page background / accent match the dark portfolio theme.
_BG = (13, 17, 23)
_PANEL = (22, 27, 34)
_BAR = (139, 92, 246)
_BAR_EDGE = (168, 85, 247)
_TEXT = (201, 209, 217)


def _revenue_chart_b64() -> str:
    """Draw a simple quarterly revenue bar chart and return it as base64 PNG."""
    w, h = 448, 448
    img = Image.new("RGB", (w, h), _BG)
    d = ImageDraw.Draw(img)

    d.rectangle([24, 24, w - 24, h - 24], fill=_PANEL)
    d.text((40, 40), "Tesla — Quarterly Revenue ($B)", fill=_TEXT)

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = [23.3, 24.9, 23.4, 25.2]  # 2023, illustrative
    base_y = h - 70
    max_v = max(values)
    bar_w = 60
    gap = 40
    x = 70
    for q, v in zip(quarters, values, strict=True):
        bar_h = int((v / max_v) * 280)
        top = base_y - bar_h
        d.rectangle([x, top, x + bar_w, base_y], fill=_BAR, outline=_BAR_EDGE, width=2)
        d.text((x + 8, top - 18), f"{v:.1f}", fill=_TEXT)
        d.text((x + 18, base_y + 8), q, fill=_TEXT)
        x += bar_w + gap

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def demo_result(question: str) -> dict:
    """Return a canned query result mirroring the real /query response shape."""
    chart_b64 = _revenue_chart_b64()
    answer = (
        "Tesla's total revenue grew from $81.5B in 2022 to $96.8B in 2023 — a 19% "
        "increase. The quarterly chart on page 42 shows revenue climbing from $23.3B "
        "in Q1 to $25.2B in Q4 2023. This figure exists only in the bar chart, not in "
        "any extractable text, which is exactly where OCR-based text-RAG fails."
    )
    return {
        "question": question,
        "answer": answer,
        "retrieved_pages": [
            {
                "doc_id": "demo-tsla-2023",
                "filename": "tesla_annual_report_2023.pdf",
                "page_num": 42,
                "score": 0.913,
                "page_b64": chart_b64,
            },
            {
                "doc_id": "demo-tsla-2023",
                "filename": "tesla_annual_report_2023.pdf",
                "page_num": 41,
                "score": 0.847,
                "page_b64": chart_b64,
            },
        ],
        "latency_ms": {"retrieval_ms": 540, "answer_ms": 1120, "total_ms": 1660},
    }


DEMO_DOCUMENTS = [
    {"doc_id": "demo-tsla-2023", "filename": "tesla_annual_report_2023.pdf", "pages": 64},
]
