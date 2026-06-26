"""Tests for PDF rendering and base64 image round-tripping."""

from __future__ import annotations

from PIL import Image

from src.indexing.pdf_renderer import PAGE_SIZE, b64_to_image, image_to_b64, render_pdf_bytes


def test_b64_round_trip_preserves_image():
    img = Image.new("RGB", (32, 32), (140, 90, 200))
    restored = b64_to_image(image_to_b64(img))
    assert restored.size == (32, 32)
    # JPEG is lossy; just confirm it decodes to roughly the same colour.
    r, g, b = restored.getpixel((16, 16))
    assert abs(r - 140) < 25 and abs(g - 90) < 25 and abs(b - 200) < 25


def test_render_pdf_bytes_returns_sized_pages(tiny_pdf_bytes):
    pages = render_pdf_bytes(tiny_pdf_bytes)
    assert len(pages) == 2
    # 1-indexed page numbers in order.
    assert [n for n, _ in pages] == [1, 2]
    # Every page is normalised to ColQwen's square input resolution.
    for _, img in pages:
        assert img.size == (PAGE_SIZE, PAGE_SIZE)
        assert img.mode == "RGB"
