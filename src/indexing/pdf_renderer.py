"""
pdf_renderer.py — Convert PDF pages to PIL Images using PyMuPDF (fitz).

Why PyMuPDF over pdf2image?
- 10× faster on large documents
- Handles scanned PDFs, mixed-font pages, and password-protected files
- No Poppler dependency (easier Docker deployment)
- Returns images directly as numpy arrays (faster path to PIL)

2026 best practice: render at 150 DPI for a good accuracy/speed balance.
Use 200 DPI only if you need higher fidelity on text-dense scanned docs.
"""

import base64
import io
import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

PDF_DPI = int(os.getenv("PDF_DPI", "150"))
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_PDF", "200"))
PAGE_SIZE = 448  # ColQwen2.5 optimal resolution (448×448)


def render_pdf(pdf_path: str) -> list[tuple[int, Image.Image]]:
    """
    Render all pages of a PDF as PIL Images.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of (page_number, PIL.Image) tuples (1-indexed).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install PyMuPDF") from None

    pages = []
    doc = fitz.open(pdf_path)

    n_pages = min(len(doc), MAX_PAGES)
    if len(doc) > MAX_PAGES:
        logger.warning(
            f"PDF has {len(doc)} pages — capping at {MAX_PAGES}. "
            "Increase MAX_PAGES_PER_PDF env var to process more."
        )

    logger.info(f"Rendering {n_pages} pages from {Path(pdf_path).name} at {PDF_DPI} DPI")

    mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)  # 72 = PDF default DPI

    for page_idx in range(n_pages):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Resize to PAGE_SIZE × PAGE_SIZE (ColQwen2.5 optimal)
        img = img.resize((PAGE_SIZE, PAGE_SIZE), Image.LANCZOS)

        pages.append((page_idx + 1, img))  # 1-indexed page numbers

    doc.close()
    logger.info(f"Rendered {len(pages)} pages")
    return pages


def render_pdf_bytes(pdf_bytes: bytes) -> list[tuple[int, Image.Image]]:
    """
    Render a PDF from bytes (from HTTP upload) — no temp file needed.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install PyMuPDF") from None

    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = min(len(doc), MAX_PAGES)
    mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)

    for page_idx in range(n_pages):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.resize((PAGE_SIZE, PAGE_SIZE), Image.LANCZOS)
        pages.append((page_idx + 1, img))

    doc.close()
    return pages


def image_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string for storage in Qdrant payload."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_image(b64: str) -> Image.Image:
    """Convert base64 string back to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))
