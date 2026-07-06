"""OCR fallback untuk CV scan (PDF/gambar) via PP-OCRv6."""

from __future__ import annotations

import logging
import os

from systems.ocr.fast_runner import run_paddleocr_fast

logger = logging.getLogger(__name__)


def _pp_ocr_tier() -> str | None:
    tier = (os.environ.get("CV_OCR_PP_TIER") or os.environ.get("OCR_FAST_TIER") or "small").strip()
    return tier or None


def _text_from_fast_result(result: dict) -> str:
    text = (result.get("text") or "").strip()
    if text:
        return text
    return (result.get("markdown") or "").strip()


def _render_pdf_pages_png(file_bytes: bytes, dpi: int = 150) -> list[bytes]:
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[bytes] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def ocr_pdf_with_fast(file_bytes: bytes, *, pp_ocr_tier: str | None = None) -> tuple[str, int]:
    tier = pp_ocr_tier if pp_ocr_tier is not None else _pp_ocr_tier()
    page_pngs = _render_pdf_pages_png(file_bytes)
    parts: list[str] = []
    for i, png in enumerate(page_pngs, start=1):
        try:
            result = run_paddleocr_fast(png, pp_ocr_tier=tier)
            md = _text_from_fast_result(result)
            if md:
                parts.append(f"## Halaman {i}\n\n{md}")
        except Exception as e:
            logger.warning("OCR CV page %s failed: %s", i, e)
    return "\n\n".join(parts), len(page_pngs)


def ocr_image_with_fast(file_bytes: bytes, *, pp_ocr_tier: str | None = None) -> str:
    tier = pp_ocr_tier if pp_ocr_tier is not None else _pp_ocr_tier()
    result = run_paddleocr_fast(file_bytes, pp_ocr_tier=tier)
    return _text_from_fast_result(result)
