"""Deteksi CV scan (teks tipis per halaman)."""

from __future__ import annotations

import re

from systems.cv.config import get_settings

_WS_RE = re.compile(r"\s+")


def normalize_text_len(text: str) -> int:
    return len(_WS_RE.sub(" ", (text or "").strip()))


def is_likely_scanned(*, text: str, num_pages: int) -> bool:
    settings = get_settings()
    pages = max(1, num_pages)
    return normalize_text_len(text) / pages < settings.scan_min_chars_per_page


def count_pdf_pages(file_bytes: bytes) -> int:
    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        n = doc.page_count
        doc.close()
        return max(1, n)
    except Exception:
        return 1
