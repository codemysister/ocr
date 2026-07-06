"""Ekstraksi teks native dari PDF (layer teks, tanpa OCR)."""

from __future__ import annotations


def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """Ambil teks per halaman via PyMuPDF. Kosong jika PDF murni gambar."""
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        text = (page.get_text("text") or "").strip()
        if text:
            parts.append(f"## Halaman {i}\n\n{text}")
    num_pages = max(1, doc.page_count)
    doc.close()
    return "\n\n".join(parts), num_pages
