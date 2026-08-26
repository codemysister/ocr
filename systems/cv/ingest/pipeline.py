"""Ingest CV: parse → chunk → embed → index."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from systems.cv.chunking.structure_chunker import (
    build_chunks,
    make_doc_id,
    sections_from_cv_text,
    sections_from_docling_document,
)
from systems.cv.ingest.docling_parser import parse_with_docling
from systems.cv.ingest.ocr_fallback import ocr_image_with_fast, ocr_pdf_with_fast
from systems.cv.ingest.pdf_text import extract_pdf_text
from systems.cv.ingest.scan_detector import count_pdf_pages, is_likely_scanned, normalize_text_len
from systems.cv.match.cv_matcher import match_cv_chunks

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
_PDF_SUFFIX = ".pdf"


def _doc_title_from_filename(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename


def _index_chunks_optional(
    chunks: list[dict[str, Any]],
    *,
    doc_id: str,
    replace_existing: bool,
) -> tuple[dict[str, Any], float, float]:
    """Embed + OpenSearch. Dilewati jika deps/OpenSearch tidak ada (pipeline match tetap jalan)."""
    try:
        from systems.cv.embedding.embedder import embed_texts
        from systems.cv.index.indexer import index_chunks
        from systems.cv.index.opensearch_client import (
            delete_by_doc_id,
            ensure_index,
        )
    except ImportError as e:
        logger.warning("CV embed/index dilewati (dependensi tidak terpasang): %s", e)
        return (
            {
                "indexed": 0,
                "skipped": True,
                "reason": "cv_index_deps_missing",
                "error": str(e),
            },
            0.0,
            0.0,
        )

    try:
        if replace_existing:
            ensure_index()
            delete_by_doc_id(doc_id)
        t_embed = time.perf_counter()
        vectors = embed_texts([c["content"] for c in chunks])
        embed_s = time.perf_counter() - t_embed
        t_index = time.perf_counter()
        index_result = index_chunks(chunks, vectors=vectors)
        index_s = time.perf_counter() - t_index
        return index_result, embed_s, index_s
    except Exception as e:
        from systems.cv.index.opensearch_client import (
            OpenSearchUnavailableError,
            is_opensearch_connection_error,
        )

        if isinstance(e, OpenSearchUnavailableError) or is_opensearch_connection_error(e):
            logger.warning("CV index dilewati (OpenSearch): %s", e)
            return (
                {
                    "indexed": 0,
                    "skipped": True,
                    "reason": "opensearch_unavailable",
                    "error": str(e),
                },
                0.0,
                0.0,
            )
        raise


def ingest_cv_bytes(
    file_bytes: bytes,
    *,
    filename: str,
    replace_existing: bool = True,
    expected_name: str = "",
    education_query: str = "",
    experience_query: str = "",
) -> dict[str, Any]:
    if not file_bytes:
        raise ValueError("File kosong.")

    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    source_file = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    doc_id = make_doc_id(source_file, file_bytes)
    doc_title = _doc_title_from_filename(source_file)
    suffix = Path(source_file).suffix.lower()

    parse_mode = "docling"
    markdown = ""
    document = None
    num_pages = 1

    if suffix == _PDF_SUFFIX:
        num_pages = count_pdf_pages(file_bytes)

    try:
        if suffix in _TEXT_SUFFIXES:
            markdown = file_bytes.decode("utf-8", errors="replace")
            parse_mode = "text_native"
        elif suffix in _IMAGE_SUFFIXES:
            markdown = ocr_image_with_fast(file_bytes)
            parse_mode = "ocr_fast"
        elif suffix == _PDF_SUFFIX:
            pdf_md, pdf_pages = extract_pdf_text(file_bytes)
            num_pages = pdf_pages
            if pdf_md.strip() and not is_likely_scanned(text=pdf_md, num_pages=num_pages):
                markdown = pdf_md
                parse_mode = "pdf_text"
                logger.info("CV PDF %s — teks native PyMuPDF (%d chars)", source_file, normalize_text_len(pdf_md))
            else:
                try:
                    document, markdown = parse_with_docling(file_bytes, source_file)
                    if is_likely_scanned(text=markdown, num_pages=num_pages):
                        raise ValueError("PDF scan — lanjut OCR")
                    parse_mode = "docling"
                except Exception as doc_err:
                    logger.info("CV PDF %s — Docling/OCR path: %s", source_file, doc_err)
                    markdown, num_pages = ocr_pdf_with_fast(file_bytes)
                    parse_mode = "ocr_fast"
                    document = None
        else:
            document, markdown = parse_with_docling(file_bytes, source_file)
    except Exception as e:
        logger.warning("Docling failed for %s: %s", source_file, e)
        if suffix == _PDF_SUFFIX:
            markdown, num_pages = ocr_pdf_with_fast(file_bytes)
            parse_mode = "ocr_fast"
        elif suffix in _IMAGE_SUFFIXES:
            markdown = ocr_image_with_fast(file_bytes)
            parse_mode = "ocr_fast"
        else:
            raise ValueError(f"Gagal parse CV: {e}") from e

    if not markdown.strip():
        raise ValueError("Tidak ada teks yang diekstrak dari CV.")

    timing["parse_s"] = round(time.perf_counter() - t0, 3)

    t_chunk = time.perf_counter()

    sections = (
        sections_from_docling_document(document)
        if document is not None
        else sections_from_cv_text(markdown)
    )
    chunks = build_chunks(
        sections,
        doc_id=doc_id,
        doc_title=doc_title,
        source_file=source_file,
        parse_mode=parse_mode,
    )
    if not chunks:
        raise ValueError("Tidak ada chunk yang dihasilkan dari CV.")

    timing["chunk_s"] = round(time.perf_counter() - t_chunk, 3)

    index_result, embed_s, index_s = _index_chunks_optional(
        chunks,
        doc_id=doc_id,
        replace_existing=replace_existing,
    )
    timing["embed_s"] = round(embed_s, 3)
    timing["index_s"] = round(index_s, 3)
    timing["total_s"] = round(time.perf_counter() - t0, 3)

    cv_match = match_cv_chunks(
        chunks,
        expected_name=expected_name,
        education_query=education_query,
        experience_query=experience_query,
    )

    return {
        "success": True,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "source_file": source_file,
        "parse_mode": parse_mode,
        "num_pages": num_pages,
        "text_chars": normalize_text_len(markdown),
        "chunk_count": len(chunks),
        "index": index_result,
        "timing": timing,
        "cv_match": cv_match,
    }
