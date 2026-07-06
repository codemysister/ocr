"""Docling parser untuk CV (PDF/DOCX)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_with_docling(file_bytes: bytes, filename: str) -> tuple[Any, str]:
    from docling.document_converter import DocumentConverter

    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = DocumentConverter().convert(tmp_path)
        document = result.document
        return document, document.export_to_markdown()
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove temp file %s", tmp_path)
