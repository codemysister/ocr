"""OCR berbayar via Mistral Document AI API (mistral-ocr-latest)."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_client: Any = None
_init_lock = threading.Lock()

# Pricing referensi Mistral OCR 3 (per 1000 halaman, USD).
PRICE_PER_1000_PAGES_USD = Decimal("2.00")
PRICE_PER_1000_PAGES_ANNOTATED_USD = Decimal("3.00")


def _guess_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    return "image/jpeg"


def _get_api_key() -> str:
    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "MISTRAL_API_KEY belum diset. Daftar di https://console.mistral.ai/ "
            "lalu: export MISTRAL_API_KEY='sk-...' atau isi file .env"
        )
    return key


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    with _init_lock:
        if _client is not None:
            return _client
        try:
            from mistralai.client import Mistral
        except ImportError as e:
            raise RuntimeError(
                "Pasang SDK Mistral: pip install -r requirements-mistral.txt"
            ) from e
        _client = Mistral(api_key=_get_api_key())
        return _client


def _to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return _to_plain(dump())
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return _to_plain({k: v for k, v in d.items() if not k.startswith("_")})
    return str(obj)


def _pages_from_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    pages = raw.get("pages") or []
    out: list[dict[str, Any]] = []
    for p in pages:
        if isinstance(p, dict):
            out.append(p)
        else:
            out.append(_to_plain(p))
    return out


def _markdown_from_pages(pages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for p in pages:
        md = (p.get("markdown") or "").strip()
        if md:
            chunks.append(md)
    return "\n\n".join(chunks)


def _lines_from_markdown(markdown: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for raw in markdown.splitlines():
        t = raw.strip()
        if not t:
            continue
        lines.append({"text": t, "score": None, "polygon": None})
    return lines


def _annotation_enabled() -> bool:
    v = (os.environ.get("MISTRAL_OCR_ANNOTATION") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _usage_from_response(raw: dict[str, Any], *, annotated: bool) -> dict[str, Any]:
    usage = raw.get("usage_info") or {}
    if not isinstance(usage, dict):
        usage = _to_plain(usage)
    pages_processed = int(usage.get("pages_processed") or 0)
    doc_size = usage.get("doc_size_bytes")
    rate = PRICE_PER_1000_PAGES_ANNOTATED_USD if annotated else PRICE_PER_1000_PAGES_USD
    cost_usd = (Decimal(pages_processed) / Decimal(1000)) * rate
    usd_to_idr = Decimal(os.environ.get("MISTRAL_USD_TO_IDR", "17000"))
    return {
        "pages_processed": pages_processed,
        "doc_size_bytes": doc_size,
        "annotated": annotated,
        "estimated_cost_usd": float(cost_usd),
        "estimated_cost_idr": float(cost_usd * usd_to_idr),
        "price_per_1000_pages_usd": float(rate),
    }


def run_mistral_ocr(
    file_bytes: bytes,
    *,
    table_format: str | None = None,
    include_image_base64: bool | None = None,
) -> dict[str, Any]:
    """
    Kirim gambar/PDF ke Mistral OCR API.
    Respons diseragamkan dengan runner Paddle: text, markdown, lines, timing, usage.
    """
    if not file_bytes:
        raise ValueError("File kosong.")

    t0 = time.perf_counter()
    timing: dict[str, Any] = {"input_bytes": len(file_bytes)}

    mime = _guess_mime(file_bytes)
    b64 = base64.b64encode(file_bytes).decode("ascii")
    if mime == "application/pdf":
        doc = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        }
    else:
        doc = {
            "type": "image_url",
            "image_url": f"data:{mime};base64,{b64}",
        }

    model = (os.environ.get("MISTRAL_OCR_MODEL") or "mistral-ocr-2512").strip()
    tf = table_format
    if tf is None:
        tf = (os.environ.get("MISTRAL_OCR_TABLE_FORMAT") or "").strip() or None

    if include_image_base64 is None:
        include_image_base64 = (
            os.environ.get("MISTRAL_OCR_INCLUDE_IMAGE_BASE64", "").strip().lower()
            in ("1", "true", "yes")
        )

    t_get = time.perf_counter()
    client = _get_client()
    timing["get_client_s"] = round(time.perf_counter() - t_get, 3)

    annotated = _annotation_enabled()
    kw: dict[str, Any] = {
        "model": model,
        "document": doc,
        "include_image_base64": include_image_base64,
    }
    if tf:
        kw["table_format"] = tf
    if annotated:
        from systems.ocr.mistral_annotation import (
            DOCUMENT_ANNOTATION_PROMPT,
            build_document_annotation_format,
        )

        kw["document_annotation_format"] = build_document_annotation_format()
        kw["document_annotation_prompt"] = (
            os.environ.get("MISTRAL_OCR_ANNOTATION_PROMPT") or DOCUMENT_ANNOTATION_PROMPT
        ).strip()

    t_api = time.perf_counter()
    ocr_response = client.ocr.process(**kw)
    timing["api_call_s"] = round(time.perf_counter() - t_api, 3)

    t_build = time.perf_counter()
    raw = _to_plain(ocr_response)
    if not isinstance(raw, dict):
        raise RuntimeError("Respons Mistral OCR tidak dikenali.")

    pages = _pages_from_response(raw)
    markdown = _markdown_from_pages(pages)
    plain = markdown
    line_items = _lines_from_markdown(markdown)
    usage = _usage_from_response(raw, annotated=annotated)
    timing["build_output_s"] = round(time.perf_counter() - t_build, 3)

    from systems.ocr.mistral_annotation import parse_document_annotation

    doc_ann_raw = raw.get("document_annotation")
    doc_ann_parsed = parse_document_annotation(doc_ann_raw)
    timing["total_wall_s"] = round(time.perf_counter() - t0, 3)

    logger.info(
        "mistral_ocr total=%.3fs api=%.3fs pages=%d bytes=%d model=%s",
        timing["total_wall_s"],
        timing["api_call_s"],
        usage["pages_processed"],
        len(file_bytes),
        raw.get("model") or model,
    )

    return {
        "success": True,
        "mode": "mistral_ocr",
        "model": raw.get("model") or model,
        "markdown": markdown,
        "text": plain,
        "lines": line_items,
        "pages": pages,
        "usage": usage,
        "document_annotation": doc_ann_raw,
        "document_annotation_parsed": doc_ann_parsed,
        "timing": timing,
    }
