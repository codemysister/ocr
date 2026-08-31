"""Fallback validasi via LLM lokal (OpenAI-compatible /v1/chat/completions)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from systems.ocr.mistral_annotation import parse_document_annotation

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://172.21.15.218:8080/v1"

_SYSTEM_PROMPT = """You extract structured fields from noisy OCR text of Indonesian identity documents (KTP, NPWP, KK, BPJS, etc.).
Return ONLY a JSON object with keys:
- document_type_label (string): e.g. KTP, NPWP, Kartu Keluarga
- holder_name (string): full legal name of the person, no labels like Nama or WNI
- identity_number (string): NIK or NPWP if visible (digits only preferred)

The OCR text may have typos, missing labels, or garbled words. Use context to infer correct values.
If uncertain, use empty string for optional fields but always try holder_name."""


def _env_bool(name: str, *, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def is_llm_fallback_enabled() -> bool:
    if not _env_bool("LLM_FALLBACK_ENABLED", default=True):
        return False
    return bool(_base_url())


def _base_url() -> str:
    return (os.environ.get("LLM_FALLBACK_BASE_URL") or _DEFAULT_BASE_URL).strip().rstrip("/")


def _model() -> str:
    return (os.environ.get("LLM_FALLBACK_MODEL") or "qwen").strip()


def _timeout_s() -> float:
    try:
        return float(os.environ.get("LLM_FALLBACK_TIMEOUT_S", "60"))
    except ValueError:
        return 60.0


def _api_key() -> str:
    return (os.environ.get("LLM_FALLBACK_API_KEY") or "").strip()


def _max_tokens() -> int:
    try:
        return max(64, int(os.environ.get("LLM_FALLBACK_MAX_TOKENS", "512")))
    except ValueError:
        return 512


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_document_annotation_via_llm(
    ocr_text: str,
    *,
    document_profile_id: str = "",
    expected_name: str = "",
    expected_nik: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Panggil chat LLM lokal; kembalikan (annotation dict, meta)."""
    meta: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "model": _model(),
        "base_url": _base_url(),
        "latency_s": None,
        "error": None,
    }
    if not is_llm_fallback_enabled():
        meta["error"] = "disabled"
        return None, meta

    ocr_text = (ocr_text or "").strip()
    if not ocr_text:
        meta["error"] = "empty_ocr_text"
        return None, meta

    meta["attempted"] = True
    profile = (document_profile_id or "").strip()
    user_lines = [
        "Noisy OCR text from Indonesian document:",
        "---",
        ocr_text[:8000],
        "---",
    ]
    if profile:
        user_lines.append(f"Expected document profile: {profile}")
    if expected_name.strip():
        user_lines.append(f"Reference holder name: {expected_name.strip()}")
    if expected_nik.strip():
        user_lines.append(f"Reference NIK/NPWP: {expected_nik.strip()}")
    user_lines.append("Extract document_type_label, holder_name, identity_number as JSON.")

    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
        "max_tokens": _max_tokens(),
        "temperature": 0.1,
    }

    url = f"{_base_url()}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = _api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_timeout_s()) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        meta["latency_s"] = round(time.perf_counter() - t0, 3)
        try:
            detail = e.read().decode("utf-8")[:500]
        except Exception:
            detail = str(e)
        meta["error"] = f"http_{e.code}: {detail}"
        logger.warning("LLM fallback HTTP error: %s", meta["error"])
        return None, meta
    except Exception as e:
        meta["latency_s"] = round(time.perf_counter() - t0, 3)
        meta["error"] = str(e)
        logger.warning("LLM fallback failed: %s", e)
        return None, meta

    meta["latency_s"] = round(time.perf_counter() - t0, 3)

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        meta["error"] = f"bad_response: {e}"
        return None, meta

    parsed = _extract_json_object(content if isinstance(content, str) else str(content))
    ann = parse_document_annotation(parsed)
    if not ann or not (ann.get("holder_name") or "").strip():
        meta["error"] = "no_holder_name_in_response"
        if isinstance(content, str):
            meta["raw_content"] = content[:500]
        return None, meta

    meta["success"] = True
    meta["annotation"] = ann
    return ann, meta
