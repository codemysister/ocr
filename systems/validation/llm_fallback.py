"""Fallback validasi via LLM lokal multimodal (gambar → baca & validasi)."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from systems.ocr.mistral_annotation import (
    document_type_profile_from_annotation,
    parse_document_annotation,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://172.21.15.218:8081/v1"

_SYSTEM_PROMPT = """You validate Indonesian identity document images (KTP, NPWP, KK, BPJS, etc.).
Read the document visually from the image and return ONLY a JSON object with keys:
- document_type_label (string): e.g. KTP, NPWP, Kartu Keluarga
- holder_name (string): full legal name of the person on the document
- identity_number (string): NIK or NPWP if visible (digits only)
- document_type_matches_expected (boolean): true if the image is the expected document type
- identity_matches_reference (boolean): true if holder_name matches the reference name (allow minor OCR/spelling variants)
- confidence (number 0-100): your confidence in this validation
- notes (string): brief reason in Indonesian

Be strict: if the image is not a readable identity document, set both boolean fields to false."""


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
        return float(os.environ.get("LLM_FALLBACK_TIMEOUT_S", "90"))
    except ValueError:
        return 90.0


def _api_key() -> str:
    return (os.environ.get("LLM_FALLBACK_API_KEY") or "").strip()


def _max_tokens() -> int:
    try:
        return max(128, int(os.environ.get("LLM_FALLBACK_MAX_TOKENS", "768")))
    except ValueError:
        return 768


def _min_confidence() -> float:
    try:
        return float(os.environ.get("LLM_FALLBACK_MIN_CONFIDENCE", "60"))
    except ValueError:
        return 60.0


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


def _normalize_mime(mime: str) -> str:
    m = (mime or "image/png").strip().casefold()
    if m in {"image/jpg", "image/jpeg"}:
        return "image/jpeg"
    if m == "image/webp":
        return "image/webp"
    return "image/png"


def _vision_user_prompt(
    *,
    document_profile_id: str,
    expected_name: str,
    expected_nik: str,
) -> str:
    lines = [
        "Validate this Indonesian identity document image.",
        "Read all visible fields directly from the image (do not assume OCR text).",
    ]
    profile = (document_profile_id or "").strip()
    if profile:
        lines.append(f"Expected document type profile: {profile}")
    if expected_name.strip():
        lines.append(f"Reference holder name: {expected_name.strip()}")
    if expected_nik.strip():
        lines.append(f"Reference NIK/NPWP: {expected_nik.strip()}")
    lines.append("Return the JSON object only.")
    return "\n".join(lines)


def _chat_completions(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "mode": "vision",
        "model": _model(),
        "base_url": _base_url(),
        "latency_s": None,
        "error": None,
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
        logger.warning("LLM vision HTTP error: %s", meta["error"])
        return None, meta
    except Exception as e:
        meta["latency_s"] = round(time.perf_counter() - t0, 3)
        meta["error"] = str(e)
        logger.warning("LLM vision failed: %s", e)
        return None, meta

    meta["latency_s"] = round(time.perf_counter() - t0, 3)
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        meta["error"] = f"bad_response: {e}"
        return None, meta
    return {"content": content, "raw": raw}, meta


def validate_document_via_llm_vision(
    image_bytes: bytes,
    *,
    image_mime: str = "image/png",
    document_profile_id: str = "",
    expected_name: str = "",
    expected_nik: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Kirim gambar dokumen ke LLM multimodal; kembalikan (hasil validasi, meta).
    """
    meta: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "mode": "vision",
        "model": _model(),
        "base_url": _base_url(),
        "latency_s": None,
        "error": None,
    }
    if not is_llm_fallback_enabled():
        meta["error"] = "disabled"
        return None, meta
    if not image_bytes:
        meta["error"] = "empty_image"
        return None, meta

    meta["attempted"] = True
    mime = _normalize_mime(image_mime)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    user_prompt = _vision_user_prompt(
        document_profile_id=document_profile_id,
        expected_name=expected_name,
        expected_nik=expected_nik,
    )

    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "max_tokens": _max_tokens(),
        "temperature": 0.1,
    }

    parsed_resp, call_meta = _chat_completions(payload)
    meta.update({k: v for k, v in call_meta.items() if k != "attempted"})
    if not parsed_resp:
        return None, meta

    content = parsed_resp.get("content")
    parsed = _extract_json_object(content if isinstance(content, str) else str(content))
    if not parsed:
        meta["error"] = "invalid_json"
        if isinstance(content, str):
            meta["raw_content"] = content[:800]
        return None, meta

    ann = parse_document_annotation(parsed) or {}
    doc_type_ok = bool(parsed.get("document_type_matches_expected"))
    identity_ok = bool(parsed.get("identity_matches_reference"))
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    profile = (document_profile_id or "").strip().casefold()
    ann_profile = document_type_profile_from_annotation(
        {
            "document_type_label": parsed.get("document_type_label") or ann.get("document_type_label"),
            "holder_name": parsed.get("holder_name") or ann.get("holder_name"),
            "identity_number": parsed.get("identity_number") or ann.get("identity_number"),
        }
    )
    if profile and ann_profile and ann_profile != profile:
        doc_type_ok = False

    exp_nik = re.sub(r"\D", "", expected_nik or "")
    llm_nik = re.sub(r"\D", "", str(parsed.get("identity_number") or ann.get("identity_number") or ""))
    if len(exp_nik) == 16 and llm_nik and llm_nik != exp_nik:
        identity_ok = False

    min_conf = _min_confidence()
    if confidence < min_conf:
        doc_type_ok = False
        identity_ok = False

    want_identity = bool(expected_name.strip())
    document_matched = doc_type_ok and (identity_ok if want_identity else True)

    result: dict[str, Any] = {
        "document_type_label": str(parsed.get("document_type_label") or ann.get("document_type_label") or ""),
        "holder_name": str(parsed.get("holder_name") or ann.get("holder_name") or ""),
        "identity_number": str(parsed.get("identity_number") or ann.get("identity_number") or ""),
        "document_type_matches_expected": doc_type_ok,
        "identity_matches_reference": identity_ok,
        "confidence": confidence,
        "notes": str(parsed.get("notes") or "").strip(),
        "document_matched": document_matched,
        "document_type_pass": doc_type_ok,
        "identity_pass": identity_ok if want_identity else None,
    }

    if not (result["holder_name"] or result["document_type_label"]):
        meta["error"] = "empty_extraction"
        return None, meta

    meta["success"] = True
    meta["validation"] = result
    meta["annotation"] = {
        "document_type_label": result["document_type_label"],
        "holder_name": result["holder_name"],
        "identity_number": result["identity_number"],
    }
    return result, meta


def merge_llm_vision_into_validation(
    validation_detail: dict[str, Any],
    llm_result: dict[str, Any],
    fb_meta: dict[str, Any],
    *,
    expected_name: str = "",
) -> dict[str, Any]:
    """Gabungkan hasil validasi Paddle dengan keputusan AI vision."""
    out = dict(validation_detail)
    ann = fb_meta.get("annotation") or {
        "document_type_label": llm_result.get("document_type_label"),
        "holder_name": llm_result.get("holder_name"),
        "identity_number": llm_result.get("identity_number"),
    }

    out["llm_fallback"] = fb_meta
    out["llm_validation"] = llm_result
    out["llm_annotation"] = ann
    out["document_type_pass"] = bool(llm_result.get("document_type_pass"))
    out["identity_pass"] = llm_result.get("identity_pass")
    out["document_matched"] = bool(llm_result.get("document_matched"))

    holder = (ann.get("holder_name") or "").strip()
    if holder:
        from systems.validation.fuzzy_compare import _normalize_name

        out["name_extraction"] = {
            "candidate_raw": holder,
            "candidate_normalized": _normalize_name(holder),
            "method": "llm_vision_validation",
        }
        if expected_name.strip():
            from systems.validation.fuzzy_compare import compare_extracted_identity_scores

            out["identity"] = compare_extracted_identity_scores(holder, expected_name)

    notes = (llm_result.get("notes") or "").strip()
    conf = llm_result.get("confidence")
    summary = (
        "Validasi AI vision: dokumen "
        + ("cocok" if out["document_matched"] else "tidak cocok")
        + (f" (confidence {conf:.0f}%)" if isinstance(conf, (int, float)) else "")
        + (f". {notes}" if notes else "")
    )
    explanation = dict(out.get("explanation") or {})
    explanation["summary"] = summary
    explanation["primary_blockers"] = [] if out["document_matched"] else ["LLM_VISION"]
    explanation["detail_lines"] = [
        f"Jenis dokumen (AI): {'lolos' if out['document_type_pass'] else 'gagal'}",
    ]
    if expected_name.strip():
        explanation["detail_lines"].append(
            f"Identitas (AI): {'lolos' if out.get('identity_pass') else 'gagal'}"
        )
    if notes:
        explanation["detail_lines"].append(notes)
    out["explanation"] = explanation

    verdict = dict(out.get("verdict") or {})
    verdict["summary"] = summary
    verdict["is_own_document"] = out.get("identity_pass")
    out["verdict"] = verdict
    out["is_own_document"] = out.get("identity_pass")
    return out
