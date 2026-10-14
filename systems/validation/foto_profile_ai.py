"""Validasi foto profil via Ornith: pas foto + latar biru."""

from __future__ import annotations

import os
from typing import Any

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict
from systems.validation.keterangan_kesehatan_ai import (
    _parse_health_json,
    assert_allowed_llm_url,
    health_ai_base_url,
    health_ai_model,
)
from systems.validation.llm_fallback import complete_vision
from systems.validation.user_errors import attach_user_errors

PORTRAIT_SYSTEM_PROMPT = """Kamu memeriksa foto untuk keperluan pas foto / foto profil.
Jangan mengarang. Nilai hanya dari gambar.

Kembalikan HANYA satu objek JSON (tanpa markdown):
- is_profile_photo (boolean): true jika ini foto orang untuk pas foto / foto profil
  (wajah orang sebagai subjek utama, pose foto formal atau close-up). false jika KTP, dokumen,
  screenshot, pemandangan, atau bukan foto orang.
- is_blue_background (boolean): true hanya jika latar belakang dominan biru solid
  seperti pas foto resmi. false jika putih, merah, abu-abu, motif, atau ruangan.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def portrait_user_prompt() -> str:
    return (
        "Apakah gambar ini foto profil/pas foto, dan apakah latar belakangnya biru? "
        "Kembalikan JSON saja."
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


def portrait_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "foto_profile",
    expected_name: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate: pas foto DAN latar biru. Tidak memakai jumlah/ukuran wajah."""
    profile = "foto_profile"
    is_profile = _as_bool(parsed.get("is_profile_photo"))
    is_blue = _as_bool(parsed.get("is_blue_background"))
    matched = is_profile and is_blue

    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_profile:
        detail_lines.append("Gambar dikenali sebagai foto profil / pas foto.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan foto profil / pas foto.")
    if is_blue:
        detail_lines.append("Latar belakang biru.")
    else:
        blockers.append("BLUE_BACKGROUND")
        detail_lines.append("Latar belakang bukan biru solid.")

    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)

    summary = (
        "Foto profil valid: pas foto berlatar biru."
        if matched
        else "Foto profil tidak valid: " + "; ".join(detail_lines[:2])
    )
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    detection = {
        "detected_profile_id": profile if is_profile else None,
        "detected_document_type": profile_label(profile) if is_profile else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_image_only_profile",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=expected_name,
        identity_pass=None,
        identity=None,
        identity_min_score=65.0,
        document_type_pass=is_profile,
        document_matched=matched,
        name_extraction=name_extraction,
    )
    return attach_user_errors(
        {
            "valid": matched,
            "document_type": (document_type or "").strip(),
            "document_profile_id": profile,
            "validation_mode": "image",
            "validation_engine": "ornith",
            "is_profile_photo": is_profile,
            "is_blue_background": is_blue,
            "document_type_pass": is_profile,
            "document_matched": matched,
            "identity_pass": None,
            "identity": None,
            "name_extraction": name_extraction,
            "image_validation": {
                "engine": "ornith",
                "is_profile_photo": is_profile,
                "is_blue_background": is_blue,
                "blue_background_pass": is_blue,
                "face_pass": None,
                "notes": notes,
                "confidence": confidence,
            },
            "portrait_ai": {"llm": llm or {}, "extraction": parsed},
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "explanation": {
                "summary": summary,
                "detail_lines": detail_lines,
                "primary_blockers": blockers,
                "hints": [],
                "document_matched": matched,
                "validation_mode": "image",
            },
            "document_type_detection": detection,
            "verdict": verdict,
            "is_own_document": verdict["is_own_document"],
            "document_type_current": verdict["document_type_current"],
            "document_type_current_label": verdict["document_type_current_label"],
        }
    )


def analyze_portrait_image(
    image_bytes: bytes,
    *,
    document_type: str = "foto_profile",
    expected_name: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Kirim gambar ke Ornith. error terisi jika model tidak menjawab."""
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_PORTRAIT_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_PORTRAIT_AI_MODEL"))
    from systems.preprocessing.pipeline import encode_image_bytes_for_ocr_remote_safe

    prepared, prep_meta = encode_image_bytes_for_ocr_remote_safe(image_bytes, max_long_side=1280)
    thinking_on = (os.environ.get("LLM_HEALTH_AI_ENABLE_THINKING") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    extra: dict[str, Any] | None = None
    if not thinking_on:
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    content, meta = complete_vision(
        prepared,
        portrait_user_prompt(),
        image_mime="image/jpeg",
        system_text=PORTRAIT_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_PORTRAIT_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(os.environ.get("LLM_PORTRAIT_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))),
        extra=extra,
    )
    llm = {
        "base_url": base_url,
        "model": model,
        "latency_s": meta.get("latency_s"),
        "success": meta.get("success"),
        "error": meta.get("error"),
    }
    if not content:
        return {"error": meta.get("error") or "empty_content", "llm": llm, "image_prepare": prep_meta}
    parsed = _parse_health_json(content)
    if not parsed:
        return {
            "error": "invalid_json",
            "llm": llm,
            "raw_content": content,
            "image_prepare": prep_meta,
        }
    validation = portrait_validation_from_extraction(
        parsed,
        document_type=document_type,
        expected_name=expected_name,
        llm=llm,
    )
    return {
        "error": None,
        "llm": llm,
        "extraction": parsed,
        "validation": validation,
        "document_matched": validation.get("document_matched"),
        "image_prepare": prep_meta,
        "raw_content": content,
    }
