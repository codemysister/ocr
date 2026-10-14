"""Validasi NPWP via Ornith. Flag format lama bila nomor bukan NIK 16 digit."""

from __future__ import annotations

import os
import re
from typing import Any

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict, compare_extracted_identity_scores
from systems.validation.keterangan_kesehatan_ai import (
    _parse_health_json,
    assert_allowed_llm_url,
    health_ai_base_url,
    health_ai_model,
)
from systems.validation.llm_fallback import complete_vision
from systems.validation.user_errors import attach_user_errors

NPWP_SYSTEM_PROMPT = """Kamu membaca kartu NPWP (Nomor Pokok Wajib Pajak) Indonesia.
Jangan mengarang. Nilai hanya dari gambar.

Kembalikan HANYA satu objek JSON (tanpa markdown):
- is_npwp (boolean): true jika ini kartu/dokumen NPWP. false jika KTP, KK, atau dokumen lain.
- holder_name (string): nama wajib pajak. Kosong jika tidak terbaca.
- npwp_number (string): nomor NPWP yang tercetak, salin apa adanya (boleh ada titik atau strip).
  Jika ada lebih dari satu nomor, ambil nomor yang menjadi NPWP, bukan NIK di dokumen lain.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def npwp_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah gambar ini NPWP? Ambil nama wajib pajak dan nomor NPWP.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi: {expected_name.strip()}")
    return "\n".join(lines)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


def npwp_digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def is_npwp_nik_format(number: str) -> bool:
    """NPWP hasil pemadanan memakai nomor yang sama dengan NIK: 16 digit."""
    return len(npwp_digits(number)) == 16


def npwp_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "npwp",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate jenis dokumen, nama, dan NIK. format_lama tidak menggagalkan dokumen."""
    profile = "npwp"
    is_npwp = _as_bool(parsed.get("is_npwp"))
    holder = str(parsed.get("holder_name") or "").strip()
    number = str(parsed.get("npwp_number") or "").strip()
    digits = npwp_digits(number)
    format_lama = not is_npwp_nik_format(number)
    exp_name = (expected_name or "").strip()
    exp_nik = npwp_digits(expected_nik)

    name_pass: bool | None = None
    identity: dict[str, Any] | None = None
    if exp_name:
        identity = compare_extracted_identity_scores(holder, exp_name)
        name_pass = bool(identity.get("token_gate_pass")) and float(
            identity.get("identity_combined_score") or 0
        ) >= 65.0

    nik_pass: bool | None = None
    if not format_lama and len(exp_nik) == 16:
        nik_pass = digits == exp_nik

    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_npwp:
        detail_lines.append("Gambar dikenali sebagai NPWP.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan NPWP.")
    if format_lama:
        detail_lines.append("Format lama: nomor NPWP bukan NIK 16 digit (belum pemadanan).")
    else:
        detail_lines.append("Nomor NPWP berformat NIK 16 digit.")
    if name_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append("Nama wajib pajak tidak cocok.")
    if nik_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append("NIK/nomor NPWP 16 digit tidak cocok.")

    # Nomor 16 digit yang cocok tetap lolos meski nama OCR/AI meleset.
    if nik_pass is True:
        identity_pass: bool | None = True
    elif not exp_name and nik_pass is None:
        identity_pass = None
    elif name_pass is False or nik_pass is False:
        identity_pass = False
    elif name_pass is True or nik_pass is True:
        identity_pass = True
    else:
        identity_pass = None

    matched = is_npwp and identity_pass is not False
    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    if matched:
        summary = (
            "NPWP valid, format lama (nomor bukan NIK)."
            if format_lama
            else "NPWP valid, nomor berformat NIK."
        )
    else:
        summary = "NPWP tidak valid: " + "; ".join(detail_lines[:2])

    detection = {
        "detected_profile_id": profile if is_npwp else None,
        "detected_document_type": profile_label(profile) if is_npwp else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": holder or None,
        "candidate_normalized": holder,
        "method": "ornith_npwp",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=exp_name,
        identity_pass=identity_pass,
        identity=identity,
        identity_min_score=65.0,
        document_type_pass=is_npwp,
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
            "is_npwp": is_npwp,
            "format_lama": format_lama,
            "npwp_number": number,
            "npwp_digits": digits,
            "is_nik_format": not format_lama,
            "name_pass": name_pass,
            "nik_pass": nik_pass,
            "document_type_pass": is_npwp,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": identity,
            "name_extraction": name_extraction,
            "npwp_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_npwp_image(
    image_bytes: bytes,
    *,
    document_type: str = "npwp",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Kirim gambar ke Ornith. error terisi jika model tidak menjawab."""
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_NPWP_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_NPWP_AI_MODEL"))
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
        npwp_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=NPWP_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_NPWP_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get("LLM_NPWP_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
        ),
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
    validation = npwp_validation_from_extraction(
        parsed,
        document_type=document_type,
        expected_name=expected_name,
        expected_nik=expected_nik,
        llm=llm,
    )
    return {
        "error": None,
        "llm": llm,
        "extraction": parsed,
        "validation": validation,
        "document_matched": validation.get("document_matched"),
        "format_lama": validation.get("format_lama"),
        "image_prepare": prep_meta,
        "raw_content": content,
    }
