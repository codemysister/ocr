"""Validasi bukti pemadanan NPWP (Ikhtisar Profil DJP) via Ornith."""

from __future__ import annotations

import os
import re
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
from systems.validation.rekening_ai import rekening_holder_name_matches
from systems.validation.user_errors import attach_user_errors

PEMADANAN_SYSTEM_PROMPT = """Kamu membaca bukti pemadanan NPWP dari Direktorat Jenderal Pajak.
Bisa berupa kartu NPWP baru, screenshot Ikhtisar Profil Wajib Pajak, atau Surat Keterangan Aktivasi NIK.
Jangan mengarang. Salin teks yang terlihat.

Kembalikan HANYA satu objek JSON valid. Kunci wajib pakai tanda kutip ganda.
Jangan awali dengan kata JSON, jangan pakai bullet, jangan pakai markdown.
- is_pemadanan_npwp (boolean): true jika salah satu ini:
  kartu NPWP baru (logo npwp, nomor 16 digit),
  Ikhtisar Profil Wajib Pajak,
  atau Surat Keterangan Aktivasi NIK (Kop DJP, judul "SURAT KETERANGAN AKTIVASI NIK").
  Pada surat aktivasi, NIK 16 digit adalah NPWP hasil pemadanan. Status Nonaktif tetap true.
  false jika KTP, KK, BPJS, atau kartu NPWP format lama 15 digit (contoh 09.123.456.7-890.000).
- holder_name (string): nama wajib pajak, apa adanya.
- npwp_number (string): nomor NPWP 16 digit. Pada surat aktivasi, salin NIK ke sini.
  Jika ada nomor 16 digit dan nomor format lama, ambil yang 16 digit. Kosong jika tidak terbaca.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def pemadanan_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah ini bukti pemadanan NPWP?",
        "Kartu NPWP 16 digit, Ikhtisar Profil DJP, dan Surat Keterangan Aktivasi NIK semuanya lolos.",
        "Pada surat aktivasi, salin NIK 16 digit sebagai nomor NPWP.",
        "KTP dan kartu format lama 15 digit bukan pemadanan.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi wajib pajak: {expected_name.strip()}")
    return "\n".join(lines)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _is_old_npwp_only(number: str) -> bool:
    text = str(number or "")
    if len(_digits(text)) == 16:
        return False
    return bool(re.search(r"\d{2}[.\s]\d{3}[.\s]\d{3}[.\s]\d[-.\s]\d{3}[.\s]\d{3}", text))


def pemadanan_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "pemadanan_npwp",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "pemadanan_npwp"
    is_doc = _as_bool(parsed.get("is_pemadanan_npwp"))
    holder = str(parsed.get("holder_name") or "").strip()
    number = str(parsed.get("npwp_number") or "").strip()
    digits = _digits(number)
    nik_field = _digits(str(parsed.get("nik") or ""))
    notes_blob = str(parsed.get("notes") or "").casefold()
    compact = re.sub(r"[\s\-]", "", notes_blob)
    aktivasi = "aktivasinik" in compact or "suratketeranganaktivasi" in compact
    if len(digits) != 16 and len(nik_field) == 16 and (aktivasi or is_doc):
        digits = nik_field
        number = nik_field
    if aktivasi:
        is_doc = True
    elif len(digits) == 16 and not any(
        marker in notes_blob for marker in ("ini ktp", "kartu keluarga", "bukan npwp", "bpjs")
    ):
        is_doc = True
    elif _is_old_npwp_only(number):
        is_doc = False
    nik = digits if len(digits) == 16 else ""
    format_lama = len(digits) != 16
    exp_name = (expected_name or "").strip()
    exp_nik = _digits(expected_nik)
    if len(exp_nik) != 16:
        exp_nik = ""

    name_info = (
        rekening_holder_name_matches(exp_name, holder)
        if exp_name and holder
        else {"pass": False if exp_name else None, "method": "none" if exp_name else "skipped", "identity": None}
    )
    name_pass = name_info["pass"]
    nik_pass: bool | None = None
    if exp_nik and nik:
        nik_pass = nik == exp_nik
    elif exp_nik:
        nik_pass = False

    if nik_pass is True:
        identity_pass: bool | None = True
        name_method = "nik" if name_pass is not True else name_info.get("method")
    elif name_pass is True:
        identity_pass = True
        name_method = name_info.get("method")
    elif exp_name or exp_nik:
        identity_pass = False
        name_method = name_info.get("method")
    else:
        identity_pass = None
        name_method = "skipped"

    matched = is_doc and identity_pass is not False
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_doc:
        detail_lines.append("Gambar dikenali sebagai bukti pemadanan NPWP.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan bukti pemadanan NPWP.")
    if format_lama:
        detail_lines.append("Nomor NPWP bukan 16 digit.")
    if name_pass is False and nik_pass is not True:
        blockers.append("IDENTITY")
        detail_lines.append("Nama wajib pajak tidak cocok.")
    if nik_pass is False and name_pass is not True:
        blockers.append("IDENTITY")
        detail_lines.append("NIK/nomor NPWP 16 digit tidak cocok.")
    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    if matched:
        summary = "Bukti pemadanan NPWP valid."
    else:
        summary = "Pemadanan NPWP tidak valid: " + "; ".join(detail_lines[:2])

    detection = {
        "detected_profile_id": profile if is_doc else None,
        "detected_document_type": profile_label(profile) if is_doc else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": holder or None,
        "candidate_normalized": holder,
        "method": "ornith_pemadanan_npwp",
    }
    identity = name_info.get("identity") if isinstance(name_info, dict) else None
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=exp_name,
        identity_pass=identity_pass,
        identity=identity,
        identity_min_score=65.0,
        document_type_pass=is_doc,
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
            "is_pemadanan_npwp": is_doc,
            "holder_name": holder,
            "npwp_number": number,
            "nik": nik,
            "format_lama": format_lama,
            "name_pass": name_pass,
            "nik_pass": nik_pass,
            "name_match_method": name_method,
            "document_type_pass": is_doc,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": identity,
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "pemadanan_npwp_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_pemadanan_npwp_image(
    image_bytes: bytes,
    *,
    document_type: str = "pemadanan_npwp",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_PEMADANAN_NPWP_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_PEMADANAN_NPWP_AI_MODEL"))
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
        pemadanan_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=PEMADANAN_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_PEMADANAN_NPWP_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get(
                "LLM_PEMADANAN_NPWP_AI_TIMEOUT_S",
                os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"),
            )
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
    validation = pemadanan_validation_from_extraction(
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
        "image_prepare": prep_meta,
        "raw_content": content,
    }
