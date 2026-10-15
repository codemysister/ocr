"""Validasi SKCK via Ornith."""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

from systems.validation.keterangan_kesehatan_ai import (
    _iso_date,
    _parse_health_json,
    assert_allowed_llm_url,
    health_ai_base_url,
    health_ai_model,
)
from systems.validation.llm_fallback import complete_vision
from systems.validation.skck_validity import format_id_date

SKCK_SYSTEM_PROMPT = """Kamu membaca Surat Keterangan Catatan Kepolisian (SKCK).
Jangan mengarang. Jangan menulis analisis. Jangan memakai tag think.

Kembalikan HANYA satu objek JSON:
- is_skck (boolean): true hanya jika ini SKCK / Surat Keterangan Catatan Kepolisian.
  false jika KTP, KK, surat lain, atau bukan SKCK.
- holder_name (string): nama pemohon yang tercetak. Kosong jika tidak ada.
- identity_number (string): NIK 16 digit jika terlihat, else "".
- valid_from (string): tanggal "Berlaku untuk" atau "Valid from", format YYYY-MM-DD. Kosong jika tidak ada.
- valid_until (string): tanggal "Sampai dengan" atau "Until", format YYYY-MM-DD. Kosong jika tidak ada.
- confidence (number 0-100)
- notes (string): maksimal 12 kata
"""


def skck_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah ini SKCK?",
        "Salin nama pemohon, tanggal Berlaku untuk, dan tanggal Sampai dengan.",
        "Balas hanya satu objek JSON.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi pemohon: {expected_name.strip()}")
    return "\n".join(lines)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


def skck_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    expected_name: str = "",
    expected_nik: str = "",
    today: date | None = None,
) -> dict[str, Any]:
    is_skck = _as_bool(parsed.get("is_skck"))
    notes = str(parsed.get("notes") or "").casefold()
    if "skck" in notes or "catatan kepolisian" in notes:
        is_skck = True

    holder = str(parsed.get("holder_name") or "").strip()
    nik = re.sub(r"\D", "", str(parsed.get("identity_number") or ""))
    if len(nik) != 16:
        nik = ""
    exp_nik = re.sub(r"\D", "", expected_nik or "")
    if len(exp_nik) != 16:
        exp_nik = ""
    valid_from = _iso_date(parsed.get("valid_from"))
    valid_until = _iso_date(parsed.get("valid_until"))
    as_of = today or date.today()

    identity_pass = None
    identity = None
    if expected_name.strip():
        from systems.validation.fuzzy_compare import compare_extracted_identity_scores

        identity = compare_extracted_identity_scores(holder, expected_name)
        identity_pass = float(identity.get("identity_combined_score") or 0.0) >= 65.0 and bool(
            identity.get("token_gate_pass", True)
        )
        if exp_nik and nik == exp_nik:
            identity_pass = True

    expiry_pass: bool | None = None
    reason = ""
    if is_skck:
        start_day = date.fromisoformat(valid_from) if valid_from else None
        until_day = date.fromisoformat(valid_until) if valid_until else None
        if until_day is None:
            expiry_pass = False
            reason = "Tanggal kadaluarsa (Sampai dengan) tidak terbaca."
        elif until_day < as_of:
            expiry_pass = False
            reason = f"SKCK sudah kadaluarsa (Sampai dengan {valid_until})."
        elif start_day is not None and start_day > as_of:
            expiry_pass = False
            reason = f"SKCK belum berlaku (Berlaku untuk {valid_from})."
        else:
            expiry_pass = True
            reason = f"SKCK masih aktif sampai {valid_until}."

    matched = bool(is_skck and expiry_pass)
    if identity_pass is not None:
        matched = matched and identity_pass

    blockers: list[str] = []
    if not is_skck:
        blockers.append("DOCUMENT_TYPE")
    if expiry_pass is False:
        blockers.append("SKCK_EXPIRY")
    if identity_pass is False:
        blockers.append("IDENTITY")

    summary = "SKCK lolos: nama cocok dan masa berlaku belum lewat."
    if not is_skck:
        summary = "Bukan SKCK."
    elif identity_pass is False:
        summary = "Nama pada SKCK tidak cocok."
    elif expiry_pass is False:
        summary = reason

    return {
        "document_profile_id": "skck",
        "validation_engine": "ornith",
        "document_type_pass": is_skck,
        "document_matched": matched,
        "holder_name": holder,
        "identity_pass": identity_pass,
        "identity": identity,
        "skck_expiry_pass": expiry_pass,
        "skck_valid_from": valid_from,
        "skck_valid_until": valid_until,
        "issued_date": valid_from,
        "issued_date_display": format_id_date(valid_from),
        "skck_expiry": {
            "today": as_of.isoformat(),
            "valid_from": valid_from,
            "valid_until": valid_until,
            "reason": reason,
        },
        "explanation": {"primary_blockers": blockers, "summary": summary},
        "summary": summary,
    }


def analyze_skck_image(
    image_bytes: bytes,
    *,
    document_type: str = "skck",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    del document_type
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_SKCK_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_SKCK_AI_MODEL"))
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
    timeout_s = float(
        os.environ.get("LLM_SKCK_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
    )
    content, meta = complete_vision(
        prepared,
        skck_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=SKCK_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_SKCK_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        extra=extra,
    )
    llm = {
        "base_url": base_url,
        "model": model,
        "latency_s": meta.get("latency_s"),
        "success": meta.get("success"),
        "error": meta.get("error"),
    }
    parsed = _parse_health_json(content) if content else None
    if not parsed:
        retry_text, retry_meta = complete_vision(
            prepared,
            "Balas hanya JSON.\n"
            '{"is_skck":true,"holder_name":"","identity_number":"","valid_from":"","valid_until":"","confidence":80,"notes":""}',
            image_mime="image/jpeg",
            system_text="Kamu membalas hanya satu objek JSON.",
            max_tokens=400,
            temperature=0.0,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            extra=extra,
        )
        if retry_text:
            content = retry_text
            parsed = _parse_health_json(retry_text)
            if isinstance(retry_meta.get("latency_s"), (int, float)):
                llm["latency_s"] = float(llm.get("latency_s") or 0) + float(retry_meta["latency_s"])
    if not parsed:
        return {
            "error": "empty_content" if not content else "invalid_json",
            "llm": llm,
            "raw_content": content,
            "image_prepare": prep_meta,
        }
    validation = skck_validation_from_extraction(
        parsed,
        expected_name=expected_name,
        expected_nik=expected_nik,
        today=today,
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
