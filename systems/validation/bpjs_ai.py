"""Validasi kartu BPJS Kesehatan (KIS) via Ornith."""

from __future__ import annotations

import os
import re
from typing import Any

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict, parse_nik16_from_filename
from systems.validation.keterangan_kesehatan_ai import (
    _parse_health_json,
    assert_allowed_llm_url,
    health_ai_base_url,
    health_ai_model,
)
from systems.validation.llm_fallback import complete_vision
from systems.validation.rekening_ai import rekening_holder_name_matches
from systems.validation.user_errors import attach_user_errors

BPJS_SYSTEM_PROMPT = """Kamu membaca kartu BPJS Kesehatan / Kartu Indonesia Sehat (KIS).
Jangan mengarang. Salin teks yang terlihat.

Kembalikan HANYA satu objek JSON valid. Kunci wajib pakai tanda kutip ganda.
Jangan awali dengan kata JSON, jangan pakai bullet, jangan pakai markdown.
- is_bpjs_kesehatan (boolean): true hanya jika ini kartu BPJS Kesehatan / Kartu Indonesia Sehat.
  false jika BPJS Ketenagakerjaan, info iuran, info peserta JKN, vaksin, KTP, atau dokumen lain.
- holder_name (string): nama peserta yang tercetak di kartu, apa adanya (boleh terpotong).
  Bukan nama faskes, alamat, atau judul "Kartu Indonesia Sehat".
- nik (string): NIK 16 digit jika terlihat, else "".
- card_number (string): nomor kartu BPJS jika terlihat, else "".
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def bpjs_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah ini kartu BPJS Kesehatan (KIS)? Ambil nama peserta dan NIK.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi peserta: {expected_name.strip()}")
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


def parse_bpjs_name_from_filename(filename: str) -> str:
    """Ambil nama dari pola `{prefix}_{Nama}_{NIK16}.ext`."""
    stem = (filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"\.(?:jpe?g|png|webp|gif|pdf)$", "", stem, flags=re.I)
    nik = parse_nik16_from_filename(stem) or ""
    if not nik or not stem.endswith("_" + nik):
        return ""
    stem = stem[: -(len(nik) + 1)]
    if "_" not in stem:
        return ""
    return stem.split("_", 1)[1].replace("_", " ").strip()


def bpjs_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "bpjs",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "bpjs"
    is_kis = _as_bool(parsed.get("is_bpjs_kesehatan"))
    holder = str(parsed.get("holder_name") or "").strip()
    nik = _digits(str(parsed.get("nik") or ""))
    if len(nik) != 16:
        nik = ""
    card_number = str(parsed.get("card_number") or "").strip()
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

    matched = is_kis and identity_pass is not False
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_kis:
        detail_lines.append("Gambar dikenali sebagai kartu BPJS Kesehatan (KIS).")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan kartu BPJS Kesehatan (KIS).")
    if name_pass is True:
        detail_lines.append(f"Nama peserta cocok: {holder}.")
    elif nik_pass is True:
        detail_lines.append("NIK di kartu cocok dengan referensi.")
    elif identity_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append(f"Nama peserta tidak cocok: {holder or 'tidak terbaca'}.")

    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    summary = (
        "Kartu BPJS Kesehatan valid."
        if matched
        else "BPJS Kesehatan tidak valid: " + "; ".join(detail_lines[:2])
    )
    detection = {
        "detected_profile_id": profile if is_kis else None,
        "detected_document_type": profile_label(profile) if is_kis else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": holder or None,
        "candidate_normalized": holder,
        "method": f"ornith_bpjs_{name_method}",
    }
    identity = name_info.get("identity") if isinstance(name_info.get("identity"), dict) else None
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=exp_name,
        identity_pass=identity_pass,
        identity=identity,
        identity_min_score=65.0,
        document_type_pass=is_kis,
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
            "is_bpjs_kesehatan": is_kis,
            "holder_name": holder,
            "nik": nik,
            "card_number": card_number,
            "name_pass": name_pass,
            "nik_pass": nik_pass,
            "name_match_method": name_method,
            "document_type_pass": is_kis,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": identity,
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "bpjs_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_bpjs_image(
    image_bytes: bytes,
    *,
    document_type: str = "bpjs",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_BPJS_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_BPJS_AI_MODEL"))
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
        bpjs_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=BPJS_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_BPJS_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get("LLM_BPJS_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
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
    validation = bpjs_validation_from_extraction(
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
