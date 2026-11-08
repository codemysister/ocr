"""Validasi Kartu Keluarga via Ornith."""

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

KK_SYSTEM_PROMPT = """Kamu membaca Kartu Keluarga (KK) Indonesia.
Jangan mengarang. Salin teks yang terlihat.

Kembalikan HANYA satu objek JSON valid. Kunci wajib pakai tanda kutip ganda.
Jangan awali dengan kata JSON, jangan pakai bullet, jangan pakai markdown.
- is_kartu_keluarga (boolean): true hanya jika ini Kartu Keluarga.
  false jika KTP, SIM, NPWP, akta, atau dokumen lain.
- family_card_number (string): nomor Kartu Keluarga jika terlihat, else "".
  Ini bukan NIK anggota.
- members (array): semua anggota yang terlihat, termasuk kepala keluarga.
  Setiap item objek {"name": string, "nik": string}.
  nik adalah NIK 16 digit anggota, else "".
  Jangan masukkan nomor KK ke nik.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def kk_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah ini Kartu Keluarga? Ambil nomor KK dan daftar anggota (nama + NIK).",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi anggota: {expected_name.strip()}")
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


def _hamming16(a: str, b: str) -> int:
    if len(a) != 16 or len(b) != 16:
        return 99
    return sum(x != y for x, y in zip(a, b))


def _member_rows(parsed: dict[str, Any]) -> list[dict[str, str]]:
    raw = parsed.get("members")
    items: list[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        items = []
    rows: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
            nik = ""
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("nama") or item.get("holder_name") or "").strip()
            nik = _digits(str(item.get("nik") or item.get("identity_number") or ""))
        else:
            continue
        if len(nik) != 16:
            nik = ""
        if name or nik:
            rows.append({"name": name, "nik": nik})
    kepala = str(parsed.get("kepala_keluarga") or parsed.get("head_name") or "").strip()
    if kepala and not any(row["name"].casefold() == kepala.casefold() for row in rows):
        rows.insert(0, {"name": kepala, "nik": ""})
    return rows


def _nik_hit(expected_nik: str, rows: list[dict[str, str]], family_no: str) -> str:
    exp = _digits(expected_nik)
    if len(exp) != 16:
        return ""
    cands = [row["nik"] for row in rows if row["nik"] and row["nik"] != family_no]
    if exp in cands:
        return exp
    for nik in cands:
        if _hamming16(nik, exp) <= 1:
            return nik
    return ""


def kk_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "kk",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "kk"
    is_kk = _as_bool(parsed.get("is_kartu_keluarga"))
    family_no = _digits(str(parsed.get("family_card_number") or ""))
    if len(family_no) != 16:
        family_no = ""
    rows = _member_rows(parsed)
    exp_name = (expected_name or "").strip()
    exp_nik = _digits(expected_nik)
    if len(exp_nik) != 16:
        exp_nik = ""

    name_info: dict[str, Any] = {"pass": None, "method": "skipped", "identity": None}
    matched_name = ""
    if exp_name:
        name_info = {"pass": False, "method": "none", "identity": None}
        for row in rows:
            if not row["name"]:
                continue
            info = rekening_holder_name_matches(exp_name, row["name"])
            if info.get("pass") is True:
                name_info = info
                matched_name = row["name"]
                break
            if name_info.get("identity") is None and info.get("identity"):
                name_info = info
    name_pass = name_info.get("pass")
    nik_found = _nik_hit(exp_nik, rows, family_no) if exp_nik else ""
    nik_pass: bool | None = None
    if exp_nik:
        nik_pass = bool(nik_found)

    want_identity = bool(exp_name or exp_nik)
    if not want_identity:
        identity_pass: bool | None = None
        name_method = "skipped"
    elif name_pass is True or nik_pass is True:
        identity_pass = True
        if name_pass is True and nik_pass is True:
            name_method = name_info.get("method") or "name"
        elif nik_pass is True:
            name_method = "nik"
        else:
            name_method = name_info.get("method") or "name"
    else:
        identity_pass = False
        name_method = name_info.get("method") or "none"

    if not matched_name and nik_found:
        for row in rows:
            if row["nik"] == nik_found or (row["nik"] and _hamming16(row["nik"], nik_found) <= 1):
                matched_name = row["name"]
                break
    holder = matched_name or (rows[0]["name"] if rows else "")

    matched = is_kk and identity_pass is not False
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_kk:
        detail_lines.append("Gambar dikenali sebagai Kartu Keluarga.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan Kartu Keluarga.")
    if name_pass is True:
        detail_lines.append(f"Nama anggota cocok: {matched_name}.")
    elif nik_pass is True:
        detail_lines.append("NIK anggota cocok dengan referensi.")
    elif identity_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append("Nama dan NIK anggota tidak cocok.")

    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    summary = (
        "Kartu Keluarga valid."
        if matched
        else "Kartu Keluarga tidak valid: " + "; ".join(detail_lines[:2])
    )
    detection = {
        "detected_profile_id": profile if is_kk else None,
        "detected_document_type": profile_label(profile) if is_kk else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": holder or None,
        "candidate_normalized": holder,
        "method": f"ornith_kk_{name_method}",
    }
    identity = name_info.get("identity") if isinstance(name_info.get("identity"), dict) else {}
    if want_identity:
        identity = {
            **identity,
            "kk_or_identity": True,
            "kk_name_pass": name_pass is True,
            "kk_nik_pass": nik_pass is True,
            "expected_nik": exp_nik,
            "ocr_nik_matched": nik_found,
            "note": "Identitas KK: nama atau NIK anggota cukup salah satu yang cocok (OR).",
        }
    else:
        identity = None
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=exp_name,
        identity_pass=identity_pass,
        identity=identity,
        identity_min_score=65.0,
        document_type_pass=is_kk,
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
            "is_kartu_keluarga": is_kk,
            "holder_name": holder,
            "family_card_number": family_no,
            "members": rows,
            "nik": nik_found or (rows[0]["nik"] if rows else ""),
            "name_pass": name_pass,
            "nik_pass": nik_pass,
            "name_match_method": name_method,
            "document_type_pass": is_kk,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": identity,
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "kk_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_kk_image(
    image_bytes: bytes,
    *,
    document_type: str = "kk",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_KK_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_KK_AI_MODEL"))
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
        kk_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=KK_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_KK_AI_MAX_TOKENS", "1024")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get("LLM_KK_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
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
    validation = kk_validation_from_extraction(
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
