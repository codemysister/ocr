"""Validasi kartu BPJS Ketenagakerjaan via Ornith."""

from __future__ import annotations

from typing import Any

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict
from systems.validation.ornith_call import as_bool, decide_identity, run_ornith_json
from systems.validation.user_errors import attach_user_errors

BPJS_TK_SYSTEM_PROMPT = """Kamu membaca kartu peserta BPJS Ketenagakerjaan.
Jangan mengarang. Salin teks yang terlihat.

Kembalikan HANYA satu objek JSON.
- is_bpjs_tk (boolean): true hanya jika ini kartu BPJS Ketenagakerjaan / Jaminan Ketenagakerjaan.
  false jika BPJS Kesehatan (KIS), info iuran, info peserta JKN, KTP, atau dokumen lain.
- holder_name (string): nama peserta di kartu, apa adanya (boleh terpotong).
- nik (string): NIK 16 digit jika terlihat, else "".
- confidence (number 0-100)
- notes (string): maksimal 12 kata
"""


def bpjs_tk_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "bpjs_tk",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "bpjs_tk"
    is_doc = as_bool(parsed.get("is_bpjs_tk"))
    holder = str(parsed.get("holder_name") or "").strip()
    ident = decide_identity(
        expected_name=expected_name,
        holder=holder,
        expected_nik=expected_nik,
        nik=str(parsed.get("nik") or ""),
    )
    identity_pass = ident["identity_pass"]
    matched = is_doc and identity_pass is not False
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_doc:
        detail_lines.append("Gambar dikenali sebagai kartu BPJS Ketenagakerjaan.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan kartu BPJS Ketenagakerjaan.")
    if ident["name_pass"] is True:
        detail_lines.append(f"Nama peserta cocok: {holder}.")
    elif ident["nik_pass"] is True:
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
        "BPJS Ketenagakerjaan valid."
        if matched
        else "BPJS Ketenagakerjaan tidak valid: " + "; ".join(detail_lines[:2])
    )
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
        "method": f"ornith_bpjs_tk_{ident['method']}",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=(expected_name or "").strip(),
        identity_pass=identity_pass,
        identity=ident["identity"],
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
            "is_bpjs_tk": is_doc,
            "holder_name": holder,
            "nik": ident["nik"],
            "name_pass": ident["name_pass"],
            "nik_pass": ident["nik_pass"],
            "name_match_method": ident["method"],
            "document_type_pass": is_doc,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": ident["identity"],
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "bpjs_tk_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_bpjs_tk_image(
    image_bytes: bytes,
    *,
    document_type: str = "bpjs_tk",
    expected_name: str = "",
    expected_nik: str = "",
    **_: Any,
) -> dict[str, Any]:
    lines = [
        "Apakah ini kartu BPJS Ketenagakerjaan? Ambil nama peserta dan NIK.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi peserta: {expected_name.strip()}")
    raw = run_ornith_json(
        image_bytes,
        user_prompt="\n".join(lines),
        system_text=BPJS_TK_SYSTEM_PROMPT,
        env_prefix="LLM_BPJS_TK_AI",
    )
    if raw.get("error"):
        return raw
    validation = bpjs_tk_validation_from_extraction(
        raw["parsed"],
        document_type=document_type,
        expected_name=expected_name,
        expected_nik=expected_nik,
        llm=raw.get("llm"),
    )
    return {
        "error": None,
        "llm": raw.get("llm"),
        "extraction": raw["parsed"],
        "validation": validation,
        "document_matched": validation.get("document_matched"),
        "image_prepare": raw.get("image_prepare"),
        "raw_content": raw.get("raw_content"),
    }
