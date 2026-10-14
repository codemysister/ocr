"""Validasi kartu/sertifikat vaksinasi COVID-19 booster (dosis 3) via Ornith."""

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

VAKSINASI_3_SYSTEM_PROMPT = """Kamu membaca kartu atau sertifikat vaksinasi COVID-19 Indonesia.
Jangan mengarang. Salin teks yang terlihat.

Kembalikan HANYA satu objek JSON valid. Kunci wajib pakai tanda kutip ganda.
Jangan awali dengan kata JSON, jangan pakai bullet, jangan pakai markdown.
- is_covid_vaccine (boolean): true jika ini kartu/sertifikat/surat vaksin COVID-19.
  false jika KTP, BPJS, info peserta, info iuran, atau dokumen lain.
- is_dose_3 (boolean): true jika riwayat memuat booster atau dosis ketiga.
  "booster", "vaksin booster", "booster pertama", "dosis ketiga", "dosis 3",
  "tahap 3", dan "3rd dose" adalah booster.
  "vaksin primer 1" dan "vaksin primer 2" bukan booster.
  Tetap true meskipun tahap 1 atau tahap 2 juga tercantum.
  false jika hanya dosis 1, hanya dosis 2 / primer 2, atau booster tidak terlihat.
- holder_name (string): nama penerima vaksin (Nama Lengkap), bukan nama kontak catatan.
- nik (string): NIK 16 digit jika terlihat, else "".
- dose_label (string): salin persis tulisan booster yang terlihat, mis. "VAKSIN BOOSTER".
  Jangan dikosongkan jika tulisan itu ada. Jangan isi "VAKSIN PRIMER 2" sebagai booster.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def vaksinasi_3_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah ini bukti vaksin COVID-19 booster (dosis ketiga)? Ambil nama penerima dan NIK.",
        "Booster tetap true bila tahap 1 atau tahap 2 juga ada. Vaksin primer 2 bukan booster.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi penerima: {expected_name.strip()}")
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


def _dose_text(*parts: Any) -> str:
    blob = " ".join(str(part or "") for part in parts).casefold()
    blob = re.sub(r"[\s\-]", "", blob)
    blob = re.sub(r"bukan(?:ini)?(?:adalah)?(?:kartu)?(?:vaksin)?booster(?:pertama)?", "", blob)
    blob = re.sub(r"bukan(?:dosis)?(?:ketiga|3)", "", blob)
    blob = re.sub(r"bukantahap3", "", blob)
    blob = re.sub(r"(?:vaksin)?primer[12]", "", blob)
    return blob


def _is_dose_3(parsed: dict[str, Any]) -> bool:
    label_blob = re.sub(r"[\s\-]", "", str(parsed.get("dose_label") or "").casefold())
    label_primer = "primer1" in label_blob or "primer2" in label_blob
    label_booster = any(
        marker in label_blob for marker in ("booster", "dosisketiga", "dosis3", "tahap3", "3rddose")
    )
    if label_primer and not label_booster:
        return False
    raw = re.sub(
        r"[\s\-]",
        "",
        f"{parsed.get('dose_label') or ''} {parsed.get('notes') or ''}".casefold(),
    )
    raw = re.sub(r"bukanbooster(?:pertama)?", "", raw)
    raw = re.sub(r"bukandosisketiga", "", raw)
    has_primer = "primer1" in raw or "primer2" in raw
    has_booster = any(marker in raw for marker in ("booster", "dosisketiga", "dosis3", "tahap3", "3rddose"))
    if has_primer and not has_booster:
        return False
    if _as_bool(parsed.get("is_dose_3")):
        return True
    blob = _dose_text(parsed.get("dose_label"), parsed.get("notes"))
    return any(
        marker in blob
        for marker in ("booster", "dosisketiga", "dosis3", "tahap3", "3rddose")
    )


def vaksinasi_3_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "vaksinasi_3",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "vaksinasi_3"
    is_card = _as_bool(parsed.get("is_covid_vaccine"))
    dose_3 = _is_dose_3(parsed)
    type_pass = is_card and dose_3
    holder = str(parsed.get("holder_name") or "").strip()
    nik = _digits(str(parsed.get("nik") or ""))
    if len(nik) != 16:
        nik = ""
    dose_label = str(parsed.get("dose_label") or "").strip()
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

    matched = type_pass and identity_pass is not False
    blockers: list[str] = []
    detail_lines: list[str] = []
    if type_pass:
        detail_lines.append("Gambar dikenali sebagai vaksinasi COVID-19 booster.")
    elif is_card:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Ini bukti vaksin, tetapi bukan booster.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan kartu atau sertifikat vaksinasi COVID-19.")
    if name_pass is True:
        detail_lines.append(f"Nama penerima cocok: {holder}.")
    elif nik_pass is True:
        detail_lines.append("NIK di kartu cocok dengan referensi.")
    elif identity_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append(f"Nama penerima tidak cocok: {holder or 'tidak terbaca'}.")

    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    summary = (
        "Vaksinasi booster valid."
        if matched
        else "Vaksinasi booster tidak valid: " + "; ".join(detail_lines[:2])
    )
    detection = {
        "detected_profile_id": profile if type_pass else None,
        "detected_document_type": profile_label(profile) if type_pass else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": holder or None,
        "candidate_normalized": holder,
        "method": f"ornith_vaksinasi_3_{name_method}",
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
        document_type_pass=type_pass,
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
            "is_covid_vaccine": is_card,
            "is_dose_3": dose_3,
            "holder_name": holder,
            "nik": nik,
            "dose_label": dose_label,
            "name_pass": name_pass,
            "nik_pass": nik_pass,
            "name_match_method": name_method,
            "document_type_pass": type_pass,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": identity,
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "vaksinasi_3_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_vaksinasi_3_image(
    image_bytes: bytes,
    *,
    document_type: str = "vaksinasi_3",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_VAKSINASI_3_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_VAKSINASI_3_AI_MODEL"))
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
        vaksinasi_3_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=VAKSINASI_3_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_VAKSINASI_3_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get(
                "LLM_VAKSINASI_3_AI_TIMEOUT_S",
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
    validation = vaksinasi_3_validation_from_extraction(
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
