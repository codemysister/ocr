"""Validasi layar Info Iuran JKN via Ornith."""

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

IURAN_SYSTEM_PROMPT = """Kamu membaca bukti iuran BPJS Kesehatan / JKN.
Jangan mengarang. Jangan menulis analisis. Jangan memakai tag think.

Kembalikan HANYA satu objek JSON:
- is_info_iuran (boolean): true untuk semua bentuk berikut.
  1. Layar Info Iuran (tagihan atau sisa saldo).
  2. Modal "tidak memiliki tagihan pribadi" atau "jenis peserta tidak terkategori",
     meskipun latar belakangnya Menu Lainnya atau ikon Info Peserta.
  3. Layar Info Riwayat Pembayaran / Pembayaran Iuran, termasuk tulisan
     "Peserta tidak memiliki riwayat pembayaran iuran".
  4. Chatbot Pandawa atau WhatsApp BPJS Kesehatan yang menampilkan
     Nama Peserta, Tagihan, dan Status Pembayaran.
  false jika hanya Info Peserta (faskes), kartu KIS, BPJS Ketenagakerjaan, vaksin, atau KTP.
- holder_name (string): nama peserta yang terlihat, termasuk huruf X yang menyensor.
  Jika ada beberapa nama, salin yang paling mirip dengan nama referensi.
  Kosong jika nama tidak tampil.
- name_displayed (boolean): false jika nama peserta tidak tampil.
- modal_tanpa_tagihan (boolean): true jika tidak ada tagihan pribadi,
  atau peserta tidak memiliki riwayat pembayaran iuran.
- nik (string): NIK 16 digit jika terlihat di layar atau di chat, else "".
- confidence (number 0-100)
- notes (string): sebut format yang terlihat, maksimal 12 kata
"""


def iuran_user_prompt(*, expected_name: str = "") -> str:
    lines = [
        "Apakah ini bukti iuran JKN (layar Info Iuran, modal tanpa tagihan, riwayat pembayaran, atau chat Pandawa)?",
        "Salin nama peserta apa adanya, termasuk huruf X. Balas hanya satu objek JSON.",
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


def iuran_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "iuran",
    expected_name: str = "",
    expected_nik: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "iuran"
    is_doc = _as_bool(parsed.get("is_info_iuran"))
    holder = str(parsed.get("holder_name") or "").strip()
    modal = _as_bool(parsed.get("modal_tanpa_tagihan"))
    notes_blob = str(parsed.get("notes") or "").casefold()
    compact = re.sub(r"[\s\-]", "", notes_blob)
    modal_notes = (
        "tidakmemilikitagihanpribadi" in compact
        or "jenispresentidakterkategori" in compact
        or "tidakmemilikiriwayatpembayaran" in compact
    )
    riwayat = "inforiwayatpembayaran" in compact or "pembayaran iuran" in notes_blob
    pandawa = "namapeserta" in compact and ("tagihan" in compact or "statuspembayaran" in compact)
    if modal_notes or riwayat or pandawa:
        is_doc = True
    if modal_notes or "tidakmemilikiriwayatpembayaran" in compact:
        modal = True
    name_displayed = _as_bool(parsed.get("name_displayed")) if "name_displayed" in parsed else bool(holder)
    nik = _digits(str(parsed.get("nik") or ""))
    if len(nik) != 16:
        nik = ""
    exp_name = (expected_name or "").strip()
    exp_nik = _digits(expected_nik)
    if len(exp_nik) != 16:
        exp_nik = ""

    name_info = (
        rekening_holder_name_matches(exp_name, holder)
        if exp_name and holder
        else {"pass": False if exp_name and holder == "" else None, "method": "none", "identity": None}
    )
    if not exp_name:
        name_info = {"pass": None, "method": "skipped", "identity": None}
    name_pass = name_info["pass"]
    nik_pass: bool | None = None
    if exp_nik and nik:
        nik_pass = nik == exp_nik
    elif exp_nik and name_displayed:
        nik_pass = False

    if modal and is_doc and not holder:
        identity_pass: bool | None = True
        name_method = "modal"
        name_pass = True
    elif nik_pass is True:
        identity_pass = True
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
        detail_lines.append("Gambar dikenali sebagai layar Info Iuran.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan layar Info Iuran.")
    if name_method == "modal":
        detail_lines.append("Modal tanpa tagihan pribadi; nama peserta tidak ditampilkan.")
    elif name_pass is True:
        detail_lines.append(f"Nama peserta cocok: {holder}.")
    elif nik_pass is True:
        detail_lines.append("NIK pada layar cocok dengan referensi.")
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
        "Info Iuran valid."
        if matched
        else "Info Iuran tidak valid: " + "; ".join(detail_lines[:2])
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
        "method": f"ornith_iuran_{name_method}",
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
            "is_info_iuran": is_doc,
            "modal_tanpa_tagihan": modal,
            "holder_name": holder,
            "nik": nik,
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
            "iuran_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_iuran_image(
    image_bytes: bytes,
    *,
    document_type: str = "iuran",
    expected_name: str = "",
    expected_nik: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_IURAN_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_IURAN_AI_MODEL"))
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
        os.environ.get("LLM_IURAN_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
    )
    content, meta = complete_vision(
        prepared,
        iuran_user_prompt(expected_name=expected_name),
        image_mime="image/jpeg",
        system_text=IURAN_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_IURAN_AI_MAX_TOKENS", "512")),
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
            '{"is_info_iuran":true,"holder_name":"","name_displayed":false,"modal_tanpa_tagihan":false,"nik":"","confidence":80,"notes":""}',
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
    validation = iuran_validation_from_extraction(
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
