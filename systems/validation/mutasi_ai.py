"""Validasi mutasi / e-statement via Ornith.

Nama tidak dicek. Yang wajib: ini layar mutasi, bank cocok, dan nomor rekening pemilik cocok.
"""

from __future__ import annotations

import os
from typing import Any

from systems.validation.bank_rekening import (
    MANDIRI_ACCOUNT_PREFIXES,
    bank_label,
    normalize_account_digits,
    normalize_expected_bank,
    validate_account_in_ocr,
)
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

MUTASI_SYSTEM_PROMPT = """Kamu membaca layar mutasi rekening atau e-statement aplikasi bank Indonesia.
Jangan mengarang. Salin angka yang terlihat.

Kembalikan HANYA satu objek JSON valid. Kunci wajib pakai tanda kutip ganda.
Jangan awali dengan kata JSON, jangan pakai bullet, jangan pakai markdown.
- is_mutation (boolean): true jika ini daftar transaksi / mutasi / e-statement tabungan.
  Tetap true bila judul produk "Tabungan NOW" atau "Tabungan Payroll" terlihat bersama daftar transaksi.
  false jika hanya halaman detail rekening tanpa daftar transaksi, KTP, atau dokumen lain.
- bank (string): bank pemilik mutasi ini. Salah satu "mandiri", "mas", "lain", atau "".
  Livin, Tabungan NOW, dan Tabungan Payroll = "mandiri".
  Aplikasi Bank MAS / MAS Saving = "mas".
  Jangan mengisi "mas" hanya karena ada baris "Transfer dari BANK MAS".
- account_number (string): nomor rekening pemilik mutasi, 10–13 digit.
  Biasanya di header, dekat nama produk tabungan, bukan di baris lawan transfer.
  Bukan nomor kartu 16 digit. Bukan nomor rekening pengirim/penerima transfer.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def mutasi_user_prompt(*, expected_bank: str = "", expected_account: str = "") -> str:
    lines = [
        "Apakah ini mutasi atau e-statement? Ambil bank pemilik dan nomor rekening pemilik.",
        "Jangan ambil nomor rekening yang hanya muncul di baris transfer.",
        "Kembalikan JSON saja.",
    ]
    if expected_bank.strip():
        lines.insert(1, f"Bank yang diharapkan: {expected_bank.strip()}")
    if expected_account.strip():
        lines.insert(1, f"Nomor rekening referensi: {expected_account.strip()}")
    return "\n".join(lines)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


def _prefix_bank(digits: str) -> str | None:
    acct = normalize_account_digits(digits)
    if len(acct) == 12 and ("0" + acct)[:3] in MANDIRI_ACCOUNT_PREFIXES:
        acct = "0" + acct
    if len(acct) == 13 and acct[:3] in MANDIRI_ACCOUNT_PREFIXES:
        return "mandiri"
    if len(acct) == 10 and acct.startswith(("100", "120")):
        return "mas"
    return None


def mutasi_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "mutasi",
    expected_bank: str = "",
    expected_account: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "mutasi"
    is_mutation = _as_bool(parsed.get("is_mutation"))
    account_raw = str(parsed.get("account_number") or "").strip()
    exp_bank = normalize_expected_bank(expected_bank or "")
    exp_acct = normalize_account_digits(expected_account)
    detected_bank = normalize_expected_bank(str(parsed.get("bank") or ""))

    account_info = validate_account_in_ocr(account_raw, exp_acct) if len(exp_acct) >= 10 else None
    account_pass = account_info.get("account_pass") if account_info else None
    if len(exp_acct) < 10:
        account_pass = False

    prefix_bank = _prefix_bank(account_raw) or (_prefix_bank(exp_acct) if account_pass is True else None)
    if detected_bank is None:
        detected_bank = prefix_bank
    elif prefix_bank and detected_bank != prefix_bank and account_pass is True:
        detected_bank = prefix_bank

    if not exp_bank:
        bank_pass: bool | None = False
    else:
        bank_pass = detected_bank == exp_bank

    matched = is_mutation and bank_pass is True and account_pass is True
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_mutation:
        detail_lines.append("Gambar dikenali sebagai mutasi / e-statement.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan mutasi atau e-statement.")
    if bank_pass is True and exp_bank:
        detail_lines.append(f"Bank cocok: {bank_label(exp_bank)}.")
    elif exp_bank:
        blockers.append("BANK")
        seen = bank_label(detected_bank) if detected_bank else "tidak terbaca"
        detail_lines.append(f"Bank tidak cocok. Di gambar: {seen}.")
    else:
        blockers.append("BANK")
        detail_lines.append("Bank referensi wajib (mandiri atau mas).")
    if account_pass is True:
        detail_lines.append("Nomor rekening cocok.")
    else:
        blockers.append("ACCOUNT")
        detail_lines.append("Nomor rekening tidak cocok.")

    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    summary = (
        "Mutasi valid."
        if matched
        else "Mutasi tidak valid: " + "; ".join(detail_lines[:2])
    )
    detection = {
        "detected_profile_id": profile if is_mutation else None,
        "detected_document_type": profile_label(profile) if is_mutation else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_profile_no_identity",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name="",
        identity_pass=None,
        identity=None,
        identity_min_score=65.0,
        document_type_pass=is_mutation,
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
            "is_mutation": is_mutation,
            "account_number": account_raw,
            "account_digits": normalize_account_digits(account_raw),
            "bank_pass": bank_pass,
            "account_pass": account_pass,
            "identity_pass": None,
            "expected_bank": exp_bank,
            "detected_bank": detected_bank,
            "detected_bank_label": bank_label(detected_bank) if detected_bank else None,
            "document_type_pass": is_mutation,
            "document_matched": matched,
            "identity": None,
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "account_detection": account_info,
            "mutasi_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_mutasi_image(
    image_bytes: bytes,
    *,
    document_type: str = "mutasi",
    expected_bank: str = "",
    expected_account: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_MUTASI_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_MUTASI_AI_MODEL"))
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
        mutasi_user_prompt(expected_bank=expected_bank, expected_account=expected_account),
        image_mime="image/jpeg",
        system_text=MUTASI_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_MUTASI_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get("LLM_MUTASI_AI_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
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
    validation = mutasi_validation_from_extraction(
        parsed,
        document_type=document_type,
        expected_bank=expected_bank,
        expected_account=expected_account,
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
