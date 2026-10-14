"""Validasi bukti rekening tabungan via Ornith.

Nama di aplikasi bank boleh tidak lengkap atau tersensor: "Deva" dan "Dxxa"
dianggap cocok dengan "Deva Apriana".
"""

from __future__ import annotations

import os
import re
from typing import Any

from systems.validation.bank_rekening import (
    MANDIRI_ACCOUNT_PREFIXES,
    bank_label,
    is_placeholder_account,
    normalize_account_digits,
    normalize_expected_bank,
    validate_account_in_ocr,
)
from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import (
    _rekening_identity_and_account_pass,
    build_document_verdict,
    compare_extracted_identity_scores,
)
from systems.validation.keterangan_kesehatan_ai import (
    _parse_health_json,
    assert_allowed_llm_url,
    health_ai_base_url,
    health_ai_model,
)
from systems.validation.llm_fallback import complete_vision
from systems.validation.user_errors import attach_user_errors

REKENING_SYSTEM_PROMPT = """Kamu membaca bukti rekening tabungan dari aplikasi bank Indonesia (bukan mutasi/e-statement).
Jangan mengarang. Salin teks yang terlihat, termasuk huruf yang disensor.

Kembalikan HANYA satu objek JSON (tanpa markdown):
- is_savings_account (boolean): true jika ini halaman bukti/detail rekening tabungan.
  false jika mutasi, e-statement, KTP, atau dokumen lain.
- is_statement (boolean): true jika ini mutasi atau e-statement.
- bank (string): salah satu "mandiri", "mas", "lain", atau "" jika tidak terlihat.
- holder_name (string): nama orang pemilik rekening.
  Utamakan nama yang tercetak pada kartu debit/ATM (di bawah gambar kartu, sering dekat logo GPN)
  atau label Nama / Nama Pemilik / Atas Nama.
  Jangan isi nama produk, alias rekening, atau judul layar.
  Bukan "Tabungan NOW", "Tabungan NOW IDR", "Tabungan Payroll", "Livin",
  "Detail Rekening", "Sebagai Rekening Utama", atau teks di samping ikon pensil.
  Jika nama disensor, salin sensornya apa adanya (mis. Dxxa, D**a).
- account_number (string): nomor rekening tabungan 10–13 digit.
  Biasanya angka besar tepat di atas atau di bawah tulisan jenis tabungan
  (misalnya di atas "Tabungan NOW IDR"). Salin angka itu.
  Bukan nomor kartu 16 digit yang tercetak di kartu debit.
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def rekening_user_prompt(*, expected_name: str = "", expected_bank: str = "") -> str:
    lines = [
        "Ambil nama pemilik dari kartu debit atau label nama orang, bukan nama produk Tabungan NOW.",
        "Ambil nomor rekening tabungan, bukan nomor kartu 16 digit.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi pemilik: {expected_name.strip()}")
    if expected_bank.strip():
        lines.insert(1, f"Bank yang diharapkan: {expected_bank.strip()}")
    return "\n".join(lines)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


_TOKEN_RE = re.compile(r"[a-z0-9*•●×]+", re.I)
_MASK_CHARS = set("x*•●×")


def _name_tokens(value: str) -> list[str]:
    # Tanda petik di nama (Sa'idah, Ma'arif) tidak memisahkan kata.
    text = re.sub(r"[''`´’]", "", value or "")
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(text)]


def _mask_matches_token(pattern: str, token: str) -> bool:
    """Dxxa cocok dengan Deva: panjang sama, huruf x/*/• adalah sensor."""
    if not any(ch in _MASK_CHARS for ch in pattern):
        return False
    if len(pattern) != len(token) or len(pattern) < 2:
        return False
    revealed = 0
    for left, right in zip(pattern, token):
        if left in _MASK_CHARS:
            continue
        revealed += 1
        if left != right:
            return False
    return revealed >= 2


def rekening_holder_name_matches(expected: str, extracted: str) -> dict[str, Any]:
    """Cocok jika nama lengkap, sebagian kata, atau kata tersensor masih menunjuk ke nama referensi."""
    exp = _name_tokens(expected)
    got = _name_tokens(extracted)
    scores = compare_extracted_identity_scores(extracted, expected) if exp and got else None
    full_name = (
        scores
        and scores.get("token_gate_pass")
        and float(scores.get("identity_combined_score") or 0) >= 65
        and len(got) >= len(exp)
    )
    if full_name:
        return {"pass": True, "method": "full", "identity": scores}

    if not exp or not got:
        return {"pass": False, "method": "none", "identity": scores}

    method = "partial"
    for token in got:
        exact = token in exp and len(token) >= 3
        masked = any(_mask_matches_token(token, item) for item in exp)
        if masked:
            method = "masked"
        if not exact and not masked:
            return {"pass": False, "method": "none", "identity": scores}
    return {"pass": True, "method": method, "identity": scores}


def rekening_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "rekening",
    expected_name: str = "",
    expected_bank: str = "",
    expected_account: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "rekening"
    is_savings = _as_bool(parsed.get("is_savings_account"))
    is_statement = _as_bool(parsed.get("is_statement"))
    holder = str(parsed.get("holder_name") or "").strip()
    account_raw = str(parsed.get("account_number") or "").strip()
    exp_name = (expected_name or "").strip()
    exp_bank = normalize_expected_bank(expected_bank or "")
    exp_acct = normalize_account_digits(expected_account)
    if is_placeholder_account(exp_acct):
        exp_acct = ""
    detected_bank = normalize_expected_bank(str(parsed.get("bank") or ""))

    name_info = (
        rekening_holder_name_matches(exp_name, holder)
        if exp_name
        else {"pass": None, "method": "skipped", "identity": None}
    )
    name_pass = name_info["pass"] if exp_name else None
    identity = name_info.get("identity")

    account_info = validate_account_in_ocr(account_raw, exp_acct) if len(exp_acct) >= 10 else None
    account_pass = account_info.get("account_pass") if account_info else None
    acct_digits = normalize_account_digits(account_raw)

    if detected_bank is None and account_pass is True:
        if len(exp_acct) == 13 and exp_acct[:3] in MANDIRI_ACCOUNT_PREFIXES:
            detected_bank = "mandiri"
        elif len(exp_acct) == 10 and exp_acct.startswith(("100", "120")):
            detected_bank = "mas"

    if not exp_bank:
        bank_pass = False
    elif detected_bank == exp_bank:
        bank_pass = True
    elif detected_bank is None and account_pass is True:
        bank_pass = False
    else:
        bank_pass = detected_bank == exp_bank

    type_pass = is_savings and not is_statement
    identity_or_account = _rekening_identity_and_account_pass(
        has_name_ref=bool(exp_name),
        has_account_ref=len(exp_acct) >= 10,
        identity_pass=name_pass if isinstance(name_pass, bool) else None,
        account_pass=account_pass if isinstance(account_pass, bool) else None,
    )
    matched = type_pass and bank_pass is True and identity_or_account is True

    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_statement:
        blockers.append("EXCLUSION")
        detail_lines.append("Ini mutasi atau e-statement, bukan bukti rekening tabungan.")
    elif is_savings:
        detail_lines.append("Gambar dikenali sebagai bukti rekening tabungan.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan bukti rekening tabungan.")
    if bank_pass is True:
        detail_lines.append(f"Bank cocok: {bank_label(exp_bank or detected_bank or '')}.")
    elif exp_bank:
        blockers.append("BANK")
        seen = bank_label(detected_bank) if detected_bank else "tidak terbaca"
        detail_lines.append(f"Bank tidak cocok. Di gambar: {seen}.")
    if name_pass is True:
        how = {"full": "nama lengkap", "partial": "sebagian nama", "masked": "nama tersensor"}.get(
            str(name_info.get("method")), "nama"
        )
        detail_lines.append(f"Nama pemilik cocok ({how}): {holder or '—'}.")
    elif name_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append(f"Nama pemilik tidak cocok: {holder or 'tidak terbaca'}.")
    if account_pass is True:
        detail_lines.append("Nomor rekening cocok.")
    elif account_pass is False:
        blockers.append("ACCOUNT")
        detail_lines.append("Nomor rekening tidak cocok.")

    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    if matched:
        summary = "Rekening tabungan valid."
    else:
        summary = "Rekening tidak valid: " + "; ".join(detail_lines[:2])

    identity_pass = name_pass if isinstance(name_pass, bool) else None
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
        "method": f"ornith_rekening_{name_info.get('method')}",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=exp_name,
        identity_pass=identity_pass,
        identity=identity if isinstance(identity, dict) else None,
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
            "is_savings_account": is_savings,
            "is_statement": is_statement,
            "holder_name": holder,
            "account_number": account_raw,
            "account_digits": acct_digits,
            "name_pass": name_pass,
            "name_match_method": name_info.get("method"),
            "bank_pass": bank_pass,
            "account_pass": account_pass,
            "identity_pass": identity_pass,
            "identity_or_account_pass": identity_or_account,
            "expected_bank": exp_bank,
            "detected_bank": detected_bank,
            "detected_bank_label": bank_label(detected_bank) if detected_bank else None,
            "document_type_pass": type_pass,
            "document_matched": matched,
            "identity": identity if isinstance(identity, dict) else None,
            "name_extraction": name_extraction,
            "exclusion_violated": is_statement,
            "keywords": [],
            "excluded_keywords": [],
            "rekening_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_rekening_image(
    image_bytes: bytes,
    *,
    document_type: str = "rekening",
    expected_name: str = "",
    expected_bank: str = "",
    expected_account: str = "",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(
        health_ai_base_url(llm_base_url or os.environ.get("LLM_REKENING_AI_BASE_URL"))
    )
    model = health_ai_model(llm_model or os.environ.get("LLM_REKENING_AI_MODEL"))
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
        rekening_user_prompt(expected_name=expected_name, expected_bank=expected_bank),
        image_mime="image/jpeg",
        system_text=REKENING_SYSTEM_PROMPT,
        max_tokens=int(os.environ.get("LLM_REKENING_AI_MAX_TOKENS", "512")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get(
                "LLM_REKENING_AI_TIMEOUT_S",
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
    validation = rekening_validation_from_extraction(
        parsed,
        document_type=document_type,
        expected_name=expected_name,
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
