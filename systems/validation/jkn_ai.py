"""Validasi layar Info Peserta JKN via Ornith.

Status AKTIF / TIDAK AKTIF diekstrak dan tidak menggagalkan validasi.
"""

from __future__ import annotations

import json
import re
from typing import Any

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict
from systems.validation.bpjs_ai import (
    _cards_in_text,
    norm_card_number,
    participant_ids_conflict,
    payload_cards_mismatch,
)
from systems.validation.ornith_call import as_bool, nik16, run_ornith_json
from systems.validation.rekening_ai import rekening_holder_name_matches
from systems.validation.user_errors import attach_user_errors

JKN_SYSTEM_PROMPT = """Kamu membaca layar Info Peserta Mobile JKN / kartu peserta JKN.
Jangan mengarang.

Kembalikan HANYA satu objek JSON.
- is_info_peserta (boolean): true hanya jika ada kartu peserta yang terlihat
  (nama, nomor peserta, NIK, tanggal lahir, atau Faskes).
  false jika yang terlihat hanya judul "Info Peserta" dan "Kepesertaan terdaftar" tanpa data peserta.
  false untuk Kartu Indonesia Sehat / kartu BPJS (header hijau, barcode, label Nomor Kartu).
  false juga untuk Info Iuran, sisa saldo, tagihan, total tagihan, riwayat pembayaran,
  mutasi, kartu BPJS Ketenagakerjaan, atau KTP.
  Daftar keluarga yang tiap kartunya berisi "Sisa Saldo" dan "Tagihan" adalah Info Iuran, bukan Info Peserta.
- is_kis_card (boolean): true jika gambar adalah Kartu Indonesia Sehat, bukan layar Info Peserta.
- is_info_iuran (boolean): true jika judulnya Info Iuran atau ada tagihan/sisa saldo.
- has_tagihan (boolean): true jika nominal tagihan atau total tagihan terlihat.
- members (array): peserta yang terlihat. Setiap item
  {"name": string, "nik": string, "card_number": string, "status": "AKTIF" atau "TIDAK AKTIF" atau ""}.
  card_number adalah nomor peserta di kartu Info Peserta, bukan NIK. Jika satu peserta, tetap pakai array.
  Jika tidak ada peserta, isi array kosong. Jangan mengarang nama dari judul layar.
- bpjs_card_number (string): nomor kartu pada Kartu Indonesia Sehat jika ikut terlihat di gambar yang sama, else "".
- bpjs_nik (string): NIK pada kartu KIS jika ikut terlihat, else "".
  Jika nomor Info Peserta dan nomor KIS berbeda, salin keduanya apa adanya. Jangan disamakan.
- confidence (number 0-100)
- notes (string): maksimal 12 kata
"""


def _participant_name(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", (name or "").casefold())
    if compact in {"jenistampilan", "kepesertaanterdaftar", "infopeserta", "ubah", "mobilejkn"}:
        return ""
    return (name or "").strip()


def _status_code(value: str) -> str | None:
    text = " ".join((value or "").casefold().split())
    if not text:
        return None
    if "tidak" in text:
        return "tidak_aktif"
    if "aktif" in text:
        return "aktif"
    return None


def _members(parsed: dict[str, Any]) -> list[dict[str, str]]:
    raw = parsed.get("members")
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _participant_name(str(item.get("name") or item.get("nama") or ""))
        nik = nik16(str(item.get("nik") or ""))
        card = str(item.get("card_number") or item.get("nomor_kartu") or "").strip()
        status = str(item.get("status") or "").strip()
        if name or nik or norm_card_number(card):
            rows.append({"name": name, "nik": nik, "card_number": card, "status": status})
    if not rows:
        name = _participant_name(str(parsed.get("holder_name") or ""))
        if name:
            rows.append(
                {
                    "name": name,
                    "nik": nik16(str(parsed.get("nik") or "")),
                    "card_number": str(parsed.get("card_number") or "").strip(),
                    "status": "",
                }
            )
    return rows


def jkn_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "jkn",
    expected_name: str = "",
    expected_nik: str = "",
    expected_card_number: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "jkn"
    is_doc = as_bool(parsed.get("is_info_peserta"))
    notes_raw = str(parsed.get("notes") or "")
    compact = re.sub(r"[^a-z0-9]", "", notes_raw.casefold())
    iuran_screen = as_bool(parsed.get("is_info_iuran")) or as_bool(parsed.get("has_tagihan")) or any(
        token in compact
        for token in ("infoiuran", "sisasaldo", "totaltagihan", "tagihan", "pembayaraniuran")
    )
    blob = re.sub(r"[^a-z0-9]", "", json.dumps(parsed, ensure_ascii=False).casefold())
    peserta_app = any(token in blob for token in ("kepesertaanterdaftar", "jenistampilan"))
    kis_only = (as_bool(parsed.get("is_kis_card")) or "kartuindonesiasehat" in blob) and not peserta_app
    if iuran_screen or kis_only:
        is_doc = False
    rows = _members(parsed)
    kis_card = str(parsed.get("bpjs_card_number") or "").strip()
    kis_nik = str(parsed.get("bpjs_nik") or "").strip()
    member_ids = [row.get("card_number", "") for row in rows] + [row.get("nik", "") for row in rows]
    numbers_conflict = participant_ids_conflict(member_ids, [kis_card, kis_nik])
    if not numbers_conflict and "kartuindonesiasehat" in compact:
        if len(_cards_in_text(json.dumps(parsed, ensure_ascii=False))) > 1:
            numbers_conflict = True
    member_cards = [row.get("card_number", "") for row in rows]
    payload_mismatch = payload_cards_mismatch(expected_card_number, member_cards, require_all=False)
    exp_card = norm_card_number(expected_card_number)
    kis_norm = norm_card_number(kis_card)
    if exp_card and kis_norm and kis_norm != exp_card:
        payload_mismatch = True
    has_participant = any(row["name"] or row["nik"] or norm_card_number(row.get("card_number", "")) for row in rows)
    if not has_participant or kis_only:
        is_doc = False
        payload_mismatch = False
        numbers_conflict = False
    if payload_mismatch:
        numbers_conflict = True
    exp_name = (expected_name or "").strip()
    exp_nik = nik16(expected_nik)
    matched_row: dict[str, str] | None = None
    name_info: dict[str, Any] = {"pass": None, "method": "skipped", "identity": None}
    if exp_name:
        name_info = {"pass": False, "method": "none", "identity": None}
        for row in rows:
            if not row["name"]:
                continue
            info = rekening_holder_name_matches(exp_name, row["name"])
            if info.get("pass") is True:
                name_info = info
                matched_row = row
                break
    nik_hit = ""
    if exp_nik:
        for row in rows:
            if row["nik"] == exp_nik:
                nik_hit = row["nik"]
                matched_row = matched_row or row
                break
    name_pass = name_info.get("pass")
    nik_pass: bool | None = bool(nik_hit) if exp_nik else None
    if not exp_name and not exp_nik:
        identity_pass: bool | None = None
        method = "skipped"
    elif name_pass is True or nik_pass is True:
        identity_pass = True
        method = "nik" if name_pass is not True else (name_info.get("method") or "name")
    else:
        identity_pass = False
        method = name_info.get("method") or "none"
    holder = (matched_row or (rows[0] if rows else {"name": ""}))["name"]
    status = _status_code((matched_row or {}).get("status") or "")
    if status is None and not exp_name and len(rows) == 1:
        status = _status_code(rows[0]["status"])
    status_display = {"aktif": "AKTIF", "tidak_aktif": "TIDAK AKTIF"}.get(status or "")
    matched = is_doc and identity_pass is not False and not numbers_conflict
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_doc:
        detail_lines.append("Gambar dikenali sebagai Info Peserta JKN.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan layar Info Peserta JKN.")
        if not has_participant and not iuran_screen and not kis_only:
            detail_lines.append("Data peserta tidak terlihat.")
        if kis_only:
            detail_lines.append("Gambar adalah Kartu Indonesia Sehat, bukan layar Info Peserta.")
        if iuran_screen:
            detail_lines.append("Gambar adalah Info Iuran, bukan Info Peserta.")
    if numbers_conflict:
        blockers.insert(0, "CARD_MISMATCH")
        detail_lines.insert(0, "Nomor kartu tidak sama dengan nomor yang dikirim.")
    if name_pass is True:
        detail_lines.append(f"Nama peserta cocok: {holder}.")
    elif nik_pass is True:
        detail_lines.append("NIK peserta cocok dengan referensi.")
    elif identity_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append(f"Nama peserta tidak cocok: {holder or 'tidak terbaca'}.")
    if status_display:
        detail_lines.append(f"Status peserta: {status_display}.")
    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    summary = "Info Peserta JKN valid." if matched else "JKN tidak valid: " + "; ".join(detail_lines[:2])
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
        "method": f"ornith_jkn_{method}",
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
    if numbers_conflict:
        verdict = dict(verdict)
        verdict["document_matched"] = False
        verdict["summary"] = "Nomor kartu tidak sama dengan nomor yang dikirim."
    return attach_user_errors(
        {
            "valid": matched,
            "document_type": (document_type or "").strip(),
            "document_profile_id": profile,
            "validation_mode": "image",
            "validation_engine": "ornith",
            "is_info_peserta": is_doc,
            "holder_name": holder,
            "nik": nik_hit or (rows[0]["nik"] if rows else ""),
            "members": rows,
            "bpjs_card_number": kis_card,
            "card_numbers_match": not numbers_conflict,
            "expected_card_number": exp_card,
            "name_pass": name_pass,
            "nik_pass": nik_pass,
            "name_match_method": method,
            "jkn_status_label": status,
            "jkn_status_display": status_display or None,
            "jkn_status": {
                "status": status,
                "status_display": status_display or None,
                "matched_name": holder or None,
                "blocking": False,
            },
            "document_type_pass": is_doc,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": identity,
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "jkn_ai": {"llm": llm or {}, "extraction": parsed},
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


def analyze_jkn_image(
    image_bytes: bytes,
    *,
    document_type: str = "jkn",
    expected_name: str = "",
    expected_nik: str = "",
    expected_card_number: str = "",
    **_: Any,
) -> dict[str, Any]:
    lines = [
        "Apakah ini layar Info Peserta JKN? Ambil nama, nomor peserta, NIK, dan status hanya jika kartu peserta terlihat.",
        "Judul Info Peserta tanpa nama atau nomor peserta bukan data. members kosong.",
        "Kartu Indonesia Sehat bukan Info Peserta. is_kis_card true dan is_info_peserta false.",
        "Jika kartu KIS ikut terlihat, isi bpjs_card_number terpisah. Jangan samakan nomor yang berbeda.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi peserta: {expected_name.strip()}")
    raw = run_ornith_json(
        image_bytes,
        user_prompt="\n".join(lines),
        system_text=JKN_SYSTEM_PROMPT,
        env_prefix="LLM_JKN_AI",
        max_tokens=768,
    )
    if raw.get("error"):
        return raw
    validation = jkn_validation_from_extraction(
        raw["parsed"],
        document_type=document_type,
        expected_name=expected_name,
        expected_nik=expected_nik,
        expected_card_number=expected_card_number,
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
