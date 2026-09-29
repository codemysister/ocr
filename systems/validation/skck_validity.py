"""Gate SKCK: tanggal kadaluarsa (Sampai dengan / Until) harus belum terlewati."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

_ID_MONTH_NAMES = (
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
)


def format_id_date(iso: str | None) -> str | None:
    """Ubah YYYY-MM-DD menjadi '15 Juni 2026' untuk tampilan benchmark."""
    if not iso or not isinstance(iso, str):
        return None
    parts = iso.strip()[:10].split("-")
    if len(parts) != 3:
        return iso.strip() or None
    try:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{day} {_ID_MONTH_NAMES[month - 1]} {year}"
    except (ValueError, IndexError):
        return iso.strip() or None

_MONTHS: dict[str, int] = {
    "januari": 1,
    "january": 1,
    "jan": 1,
    "februari": 2,
    "february": 2,
    "feb": 2,
    "maret": 3,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "mei": 5,
    "may": 5,
    "juni": 6,
    "june": 6,
    "jun": 6,
    "juli": 7,
    "july": 7,
    "jul": 7,
    "agustus": 8,
    "august": 8,
    "agu": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "oktober": 10,
    "october": 10,
    "okt": 10,
    "oct": 10,
    "november": 11,
    "nopember": 11,
    "nov": 11,
    "desember": 12,
    "december": 12,
    "des": 12,
    "dec": 12,
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_TEXT_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\s+(20\d{{2}})\b",
    re.I,
)
_NUM_DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](20\d{2})\b")
_UNTIL_LABEL_RE = re.compile(
    r"(?:sampai(?:\s+dengan)?|until|valid\s+until|berlaku\s+(?:hingga|sampai))",
    re.I,
)
_FROM_LABEL_RE = re.compile(
    r"(?:berlaku\s+untuk|valid\s+from)",
    re.I,
)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _parse_text_dates(ocr_n: str) -> list[tuple[int, date]]:
    out: list[tuple[int, date]] = []
    for m in _TEXT_DATE_RE.finditer(ocr_n):
        day = int(m.group(1))
        month = _MONTHS.get(m.group(2).casefold())
        year = int(m.group(3))
        if not month:
            continue
        parsed = _safe_date(year, month, day)
        if parsed:
            out.append((m.start(), parsed))
    for m in _NUM_DATE_RE.finditer(ocr_n):
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        parsed = _safe_date(year, month, day)
        if parsed:
            out.append((m.start(), parsed))
    out.sort(key=lambda t: t[0])
    return out


def _first_date_after(label_end: int, dates: list[tuple[int, date]]) -> date | None:
    for pos, dt in dates:
        if pos >= label_end:
            return dt
    return None


def extract_skck_validity_dates(ocr_text: str) -> dict[str, Any]:
    """Ambil tanggal berlaku / kadaluarsa dari OCR SKCK."""
    ocr_n = " ".join((ocr_text or "").casefold().split())
    dates = [(p, d) for p, d in _parse_text_dates(ocr_n) if d.year >= 2020]
    valid_from: date | None = None
    valid_until: date | None = None

    um = _UNTIL_LABEL_RE.search(ocr_n)
    if um:
        valid_until = _first_date_after(um.end(), dates)
    fm = _FROM_LABEL_RE.search(ocr_n)
    if fm:
        valid_from = _first_date_after(fm.end(), dates)

    uniq = sorted({d for _, d in dates})
    if valid_until is None and uniq:
        valid_until = uniq[-1]
    if valid_from is None and len(uniq) >= 2:
        candidate = uniq[0]
        if valid_until is None or candidate < valid_until:
            valid_from = candidate

    return {
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_until": valid_until.isoformat() if valid_until else None,
        "valid_from_date": valid_from,
        "valid_until_date": valid_until,
        "dates_found": [d.isoformat() for d in uniq],
    }


def evaluate_skck_expiry_gate(
    ocr_text: str,
    *,
    profile_id: str,
    today: date | None = None,
) -> dict[str, Any] | None:
    """
    None jika bukan SKCK.
    Aktif/valid jika tanggal kadaluarsa (Sampai dengan) >= hari ini
    dan tanggal mulai (Berlaku untuk) tidak di masa depan.
    """
    if (profile_id or "").strip().casefold() != "skck":
        return None
    as_of = today or date.today()
    extracted = extract_skck_validity_dates(ocr_text)
    until = extracted.get("valid_until_date")
    start = extracted.get("valid_from_date")
    if until is None:
        expiry_pass = False
        reason = (
            "SKCK tidak lolos: tanggal kadaluarsa (Sampai dengan / Until) "
            "tidak terbaca di OCR."
        )
    elif until < as_of:
        expiry_pass = False
        reason = (
            "SKCK tidak lolos: sudah kadaluarsa "
            f"(Sampai dengan {until.isoformat()}, hari ini {as_of.isoformat()})."
        )
    elif start is not None and start > as_of:
        expiry_pass = False
        reason = (
            "SKCK tidak lolos: belum berlaku "
            f"(Berlaku untuk {start.isoformat()}, hari ini {as_of.isoformat()})."
        )
    else:
        expiry_pass = True
        reason = (
            "SKCK masih aktif: tanggal kadaluarsa "
            f"{until.isoformat()} belum terlewati (hari ini {as_of.isoformat()})."
        )
    return {
        "evaluated": True,
        "expiry_pass": expiry_pass,
        "today": as_of.isoformat(),
        "valid_from": extracted.get("valid_from"),
        "valid_until": extracted.get("valid_until"),
        "dates_found": extracted.get("dates_found"),
        "reason": reason,
    }


def apply_skck_expiry_gate(
    validation: dict[str, Any],
    *,
    ocr_text: str,
    profile_id: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    out = dict(validation)
    profile = (profile_id or out.get("document_profile_id") or "").strip().casefold()
    gate = evaluate_skck_expiry_gate(ocr_text, profile_id=profile, today=today)
    if gate is None:
        out.setdefault("skck_expiry_pass", None)
        return out

    out["skck_expiry"] = {
        k: gate[k]
        for k in ("today", "valid_from", "valid_until", "dates_found", "reason")
    }
    out["skck_expiry_pass"] = gate["expiry_pass"]
    out["skck_valid_until"] = gate["valid_until"]
    out["skck_valid_from"] = gate["valid_from"]
    out["issued_date"] = gate["valid_from"]
    out["issued_date_display"] = format_id_date(gate["valid_from"])

    explanation = dict(out.get("explanation") or {})
    details = [gate["reason"], *list(explanation.get("detail_lines") or [])]
    explanation["detail_lines"] = details
    gates = dict(explanation.get("gates") or {})
    gates["skck_expiry"] = {
        "pass": gate["expiry_pass"],
        "valid_until": gate["valid_until"],
        "valid_from": gate["valid_from"],
        "today": gate["today"],
    }
    explanation["gates"] = gates

    if not gate["expiry_pass"]:
        out["document_matched"] = False
        explanation["summary"] = gate["reason"]
        blockers = list(explanation.get("primary_blockers") or [])
        if "SKCK_EXPIRY" not in blockers:
            blockers.insert(0, "SKCK_EXPIRY")
        explanation["primary_blockers"] = blockers
        verdict = dict(out.get("verdict") or {})
        verdict["document_matched"] = False
        verdict["summary"] = gate["reason"]
        out["verdict"] = verdict

    out["explanation"] = explanation
    return out
