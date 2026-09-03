"""Deteksi bank pada bukti rekening tabungan (Mandiri vs Bank MAS)."""

from __future__ import annotations

import re
from typing import Any, Final

from rapidfuzz import fuzz

SUPPORTED_BANKS: Final[frozenset[str]] = frozenset({"mandiri", "mas"})

BANK_LABELS: Final[dict[str, str]] = {
    "mandiri": "Bank Mandiri",
    "mas": "Bank MAS",
}

BANK_ALIASES: Final[dict[str, str]] = {
    "mandiri": "mandiri",
    "bank mandiri": "mandiri",
    "livin": "mandiri",
    "livin by mandiri": "mandiri",
    "mas": "mas",
    "bank mas": "mas",
    "bank mas saving": "mas",
    "multi arta sentosa": "mas",
}

# Awalan no. rekening umum di dataset internal (13 digit Mandiri, 10 digit MAS).
MANDIRI_ACCOUNT_PREFIXES: Final[frozenset[str]] = frozenset(
    {
        "156",
        "173",
        "166",
        "167",
        "125",
        "134",
        "180",
        "187",
        "120",
        "121",
        "126",
        "182",
    }
)

MANDIRI_PHRASES: Final[list[str]] = [
    "livin by mandiri",
    "livin",
    "bank mandiri",
    "detail rekening",
    "sebagai rekening utama",
]

MAS_PHRASES: Final[list[str]] = [
    "bank mas",
    "mas saving",
    "bebaspoin",
    "bebas poin",
    "multi arta sentosa",
]

_BANKING_CONTEXT = frozenset(
    {
        "rekening",
        "tabungan",
        "livin",
        "detail",
        "saving",
        "saldo",
        "transfer",
        "rekeningku",
    }
)

_MANDIRI_FALSE_POSITIVE = (
    "peserta mandiri",
    "binakarya mandiri",
    "kemandirian",
)


def list_supported_banks() -> list[str]:
    return sorted(SUPPORTED_BANKS)


def bank_label(bank_id: str) -> str:
    bid = (bank_id or "").strip().casefold()
    return BANK_LABELS.get(bid, bank_id)


def normalize_expected_bank(raw: str) -> str | None:
    s = " ".join((raw or "").strip().casefold().split())
    if not s:
        return None
    if s in BANK_ALIASES:
        return BANK_ALIASES[s]
    if s in SUPPORTED_BANKS:
        return s
    return None


def _normalize_ocr(s: str) -> str:
    return " ".join((s or "").casefold().split())


def _phrase_score(ocr_n: str, phrase: str) -> float:
    if phrase in ocr_n:
        return 100.0
    return float(fuzz.partial_ratio(phrase, ocr_n))


def _extract_account_numbers(ocr_text: str) -> list[str]:
    candidates: list[str] = []
    for m in re.finditer(r"\d[\d\s.\-]{8,24}\d", ocr_text):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:
            candidates.append(digits)
    compact = re.sub(r"\D", "", ocr_text)
    for length in (13, 10):
        for i in range(0, max(0, len(compact) - length + 1)):
            chunk = compact[i : i + length]
            if chunk.isdigit():
                candidates.append(chunk)
    seen: set[str] = set()
    out: list[str] = []
    for n in candidates:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def score_bank_signals(ocr_text: str) -> dict[str, Any]:
    ocr_n = _normalize_ocr(ocr_text)
    mandiri_score = 0.0
    mas_score = 0.0
    signals: dict[str, list[str]] = {"mandiri": [], "mas": []}

    for phrase in MANDIRI_PHRASES:
        sc = _phrase_score(ocr_n, phrase)
        if sc >= 82.0:
            mandiri_score = max(mandiri_score, sc)
            signals["mandiri"].append(f"phrase:{phrase}")

    if not any(fp in ocr_n for fp in _MANDIRI_FALSE_POSITIVE):
        if re.search(r"\bmandiri\b", ocr_n) and any(ctx in ocr_n for ctx in _BANKING_CONTEXT):
            mandiri_score = max(mandiri_score, 88.0)
            signals["mandiri"].append("keyword:mandiri+banking_context")

    for phrase in MAS_PHRASES:
        sc = _phrase_score(ocr_n, phrase)
        if sc >= 82.0:
            mas_score = max(mas_score, sc)
            signals["mas"].append(f"phrase:{phrase}")

    if re.search(r"\bmas saving\b", ocr_n) or re.search(r"\bbank mas\b", ocr_n):
        mas_score = max(mas_score, 92.0)

    for acct in _extract_account_numbers(ocr_text):
        if len(acct) == 13 and acct[:3] in MANDIRI_ACCOUNT_PREFIXES:
            mandiri_score = max(mandiri_score, 96.0)
            signals["mandiri"].append(f"account:{acct}")
        elif len(acct) == 10 and acct.startswith(("100", "120")):
            mas_score = max(mas_score, 96.0)
            signals["mas"].append(f"account:{acct}")

    return {
        "mandiri": round(mandiri_score, 2),
        "mas": round(mas_score, 2),
        "signals": signals,
    }


def detect_bank_from_ocr(ocr_text: str, *, min_score: float = 70.0) -> str | None:
    scores = score_bank_signals(ocr_text)
    m_sc = float(scores["mandiri"])
    mas_sc = float(scores["mas"])
    if m_sc < min_score and mas_sc < min_score:
        return None
    if m_sc >= min_score and mas_sc >= min_score:
        if abs(m_sc - mas_sc) < 8.0:
            return None
    if m_sc > mas_sc:
        return "mandiri"
    if mas_sc > m_sc:
        return "mas"
    return None


def validate_rekening_bank(
    ocr_text: str,
    *,
    expected_bank: str | None = None,
    min_score: float = 70.0,
) -> dict[str, Any]:
    """Deteksi bank dari OCR; validasi terhadap expected_bank bila diisi."""
    exp = normalize_expected_bank(expected_bank or "")
    scores = score_bank_signals(ocr_text)
    detected = detect_bank_from_ocr(ocr_text, min_score=min_score)

    out: dict[str, Any] = {
        "expected_bank": exp,
        "expected_bank_label": bank_label(exp) if exp else None,
        "detected_bank": detected,
        "detected_bank_label": bank_label(detected) if detected else None,
        "scores": {"mandiri": scores["mandiri"], "mas": scores["mas"]},
        "signals": scores["signals"],
        "min_score": min_score,
        "bank_pass": None,
    }

    if not exp:
        return out

    exp_score = float(scores["mandiri"] if exp == "mandiri" else scores["mas"])

    if detected == exp and exp_score >= min_score:
        out["bank_pass"] = True
    elif detected and detected != exp:
        out["bank_pass"] = False
        out["mismatch"] = {
            "expected": exp,
            "detected": detected,
        }
    elif exp_score >= min_score:
        out["bank_pass"] = True
        out["note"] = "Bank terdeteksi lewat sinyal OCR tanpa pemenang eksplisit."
    else:
        out["bank_pass"] = False
        out["note"] = f"Sinyal {bank_label(exp)} di OCR di bawah ambang {min_score:.0f}."

    return out
