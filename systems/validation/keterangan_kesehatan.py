"""Gate surat keterangan kesehatan: kesimpulan SEHAT + tanggal cek belum kadaluarsa."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Any

from systems.validation.skck_validity import (
    _FROM_LABEL_RE,
    _parse_text_dates,
    format_id_date,
)

# Tanggal kadaluarsa hanya dihitung bila surat menulis durasi/tanggal sampai.
# Jangan mengarang masa berlaku (mis. +90 hari) jika tidak ada di dokumen.

_NEGATIVE_RE = re.compile(
    r"tidak\s+sehat|kurang\s+sehat|dinyatakan\s+sakit|\bunfit\b|tidak\s+fit",
    re.I,
)
_POSITIVE_RE = re.compile(
    r"dinyatakan\s+sehat|dalam\s+keadaan\s+sehat|keadaan\s+sehat|"
    r"sehat\s+jasmani|sehat\s+rohani|fit\s+to\s+work|fit\s+for\s+work|"
    r"surat\s+keterangan\s+sehat|surat\s+keterangan\s+dokter|"
    r"kesehatan\s+badan|\bsehat\b",
    re.I,
)
# Formulir MCU: opsi "Sehat / Tidak Sehat" — bukan kesimpulan sakit.
_TEMPLATE_CHOICE_RE = re.compile(
    r"seha[t]?\s*/\s*tidak\s*sehat|tidak\s+sehat\s*/\s*seha[t]?",
    re.I,
)
_NUM_DATE_FLEX_RE = re.compile(
    r"\b(\d{1,2})[/\-](\d{1,2})[/\-. ]+(20\d{2}|\d{2}|2\d{2})\b"
)
_OCR_JUNK_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s*[\(/]\s*([a-z0-9])\s*/?\s*20[2c][0-9c6]",
    re.I,
)
_DURATION_RE = re.compile(
    r"(?:berlaku|masa\s+berlaku)[^\d]{0,52}(?:maksimal\s+)?(\d+)\s*"
    r"(?:\([^)]{0,24}\)\s*)?(hari|bulan|minggu)",
    re.I,
)
_LAHIR_NEAR_RE = re.compile(r"lahir|\bttl\b|tempat\s+dan\s+tgl", re.I)
_VITAL_NEAR_RE = re.compile(
    r"visus|\btd\b|tensi|mmhg|nadi|sistol|diastole|\bbb\b|\btb\b|berat\s+badan|"
    r"tinggi\s+badan|spo2",
    re.I,
)
_UNTIL_EXPLICIT_RE = re.compile(
    r"(?:sampai\s+dengan|yaitu\s+sampai|until|valid\s+until|"
    r"berlaku\s+(?:hingga|sampai)(?!\s+selama))",
    re.I,
)
_SIGN_AFTER_RE = re.compile(
    r"demikian\s+surat|dokter(?:\s+(?:pemeriksa|yang\s+memeriksa))?|"
    r"yang\s+memeriksa|tgl\.?\s*cetak|dikeluarkan|terverifikasi",
    re.I,
)
_CITIES_RE = (
    r"(?:kota\s+|kabupaten[\s\-]?)?(?:bekasi|cikarang|jakarta|depok|karawang|"
    r"tangerang|bogor|bandung|tasikmalaya|darangdan|cilamaya|medansatria|"
    r"bekas[li]|pejuang|tegal|tambun|cibitung|surabaya|serang|brebes|"
    r"indramayu|senen|pegagan|cidempet|cikeusal|jababeka|semarang)"
)
_MONTH_OCR: dict[str, int] = {
    "januari": 1, "january": 1, "jan": 1,
    "februari": 2, "february": 2, "feb": 2,
    "maret": 3, "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5, "may": 5, "mel": 5, "wei": 5,
    "juni": 6, "june": 6, "jun": 6, "juri": 6, "funi": 6, "junl": 6,
    "jumi": 6, "suni": 6, "jomi": 6, "juai": 6, "jusi": 6,
    "juli": 7, "july": 7, "jul": 7,
    "agustus": 8, "august": 8, "agu": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "oktober": 10, "october": 10, "okt": 10, "oct": 10,
    "november": 11, "nopember": 11, "nov": 11,
    "desember": 12, "december": 12, "des": 12, "dec": 12, "besember": 12,
}
_MONTH_ALT_OCR = "|".join(sorted(_MONTH_OCR, key=len, reverse=True))
_PLACE_TEXT_DATE_RE = re.compile(
    rf"(?:{_CITIES_RE}|[a-z]{{3,20}})\s*[,.\-]?\s*(\d{{1,2}})\s+({_MONTH_ALT_OCR})\s*[.\-]*\s*(20\d{{2}}|20\d|\d{{2}})",
    re.I,
)
_GLUED_PLACE_DATE_RE = re.compile(
    rf"(?:{_CITIES_RE}|[a-z]{{3,20}})[\s\-]*(\d{{1,2}})({_MONTH_ALT_OCR})\s+(20\d{{2}}|\d{{2}})",
    re.I,
)
_PLACE_NUM_DATE_RE = re.compile(
    rf"(?:{_CITIES_RE})\s*[,.\-]*\s*(\d{{1,2}})[/\-](\d{{1,2}})(?:[/\-.](\d{{2,4}}))?",
    re.I,
)
_DATE_BEFORE_PLACE_RE = re.compile(
    r"\b(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?\s+[a-z]{3,20}\s*[,.]?\s*(20\d{2}|20[\s.]*[2-3]\d)?",
    re.I,
)
_GLUED_TEXT_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})[\s.\-]*({_MONTH_ALT_OCR})[\s.\-]*?(20\d{{2}}|\d{{2}})\b",
    re.I,
)
_EN_HYPHEN_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})[-/]({_MONTH_ALT_OCR})[-/](20\d{{2}}|\d{{2}})\b",
    re.I,
)
_COMPACT_YMD_RE = re.compile(
    r"(?<!\d)(20[2-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"
)
_SLASH_DM_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})(?![/\d])")
_TRAILING_YEAR_RE = re.compile(r"20[\s.]*([2-3]\d)|2[a-z0](6|7|8|9)", re.I)
_OCR_YEAR_DATE_RE = re.compile(
    r"\b(\d{1,2})[/\-](\d{1,2})[/\-]2[a-z0](\d)\b",
    re.I,
)
_SPACED_NUM_DATE_RE = re.compile(
    r"\b(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(20\d{2})\b"
)
_MONTH_SPLIT_FIX = (
    "januari|februari|maret|april|mei|juni|juli|agustus|september|"
    "oktober|november|desember|june|july"
)
_VISUS_PAIRS = {
    (4, 6), (4, 60), (5, 5), (5, 6), (5, 60),
    (6, 5), (6, 6), (6, 9), (6, 12), (6, 18), (6, 24), (6, 36), (6, 60),
}
_SIP_NEAR_RE = re.compile(
    r"\bsip\b|dpm[-\s]?ptsp|/du/\d{4}|izin\s+nomor|nip\s*:",
    re.I,
)
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})[\s.:\-]*({_MONTH_ALT_OCR})\b",
    re.I,
)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALT_OCR})\s+(\d{{1,2}})(?!\d)[\s.,]+(20\d{{2}}|\d{{2}})\b",
    re.I,
)
_COMPACT_DMY_RE = re.compile(
    r"(?<!\d)(\d{2})(\d{2})(20[2-3]\d)(?!\d)"
)
_LOOSE_SLASH_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s*[/\-]\s*(\d{1,2})\s+(20\d{2}|\d{2})\b"
)
_SPACED_DMY_PARTS_RE = re.compile(
    r"\b(\d{1,2})\s+(\d{1,2})\s+20(?:\s*(\d{2}))?\b"
)
_GAP_DMY_RE = re.compile(
    r"\b(\d{1,2})\s+(1[0-2]|0?[1-9]|16)\b.{0,96}?\b20(?:\s*(\d{2}))?\b",
    re.I,
)
_ROMAN_MONTH_RE = re.compile(
    r"\b(\d{1,2})\s+(viii|vii|xii|xi|ix|vi|iv)\b",
    re.I,
)
_ROMAN_MONTH = {
    "iv": 4, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "xi": 11, "xii": 12,
}
_GLUED_DMY_FLEX_RE = re.compile(
    r"(?<!\d)(\d{2})[01]?(\d{2})(20[2-3]\d)(?!\d)"
)


def extract_health_conclusion(ocr_text: str) -> dict[str, Any]:
    blob = ocr_text or ""
    template = bool(_TEMPLATE_CHOICE_RE.search(blob))
    cleaned = _TEMPLATE_CHOICE_RE.sub(" ", blob)
    negative = bool(_NEGATIVE_RE.search(cleaned))
    positive = bool(_POSITIVE_RE.search(cleaned)) or template
    # Lembar yang mencetak kedua opsi tanpa kesimpulan "tidak sehat" terpisah = lolos.
    healthy = (positive and not negative) or (template and not negative)
    return {
        "healthy": healthy,
        "positive_signal": positive,
        "negative_signal": negative,
        "template_choice": template,
    }


def _add_months(d: date, months: int) -> date:
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def _duration_match(ocr_text: str) -> re.Match[str] | None:
    blob = _normalize_health_ocr(ocr_text)
    return _DURATION_RE.search(blob) or _DURATION_RE.search(ocr_text or "")


def _validity_delta(ocr_text: str, exam: date) -> date | None:
    m = _duration_match(ocr_text)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).casefold()
    if unit.startswith("hari"):
        return exam + timedelta(days=max(1, n))
    if unit.startswith("minggu"):
        return exam + timedelta(days=max(1, n * 7))
    return _add_months(exam, max(1, n))


def _validity_days(ocr_text: str) -> int | None:
    m = _duration_match(ocr_text)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).casefold()
    if unit.startswith("hari"):
        return max(1, n)
    if unit.startswith("minggu"):
        return max(1, n * 7)
    return max(1, n * 30)


def _normalize_health_ocr(ocr_text: str) -> str:
    t = (ocr_text or "").casefold()
    t = t.replace("we!", "mei")
    t = re.sub(r"\b(\d{2})(\d{2})\s+20\s+(\d{2})\b", r"\1/\2/20\3", t)
    t = re.sub(r"(?<!\d)20[.\s]{2,}(\d)\b", r"202\1", t)
    t = re.sub(r"\b20[\s.]+([2-3]\d)\b", r"20\1", t)
    t = re.sub(r"\b20[.\s]*([2-3])[.\s]+(\d)\b", r"20\1\2", t)
    t = re.sub(r"\.{2,}(20\d{0,2})\b", r" \1", t)
    t = re.sub(r"\b20\s+26\b", "2026", t)
    t = re.sub(r"\b(\d{1,2})\s+106\s+\d?\s*(20[2-3]\d)\b", r"\1 06 \2", t)
    t = re.sub(r"\b2[oa]([6-9])\b", r"202\1", t)
    t = re.sub(r"\b1202\b", "2026", t)
    t = re.sub(r"(\d{1,2}\s*[/\-]\s*\d{1,2}\s+)1(20[2-3]\d)\b", r"\1\2", t)
    t = re.sub(r"(\d{1,2})[/\-]0/+?(20[2-3]\d)", r"\1/06/\2", t)
    t = re.sub(r"\bo(\d)\b", r"0\1", t)
    t = re.sub(r"\b(\d{1,2})\s*[\-./]\s*(\d{1,2})\s*[\-./]\s*20[69]\b", r"\1/\2/2026", t)
    # Jam cetak "11 Juni 2026, 20.01" / "15/06/26, 13.13" bukan tanggal.
    t = re.sub(r"(?:,\s*)?(?:[01]?\d|2[0-3])[.:][0-5]\d\b", " ", t)
    for bad, good in (
        ("besember", "desember"),
        ("funi", "juni"),
        ("suni", "juni"),
        ("jomi", "juni"),
        ("juai", "juni"),
        ("juri", "juni"),
        ("jumi", "juni"),
        ("junl", "juni"),
        ("jusi", "juni"),
    ):
        t = t.replace(bad, good)
    t = re.sub(rf"(?<=\d)({_MONTH_SPLIT_FIX})", r" \1", t, flags=re.I)
    t = re.sub(rf"({_MONTH_SPLIT_FIX})(?=\d)", r"\1 ", t, flags=re.I)
    t = re.sub(r"\bmel\b", "mei", t)
    t = re.sub(r"\bmer\b", "mei", t)
    t = re.sub(r"\bwei\b", "mei", t)
    return " ".join(t.split())


def _coerce_exam_year(raw: str, ocr_n: str) -> int | None:
    token = raw.strip().casefold().replace("o", "0").replace("c", "6")
    digits = re.sub(r"\D", "", token)
    if not digits:
        return None
    y = int(digits)
    if 2024 <= y <= 2035:
        return y
    if 24 <= y <= 29:
        return 2000 + y
    if 226 <= y <= 229:
        return 2000 + (y - 200)
    if y in {202, 206, 209}:
        return 2026
    if y == 20:
        found = [int(x) for x in re.findall(r"20[2-3]\d", ocr_n)]
        recent = [x for x in found if x >= 2025]
        if recent:
            return max(recent)
        return 2026
    return None


def _safe_health_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _append_dated(
    dated: list[tuple[int, date]],
    seen: set[tuple[int, str]],
    pos: int,
    parsed: date | None,
) -> None:
    if parsed is None:
        return
    key = (pos, parsed.isoformat())
    if key in seen:
        return
    seen.add(key)
    dated.append((pos, parsed))


def _near_noise(ocr_n: str, pos: int, end: int | None = None) -> bool:
    a = max(0, pos - 28)
    b = min(len(ocr_n), (end if end is not None else pos) + 36)
    chunk = ocr_n[a:b]
    named = _has_month_name(ocr_n, pos)
    has_year = bool(re.search(r"20[2-3]\d", ocr_n[pos : min(len(ocr_n), pos + 28)]))
    if _SIP_NEAR_RE.search(chunk) and not named:
        return True
    if re.search(r"\b(?:visus|ods?)\b", chunk) and not named and not has_year:
        return True
    return False


def _doc_year(ocr_n: str) -> int | None:
    years = [int(y) for y in re.findall(r"20[2-3]\d", ocr_n)]
    recent = [y for y in years if 2025 <= y <= 2027]
    if recent:
        return max(recent)
    return None


def _year_near(ocr_n: str, pos: int, end: int | None = None) -> int | None:
    start = max(0, pos - 12)
    stop = min(len(ocr_n), (end if end is not None else pos) + 64)
    chunk = ocr_n[start:stop]
    m = _TRAILING_YEAR_RE.search(chunk)
    if m:
        if m.group(1):
            return _coerce_exam_year("20" + m.group(1), ocr_n)
        return _coerce_exam_year("202" + m.group(2), ocr_n)
    found = re.findall(r"20[2-3]\d|20[269]", chunk)
    if found:
        return _coerce_exam_year(found[-1], ocr_n)
    if pos >= int(len(ocr_n) * 0.42):
        return _doc_year(ocr_n)
    return None


def _dates_from_city_windows(
    ocr_n: str,
    dated: list[tuple[int, date]],
    seen: set[tuple[int, str]],
) -> None:
    city_re = re.compile(_CITIES_RE, re.I)
    month_re = re.compile(rf"\b({_MONTH_ALT_OCR})\b", re.I)
    for cm in city_re.finditer(ocr_n):
        a = max(0, cm.start() - 56)
        b = min(len(ocr_n), cm.end() + 80)
        win = ocr_n[a:b]
        mmatch = month_re.search(win)
        if not mmatch:
            continue
        month = _MONTH_OCR.get(mmatch.group(1).casefold())
        if not month:
            continue
        year = None
        for tok in re.findall(r"20[2-3]\d|20[269]", win):
            year = _coerce_exam_year(tok, ocr_n) or year
        if year is None:
            year = _year_near(ocr_n, cm.start(), cm.end())
        if year is None:
            continue
        after = ocr_n[cm.end() : cm.end() + 16]
        dm = re.search(r"^\s*[,.\-]?\s*(\d{1,2})\b", after)
        day: int | None = None
        if dm and 1 <= int(dm.group(1)) <= 31:
            day = int(dm.group(1))
        else:
            before = ocr_n[max(0, cm.start() - 16) : cm.start()]
            dm2 = re.search(r"\b(\d{1,2})\s*$", before)
            if dm2 and 1 <= int(dm2.group(1)) <= 31:
                day = int(dm2.group(1))
        if day is None:
            continue
        _append_dated(
            dated, seen, cm.start(),
            _safe_health_date(year, month, day),
        )


def _parse_health_dates(ocr_n: str) -> list[tuple[int, date]]:
    dated = list(_parse_text_dates(ocr_n))
    seen = {(pos, dt.isoformat()) for pos, dt in dated}

    for m in _NUM_DATE_FLEX_RE.finditer(ocr_n):
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        month = int(m.group(2))
        if month == 16:
            month = 6
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _SPACED_NUM_DATE_RE.finditer(ocr_n):
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, int(m.group(2)), int(m.group(1))),
        )
    for m in _OCR_YEAR_DATE_RE.finditer(ocr_n):
        year = _coerce_exam_year("202" + m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, int(m.group(2)), int(m.group(1))),
        )
    for m in _OCR_JUNK_YEAR_RE.finditer(ocr_n):
        month_tok = m.group(2).casefold().replace("o", "0").replace("c", "6")
        if not month_tok.isdigit():
            continue
        year = _year_near(ocr_n, m.start(), m.end()) or 2026
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, int(month_tok), int(m.group(1))),
        )
    month_re = re.compile(
        rf"\b(\d{{1,2}})\s+({_MONTH_ALT_OCR})\s+(20\d{{2}}|2[a-z0]\d|\d{{2}})\b",
        re.I,
    )
    for m in month_re.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(2).casefold())
        if not month:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _PLACE_TEXT_DATE_RE.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(2).casefold())
        if not month:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(1),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _GLUED_PLACE_DATE_RE.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(2).casefold())
        if not month:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(1),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _PLACE_NUM_DATE_RE.finditer(ocr_n):
        day = int(m.group(1))
        month = int(m.group(2))
        year_raw = m.group(3)
        year = _coerce_exam_year(year_raw, ocr_n) if year_raw else _year_near(
            ocr_n, m.start(), m.end()
        )
        if month > 12:
            year = _coerce_exam_year(str(month), ocr_n) or year
            if 1 <= day <= 12:
                month = day
        if year is None or month < 1 or month > 12:
            continue
        _append_dated(
            dated, seen, m.start(1),
            _safe_health_date(year, month, day),
        )
    for m in _DATE_BEFORE_PLACE_RE.finditer(ocr_n):
        month = int(m.group(2))
        if month < 1 or month > 12:
            continue
        year_raw = m.group(3) or m.group(4)
        year = (
            _coerce_exam_year(year_raw, ocr_n)
            if year_raw
            else _year_near(ocr_n, m.start(), m.end())
        )
        if year is None:
            continue
        day = int(m.group(1))
        if (day, month) in _VISUS_PAIRS:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _GLUED_TEXT_DATE_RE.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(2).casefold())
        if not month:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _EN_HYPHEN_DATE_RE.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(2).casefold())
        if not month:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _COMPACT_YMD_RE.finditer(ocr_n):
        if _near_noise(ocr_n, m.start(), m.end()):
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(int(m.group(1)), int(m.group(2)), int(m.group(3))),
        )
    for m in _COMPACT_DMY_RE.finditer(ocr_n):
        if _near_noise(ocr_n, m.start(), m.end()):
            continue
        month = int(m.group(2))
        if month == 16:
            month = 6
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(int(m.group(3)), month, int(m.group(1))),
        )
    for m in _MONTH_DAY_YEAR_RE.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(1).casefold())
        if not month:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(2),
            _safe_health_date(year, month, int(m.group(2))),
        )
    for m in _DAY_MONTH_RE.finditer(ocr_n):
        month = _MONTH_OCR.get(m.group(2).casefold())
        if not month:
            continue
        year = _year_near(ocr_n, m.start(), m.end())
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    for m in _SLASH_DM_RE.finditer(ocr_n):
        month = int(m.group(2))
        day = int(m.group(1))
        if month < 1 or month > 12 or day < 1 or day > 31:
            continue
        if (day, month) in _VISUS_PAIRS:
            continue
        around = ocr_n[max(0, m.start() - 24) : m.end() + 8]
        if re.search(r"\b(?:rt|rw)\b", around):
            continue
        if _near_noise(ocr_n, m.start(), m.end()) or _VITAL_NEAR_RE.search(around):
            continue
        year = _year_near(ocr_n, m.start(), m.end())
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, day),
        )
    for m in _LOOSE_SLASH_YEAR_RE.finditer(ocr_n):
        month = int(m.group(2))
        day = int(m.group(1))
        if month == 16:
            month = 6
        if month < 1 or month > 12 or day < 1 or day > 31:
            continue
        if (day, month) in _VISUS_PAIRS:
            continue
        year = _coerce_exam_year(m.group(3), ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, day),
        )
    for m in _SPACED_DMY_PARTS_RE.finditer(ocr_n):
        month = int(m.group(2))
        day = int(m.group(1))
        if month == 16:
            month = 6
        if month < 1 or month > 12 or day < 1 or day > 31:
            continue
        if (day, month) in _VISUS_PAIRS:
            continue
        year_tok = ("20" + m.group(3)) if m.group(3) else "20"
        year = _coerce_exam_year(year_tok, ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, day),
        )
    for m in _GAP_DMY_RE.finditer(ocr_n):
        month = int(m.group(2))
        day = int(m.group(1))
        if month == 16:
            month = 6
        if month < 1 or month > 12 or day < 1 or day > 31:
            continue
        if (day, month) in _VISUS_PAIRS:
            continue
        if _near_noise(ocr_n, m.start(), m.start() + 8):
            continue
        year_tok = ("20" + m.group(3)) if m.group(3) else "20"
        year = _coerce_exam_year(year_tok, ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, day),
        )
    for m in _ROMAN_MONTH_RE.finditer(ocr_n):
        month = _ROMAN_MONTH.get(m.group(2).casefold())
        if not month:
            continue
        year = _year_near(ocr_n, m.start(), m.end()) or _doc_year(ocr_n)
        if year is None:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(year, month, int(m.group(1))),
        )
    _dates_from_city_windows(ocr_n, dated, seen)
    for m in _GLUED_DMY_FLEX_RE.finditer(ocr_n):
        if _near_noise(ocr_n, m.start(), m.end()):
            continue
        month = int(m.group(2))
        if month == 16:
            month = 6
        if month < 1 or month > 12:
            continue
        _append_dated(
            dated, seen, m.start(),
            _safe_health_date(int(m.group(3)), month, int(m.group(1))),
        )
    dated.sort(key=lambda t: t[0])
    return dated


def _until_label_spans(ocr_n: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _UNTIL_EXPLICIT_RE.finditer(ocr_n)]


def _date_after_spans(
    dated: list[tuple[int, date]],
    ocr_n: str,
    spans: list[tuple[int, int]],
    *,
    window: int = 90,
) -> date | None:
    for _, end in sorted(spans, key=lambda t: t[0]):
        chunk = ocr_n[end : end + window]
        if re.search(_CITIES_RE, chunk[:36], re.I) and not re.search(
            r"tanggal|tgl\.?", chunk[:28], re.I
        ):
            continue
        for pos, dt in dated:
            if end <= pos <= end + window and 2024 <= dt.year <= 2029:
                return dt
    return None


def _until_date_positions(
    dated: list[tuple[int, date]], spans: list[tuple[int, int]], *, window: int = 90
) -> set[int]:
    """Hanya tanggal pertama setelah label 'sampai dengan', bukan seluruh kaki surat."""
    used: set[int] = set()
    for _, end in sorted(spans, key=lambda t: t[0]):
        for pos, dt in dated:
            if end <= pos <= end + window and 2024 <= dt.year <= 2029:
                used.add(pos)
                break
    return used


def _has_month_name(ocr_n: str, pos: int) -> bool:
    chunk = ocr_n[max(0, pos - 4) : min(len(ocr_n), pos + 28)]
    return bool(re.search(rf"\b(?:{_MONTH_ALT_OCR})\b", chunk, re.I))


def _looks_like_signature(ocr_n: str, pos: int) -> bool:
    prev = ocr_n[max(0, pos - 96) : pos]
    if _SIGN_AFTER_RE.search(prev):
        return True
    if re.search(_CITIES_RE, prev, re.I):
        return True
    if _has_month_name(ocr_n, pos) and pos >= int(len(ocr_n) * 0.35):
        return True
    return pos >= int(len(ocr_n) * 0.48)


def extract_health_check_dates(ocr_text: str) -> dict[str, Any]:
    ocr_n = _normalize_health_ocr(ocr_text)
    dated = _parse_health_dates(ocr_n)
    until_spans = _until_label_spans(ocr_n)
    until = _date_after_spans(dated, ocr_n, until_spans)
    until_pos = _until_date_positions(dated, until_spans)

    exam_hits: list[tuple[int, date]] = []
    sign_hits: list[tuple[int, date]] = []
    text_hits: list[tuple[int, date]] = []
    for pos, dt in dated:
        if dt.year < 2024 or dt.year > 2029:
            continue
        window = ocr_n[max(0, pos - 48) : pos]
        if _LAHIR_NEAR_RE.search(window):
            continue
        if _near_noise(ocr_n, pos):
            continue
        if pos in until_pos:
            continue
        exam_hits.append((pos, dt))
        named = _has_month_name(ocr_n, pos)
        if named:
            text_hits.append((pos, dt))
        if _looks_like_signature(ocr_n, pos):
            sign_hits.append((pos, dt))

    exam: date | None = None
    text_sign = [(p, d) for p, d in sign_hits if _has_month_name(ocr_n, p)]
    if text_sign:
        exam = text_sign[-1][1]
    elif text_hits:
        exam = text_hits[-1][1]
    elif sign_hits:
        exam = sign_hits[-1][1]
    elif exam_hits:
        exam = exam_hits[-1][1]
    if exam is None:
        named = [
            (p, d)
            for p, d in dated
            if 2024 <= d.year <= 2029 and _has_month_name(ocr_n, p)
        ]
        if named:
            exam = named[-1][1]
        else:
            recent = [(p, d) for p, d in dated if 2024 <= d.year <= 2029]
            if recent:
                exam = recent[-1][1]

    if exam is not None and until is not None and exam == until and exam_hits:
        earlier = [dt for _, dt in exam_hits if dt < until]
        if earlier:
            exam = earlier[-1]

    start: date | None = None
    fm = _FROM_LABEL_RE.search(ocr_n)
    if fm:
        for pos, dt in dated:
            if pos >= fm.end() and dt.year >= 2024:
                start = dt
                break

    if until is None and exam is not None:
        until = _validity_delta(ocr_text, exam)
    dur = _duration_match(ocr_text)
    if exam is None and until is not None and dur:
        n = int(dur.group(1))
        unit = dur.group(2).casefold()
        if unit.startswith("bulan"):
            exam = _add_months(until, -max(1, n))
        elif unit.startswith("minggu"):
            exam = until - timedelta(days=max(1, n * 7))
        else:
            exam = until - timedelta(days=max(1, n))
    if start is None:
        start = exam

    days = _validity_days(ocr_text)
    if days is None and exam is not None and until is not None:
        days = max(0, (until - exam).days)

    uniq = sorted({dt for _, dt in dated if 2024 <= dt.year <= 2029})
    return {
        "exam_date": exam.isoformat() if exam else None,
        "exam_date_obj": exam,
        "valid_from": start.isoformat() if start else None,
        "valid_until": until.isoformat() if until else None,
        "valid_from_date": start,
        "valid_until_date": until,
        "validity_days": days,
        "dates_found": [d.isoformat() for d in uniq],
    }


def evaluate_keterangan_kesehatan_gates(
    ocr_text: str,
    *,
    profile_id: str,
    today: date | None = None,
) -> dict[str, Any] | None:
    if (profile_id or "").strip().casefold() != "keterangan_kesehatan":
        return None
    as_of = today or date.today()
    conclusion = extract_health_conclusion(ocr_text)
    dates = extract_health_check_dates(ocr_text)
    until = dates.get("valid_until_date")
    exam = dates.get("exam_date_obj")

    if not conclusion["healthy"]:
        if conclusion["negative_signal"]:
            health_reason = (
                "Surat keterangan kesehatan tidak lolos: tertera kondisi tidak sehat."
            )
        else:
            health_reason = (
                "Surat keterangan kesehatan tidak lolos: tidak tertera bahwa "
                "yang bersangkutan dinyatakan SEHAT."
            )
    else:
        health_reason = "Surat keterangan kesehatan terbaca: dinyatakan SEHAT."

    if exam is None and until is None:
        expiry_pass = False
        expiry_reason = (
            "Surat keterangan kesehatan tidak lolos: tanggal cek kesehatan "
            "tidak terbaca di OCR."
        )
    elif exam is not None and exam > as_of:
        expiry_pass = False
        expiry_reason = (
            "Surat keterangan kesehatan tidak lolos: tanggal cek di masa depan "
            f"({exam.isoformat()}, hari ini {as_of.isoformat()})."
        )
    elif until is not None and until < as_of:
        expiry_pass = False
        expiry_reason = (
            "Surat keterangan kesehatan tidak lolos: sudah kadaluarsa "
            f"(berlaku sampai {until.isoformat()}, hari ini {as_of.isoformat()})."
        )
    elif until is not None:
        expiry_pass = True
        expiry_reason = (
            "Tanggal cek kesehatan masih berlaku sampai "
            f"{until.isoformat()} (hari ini {as_of.isoformat()})."
        )
    else:
        expiry_pass = True
        expiry_reason = (
            f"Tanggal surat {exam.isoformat()} (dokumen tidak menulis "
            f"tanggal kadaluarsa, hari ini {as_of.isoformat()})."
        )

    return {
        "evaluated": True,
        "health_status_pass": conclusion["healthy"],
        "health_expiry_pass": expiry_pass,
        "today": as_of.isoformat(),
        "exam_date": dates.get("exam_date"),
        "valid_from": dates.get("valid_from"),
        "valid_until": dates.get("valid_until"),
        "validity_days": dates.get("validity_days"),
        "dates_found": dates.get("dates_found"),
        "health_reason": health_reason,
        "expiry_reason": expiry_reason,
        "positive_signal": conclusion["positive_signal"],
        "negative_signal": conclusion["negative_signal"],
    }


def apply_keterangan_kesehatan_gates(
    validation: dict[str, Any],
    *,
    ocr_text: str,
    profile_id: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    out = dict(validation)
    profile = (profile_id or out.get("document_profile_id") or "").strip().casefold()
    gate = evaluate_keterangan_kesehatan_gates(
        ocr_text, profile_id=profile, today=today
    )
    if gate is None:
        out.setdefault("health_status_pass", None)
        out.setdefault("health_expiry_pass", None)
        return out

    out["health_status"] = {
        "pass": gate["health_status_pass"],
        "reason": gate["health_reason"],
        "positive_signal": gate["positive_signal"],
        "negative_signal": gate["negative_signal"],
    }
    out["health_check"] = {
        "today": gate["today"],
        "exam_date": gate["exam_date"],
        "valid_from": gate["valid_from"],
        "valid_until": gate["valid_until"],
        "validity_days": gate["validity_days"],
        "dates_found": gate["dates_found"],
        "reason": gate["expiry_reason"],
    }
    out["health_status_pass"] = gate["health_status_pass"]
    out["health_expiry_pass"] = gate["health_expiry_pass"]
    out["health_exam_date"] = gate["exam_date"]
    out["health_valid_until"] = gate["valid_until"]
    out["issued_date"] = gate["exam_date"]
    out["issued_date_display"] = format_id_date(gate["exam_date"])

    explanation = dict(out.get("explanation") or {})
    details = [
        gate["health_reason"],
        gate["expiry_reason"],
        *list(explanation.get("detail_lines") or []),
    ]
    explanation["detail_lines"] = details
    gates = dict(explanation.get("gates") or {})
    gates["health_status"] = {"pass": gate["health_status_pass"]}
    gates["health_expiry"] = {
        "pass": gate["health_expiry_pass"],
        "exam_date": gate["exam_date"],
        "valid_until": gate["valid_until"],
        "today": gate["today"],
    }
    explanation["gates"] = gates

    blockers = list(explanation.get("primary_blockers") or [])
    failed = False
    if not gate["health_status_pass"]:
        failed = True
        if "HEALTH_STATUS" not in blockers:
            blockers.insert(0, "HEALTH_STATUS")
        explanation["summary"] = gate["health_reason"]
    if not gate["health_expiry_pass"]:
        failed = True
        if "HEALTH_CHECK_EXPIRY" not in blockers:
            blockers.insert(0, "HEALTH_CHECK_EXPIRY")
        if not gate["health_status_pass"]:
            explanation["summary"] = gate["health_reason"]
        else:
            explanation["summary"] = gate["expiry_reason"]

    if failed:
        out["document_matched"] = False
        explanation["primary_blockers"] = blockers
        verdict = dict(out.get("verdict") or {})
        verdict["document_matched"] = False
        verdict["summary"] = explanation.get("summary")
        out["verdict"] = verdict

    out["explanation"] = explanation
    return out
