"""Heuristik ekstraksi nama pemilik dari teks OCR KTP/NPWP (tanpa layout bbox)."""

from __future__ import annotations

import re


def _normalize(s: str) -> str:
    s = (s or "").strip().casefold()
    return re.sub(r"\s+", " ", s)


def _split_segments(ocr_text: str) -> list[str]:
    """Pisah sel tabel / baris; jangan pecah tanda hubung di nama (mis. el-sharawy)."""
    text = re.sub(r"\|?\s*---+(\s*\|?\s*---+)*\s*\|?", " | ", ocr_text or "")
    parts = re.split(r"\s*\|\s*|\n+|•+", text)
    return [p.strip() for p in parts if p.strip()]


# Baris anggota KK (markdown Mistral): | no | nama | nik 16 digit | ...
_KK_MEMBER_ROW_RE = re.compile(
    r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*(\d{16})\s*\|",
    re.IGNORECASE,
)


_LABEL_ONE_WORD = frozenset(
    {
        "nama",
        "nana",
        "name",
        "nik",
        "npwp",
        "ttl",
        "wni",
        "wna",
    }
)


def _is_probably_field_label(seg: str) -> bool:
    s = _normalize(seg)
    if not s:
        return True
    if s in _LABEL_ONE_WORD:
        return True
    if len(s) <= 3:
        return True
    low = s.replace(" ", "")
    if low.startswith("tempat") or low.startswith("ttl") or low.startswith("lahir"):
        return True
    return False


_PLACE_PREFIX = re.compile(r"^(kota|kab\.?|kabupaten|kecamatan|kelurahan|provinsi|desa)\b", re.I)


def _is_likely_tempat_tgl_ocr_noise(s: str) -> bool:
    """Bukan nama orang: sisa OCR dari Tempat / Tgl lahir (mis. «tempattol»)."""
    low = (s or "").casefold().replace(" ", "")
    if "tempat" in low:
        return True
    if low.startswith("ttl") and len(low) <= 12:
        return True
    if "lahir" in low and len(low) <= 14:
        return True
    return False


def _looks_like_person_name(seg: str) -> bool:
    s = (seg or "").strip()
    if len(s) < 4:
        return False
    if _is_likely_tempat_tgl_ocr_noise(s):
        return False
    if _PLACE_PREFIX.search(s):
        return False
    alpha = sum(1 for c in s if c.isalpha())
    if alpha < 3 or alpha / max(len(s), 1) < 0.45:
        return False
    if re.fullmatch(r"\d[\d\s./-]*", s):
        return False
    if re.search(r"\d{10,}", s):
        return False
    if re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", s):
        return False
    words = s.split()
    if len(words) >= 2:
        return True
    return len(s) >= 8


def extract_kk_member_names(ocr_text: str) -> list[str]:
    """Nama dari tabel anggota Kartu Keluarga (kolom nama lengkap + NIK 16 digit)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _KK_MEMBER_ROW_RE.finditer(ocr_text or ""):
        raw = m.group(1).strip()
        if not raw or raw in {"-", "—"}:
            continue
        if not _looks_like_person_name(raw):
            continue
        key = _normalize(raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


_GREETING_PREFIX_RE = re.compile(
    r"^(?:selamat\s+(?:pagi|siang|sore|malam|datang)|halo|hai|hi|hello)\s*[,!]?\s*",
    re.I,
)

# Nama setelah sapaan aplikasi/perbankan di tengah teks OCR (mis. «selamat siang, usep maulidin!»).
_GREETING_NAME_INLINE_RE = re.compile(
    r"(?:^|[\s,.])(?:selamat\s+(?:pagi|siang|sore|malam)|halo|hai|hi|hello)\s*[,!]?\s*"
    r"([a-zA-Z][a-zA-Z\s'.-]{2,50}?)"
    r"(?=\s*[!?.]|\s+(?:mau|ingin|dan|yang|untuk|silakan|welcome|good|terima)\b|$)",
    re.I,
)

_AFTER_COMMA_NAME_RE = re.compile(
    r",\s*([A-Za-z][A-Za-z\s'.-]{2,50}?)(?=\s*[!?.]|\s{2,}|\s+[a-z]{3,}\b|$)",
)

_CAPS_NAME_RUN_RE = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b")


def _trim_name_punctuation(cand: str) -> str:
    return re.sub(r"[!?.;]+$", "", (cand or "").strip()).strip()


def _refined_name_candidates(seg: str) -> list[str]:
    """Kandidat nama lebih rapat dari segmen kasar (sapaan UI, koma, huruf kapital)."""
    s = (seg or "").strip()
    if not s:
        return []

    out: list[str] = []

    def _add(cand: str) -> None:
        c = _trim_name_punctuation(cand)
        if c and _looks_like_person_name(c):
            out.append(c)

    stripped = _GREETING_PREFIX_RE.sub("", s).strip()
    if stripped and stripped != s:
        first_clause = re.split(r"[!?]", stripped, maxsplit=1)[0].strip()
        _add(first_clause)

    for m in _GREETING_NAME_INLINE_RE.finditer(s):
        _add(m.group(1))

    for m in _AFTER_COMMA_NAME_RE.finditer(s):
        _add(m.group(1))

    for m in _CAPS_NAME_RUN_RE.finditer(s):
        _add(m.group(1))

    return out


def sliding_name_windows(text: str, expected_name: str, *, max_extra_words: int = 1) -> list[str]:
    """Jendela kata berurutan dengan panjang mendekati nama referensi (untuk OCR blob panjang)."""
    exp_words = _normalize(expected_name).split()
    if len(exp_words) < 2:
        return []
    words = re.findall(r"[a-zA-Z][a-zA-Z'.-]*", text or "", re.I)
    if len(words) < len(exp_words):
        return []

    n = len(exp_words)
    seen: set[str] = set()
    out: list[str] = []
    for width in range(n, min(n + max_extra_words + 1, len(words) + 1)):
        for i in range(len(words) - width + 1):
            window = " ".join(words[i : i + width])
            key = _normalize(window)
            if key in seen:
                continue
            if not _looks_like_person_name(window):
                continue
            seen.add(key)
            out.append(window)
    return out


def person_like_segments(ocr_text: str) -> list[str]:
    """Segmen OCR yang lolos heuristik nama orang; urutan asli, tanpa duplikat (normalisasi)."""
    seen: set[str] = set()
    out: list[str] = []

    def _append(cand: str) -> None:
        key = _normalize(cand)
        if key in seen:
            return
        seen.add(key)
        out.append(cand.strip())

    for seg in _split_segments(ocr_text):
        if _is_probably_field_label(seg):
            continue
        if _looks_like_person_name(seg):
            _append(seg.strip())
        for refined in _refined_name_candidates(seg):
            _append(refined)

    for refined in _refined_name_candidates(ocr_text or ""):
        _append(refined)

    return out


def extract_holder_name_candidate(ocr_text: str, document_profile_id: str) -> tuple[str | None, str]:
    """
    Mengembalikan (teks nama perkiraan atau None, metode / alasan).
    `document_profile_id` kanonik: ktp, npwp, dll.
    """
    profile = (document_profile_id or "").strip().casefold()
    if profile == "kk":
        kk_names = extract_kk_member_names(ocr_text)
        if kk_names:
            return kk_names[0], "kk_table_first_member"

    segments = _split_segments(ocr_text)

    for i, seg in enumerate(segments):
        raw = seg.strip()
        low = _normalize(raw)
        if ":" in raw and low.split(":", 1)[0].strip() in {"nama", "nana", "name"}:
            after = raw.split(":", 1)[1].strip()
            if _looks_like_person_name(after):
                return after, "nama_inline_colon"
        if low in {"nama", "nana", "name"} and i + 1 < len(segments):
            nxt = segments[i + 1].strip()
            if not _is_probably_field_label(nxt) and _looks_like_person_name(nxt):
                return nxt, "after_nama_segment"

    # Blob huruf kecil: pola memakai \b agar "nama" tidak cocok di dalam kata (mis. kewarganegaraan).
    blob = _normalize(ocr_text)
    m = re.search(
        r"(?:^|[\s|•/+\-])(?:nama|nana|name)\b\s*[:-]?\s*"
        r"([a-z][a-z0-9\s'.]{2,80}?)"
        r"(?=\s*(?:[-|]|\n|tempat|ttl|lahir|nik|npwp|alamat|agama|gol|status|pekerjaan|kewarganegaraan|wni|wna)\b|$)",
        blob,
    )
    if m:
        c = m.group(1).strip()
        if _looks_like_person_name(c):
            return c, "regex_nama"

    return None, "failed"
