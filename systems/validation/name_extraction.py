"""Heuristik ekstraksi nama pemilik dari teks OCR KTP/NPWP (tanpa layout bbox)."""

from __future__ import annotations

import re


def _normalize(s: str) -> str:
    s = (s or "").strip().casefold()
    return re.sub(r"\s+", " ", s)


def _split_segments(ocr_text: str) -> list[str]:
    parts = re.split(r"\s*[-|•]+\s*|\n+", ocr_text)
    return [p.strip() for p in parts if p.strip()]


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


def person_like_segments(ocr_text: str) -> list[str]:
    """Segmen OCR yang lolos heuristik nama orang; urutan asli, tanpa duplikat (normalisasi)."""
    seen: set[str] = set()
    out: list[str] = []
    for seg in _split_segments(ocr_text):
        if _is_probably_field_label(seg):
            continue
        if not _looks_like_person_name(seg):
            continue
        key = _normalize(seg)
        if key in seen:
            continue
        seen.add(key)
        out.append(seg.strip())
    return out


def extract_holder_name_candidate(ocr_text: str, document_profile_id: str) -> tuple[str | None, str]:
    """
    Mengembalikan (teks nama perkiraan atau None, metode / alasan).
    `document_profile_id` kanonik: ktp, npwp, dll.
    """
    _ = document_profile_id
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
