"""Pemetaan jenis dokumen → daftar keyword (hanya backend, tidak dari klien)."""

from __future__ import annotations

from typing import Final

# id kanonik → label yang diharapkan muncul di OCR (dicek fuzzy satu per satu)
CANONICAL_KEYWORDS: Final[dict[str, list[str]]] = {
    "ktp": ["nik", "kewarganegaraan", "agama"],
    "npwp": ["npwp", "pajak"],
    'kk': ['kartu keluarga', 'dinas kependudukan'],
}

# Sinonim input pengguna (setelah strip + casefold, spasi tunggal antar kata)
ALIASES: Final[dict[str, str]] = {
    "kartu tanda penduduk": "ktp",
    "kartu tandapenduduk": "ktp",
    "id card": "ktp",
    "identitas": "ktp",
    "nomor pokok wajib pajak": "npwp",
    "npwp 16 digit": "npwp",
}


def list_supported_document_types() -> list[str]:
    return sorted(CANONICAL_KEYWORDS.keys())


def resolve_keywords(document_type: str) -> tuple[str, list[str]] | None:
    """
    Selesaikan `document_type` (teks bebas ringkas) ke (id kanonik, daftar keyword).
    Mengembalikan None jika kosong atau tidak cocok profil manapun.
    """
    raw = (document_type or "").strip().casefold()
    if not raw:
        return None
    spaced = " ".join(raw.split())
    if spaced in ALIASES:
        cid = ALIASES[spaced]
        return cid, list(CANONICAL_KEYWORDS[cid])
    if spaced in CANONICAL_KEYWORDS:
        return spaced, list(CANONICAL_KEYWORDS[spaced])
    compact = spaced.replace(" ", "")
    for cid in CANONICAL_KEYWORDS:
        if compact == cid.replace(" ", ""):
            return cid, list(CANONICAL_KEYWORDS[cid])
    return None
