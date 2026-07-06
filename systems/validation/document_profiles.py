"""Pemetaan jenis dokumen → daftar keyword (hanya backend, tidak dari klien)."""

from __future__ import annotations

from typing import Final

# id kanonik → label yang diharapkan muncul di OCR (dicek fuzzy satu per satu).
# KTP: campur label pendek (nik, nama, agama) yang sering kebaca meski field panjang rusak.
CANONICAL_KEYWORDS: Final[dict[str, list[str]]] = {
    "ktp": [
        "nik",
        "nama",
        "provinsi",
        "kabupaten",
        "agama",
        "kewarganegaraan",
        "status perkawinan",
    ],
    "npwp": ["npwp", "pajak", "wajib pajak"],
    "kk": ["kartu keluarga", "kepala keluarga", "nik"],
    # Rekening: hanya tabungan; e-statement dilarang (lihat PROFILE_EXCLUDED_KEYWORDS).
    "rekening": ["tabungan"],
    # Mutasi: tabungan + e-statement wajib keduanya.
    "mutasi": ["tabungan", "e-statement"],
    "skck": ["skck", "kepolisian"],
    # JKN: screenshot Mobile JKN — gate keyword lewat PROFILE_ANY_KEYWORD_GROUPS.
    "jkn": [],
    # Validasi berbasis gambar (wajah + latar biru), bukan OCR keyword.
    "foto_profile": [],
    # Ingest + index ke OpenSearch (tanpa validasi OCR).
    "cv": [],
}

# Profil yang divalidasi dari pixel gambar, bukan teks OCR.
IMAGE_ONLY_PROFILES: Final[frozenset[str]] = frozenset({"foto_profile"})

# Profil yang di-pipeline lewat subsistem CV (ingest + search).
CV_INGEST_PROFILES: Final[frozenset[str]] = frozenset({"cv"})

# Screenshot aplikasi / e-statement — jangan crop kartu atau full-bleed di preprocessing.
SCREEN_CAPTURE_PROFILES: Final[frozenset[str]] = frozenset({"mutasi", "rekening", "jkn"})

# Minimal satu keyword per grup harus lolos fuzzy (OR dalam grup, AND antar grup).
PROFILE_ANY_KEYWORD_GROUPS: Final[dict[str, list[list[str]]]] = {
    "jkn": [["info peserta", "faskes"]],
}

# Profil yang tidak memvalidasi nama vs expected_name (hanya keyword dokumen).
PROFILES_WITHOUT_IDENTITY: Final[frozenset[str]] = frozenset({"mutasi"})

# Keyword yang membuat profil gagal bila terdeteksi di OCR (fuzzy).
PROFILE_EXCLUDED_KEYWORDS: Final[dict[str, list[str]]] = {
    "rekening": ["e-statement"],
}

PROFILE_LABELS: Final[dict[str, str]] = {
    "ktp": "KTP",
    "npwp": "NPWP",
    "kk": "Kartu Keluarga",
    "rekening": "Rekening",
    "mutasi": "Mutasi",
    "skck": "SKCK",
    "jkn": "JKN (Info Peserta)",
    "foto_profile": "Foto Profil",
    "cv": "CV",
}

# Sinonim input pengguna (setelah strip + casefold, spasi tunggal antar kata)
ALIASES: Final[dict[str, str]] = {
    "kartu tanda penduduk": "ktp",
    "kartu tandapenduduk": "ktp",
    "id card": "ktp",
    "identitas": "ktp",
    "nomor pokok wajib pajak": "npwp",
    "npwp 16 digit": "npwp",
    "rekening koran": "rekening",
    "rekening tabungan": "rekening",
    "mutasi rekening": "mutasi",
    "e-statement": "mutasi",
    "surat keterangan catatan kepolisian": "skck",
    "jaminan kesehatan nasional": "jkn",
    "bpjs kesehatan": "jkn",
    "info peserta": "jkn",
    "mobile jkn": "jkn",
    "foto profil": "foto_profile",
    "foto profile": "foto_profile",
    "pas foto": "foto_profile",
    "pass foto": "foto_profile",
    "passport photo": "foto_profile",
    "curriculum vitae": "cv",
    "resume": "cv",
    "daftar riwayat hidup": "cv",
}


def list_supported_document_types() -> list[str]:
    return sorted(CANONICAL_KEYWORDS.keys())


def profile_label(profile_id: str) -> str:
    pid = (profile_id or "").strip().casefold()
    return PROFILE_LABELS.get(pid, profile_id.upper() if profile_id else "")


def is_image_only_profile(profile_id: str) -> bool:
    pid = (profile_id or "").strip().casefold()
    return pid in IMAGE_ONLY_PROFILES


def is_cv_ingest_profile(profile_id: str) -> bool:
    pid = (profile_id or "").strip().casefold()
    return pid in CV_INGEST_PROFILES


def is_screen_capture_profile(profile_id: str) -> bool:
    pid = (profile_id or "").strip().casefold()
    return pid in SCREEN_CAPTURE_PROFILES


def skip_physical_preprocess_isolation(profile_id: str) -> bool:
    """
    Lewati deteksi bidang kartu / full-bleed di preprocessing.

    Screenshot mutasi/rekening dan foto profil (crop biru hanya di validasi foto_profile).
    """
    pid = (profile_id or "").strip().casefold()
    return pid in IMAGE_ONLY_PROFILES or pid in SCREEN_CAPTURE_PROFILES


def skip_identity_validation(profile_id: str) -> bool:
    """Profil yang tidak membandingkan nama diekstrak vs expected_name."""
    pid = (profile_id or "").strip().casefold()
    return pid in PROFILES_WITHOUT_IDENTITY


def excluded_keywords_for_profile(profile_id: str) -> list[str]:
    pid = (profile_id or "").strip().casefold()
    return list(PROFILE_EXCLUDED_KEYWORDS.get(pid, []))


def any_keyword_groups_for_profile(profile_id: str) -> list[list[str]]:
    pid = (profile_id or "").strip().casefold()
    return [list(group) for group in PROFILE_ANY_KEYWORD_GROUPS.get(pid, [])]


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
