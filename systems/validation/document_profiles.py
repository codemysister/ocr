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
    # Kartu Indonesia Sehat (KIS) / kartu fisik BPJS Kesehatan.
    "bpjs": [
        "kartu indonesia sehat",
        "bpjs kesehatan",
        "nomor kartu",
        "nik",
        "faskes",
        "syarat dan ketentuan",
    ],
    # Kartu Peserta BPJS Ketenagakerjaan — gate keyword lewat PROFILE_ANY_KEYWORD_GROUPS.
    "bpjs_tk": [],
    # Surat pernyataan kesanggupan menanggung biaya BPJS Kesehatan.
    "bpjs_kesanggupan": [],
    # JKN: screenshot Mobile JKN — gate keyword lewat PROFILE_ANY_KEYWORD_GROUPS.
    "jkn": [],
    # Iuran JKN: layar Info Iuran (tagihan / modal tidak punya tagihan pribadi).
    "iuran": [],
    # Sertifikat / kartu vaksinasi COVID-19 dosis pertama.
    "vaksinasi_1": [],
    "vaksinasi_2": [],
    "vaksinasi_3": [],
    "ijasah": ["ijazah", "pendidikan", "lulus", "nomor ijazah"],
    "transkrip": ["transkrip", "nilai", "semester", "ipk"],
    "formulir_okb": ["formulir okb", "okb", "pemeriksaan"],
    "formulir_lamaran": ["formulir lamaran", "lamaran pekerjaan", "data pribadi"],
    "surat_lamaran": ["surat lamaran", "yang bertanda tangan", "pekerjaan"],
    "pemadanan_npwp": ["pemadanan", "npwp", "direktorat jenderal pajak"],
    "keterangan_kesehatan": ["surat keterangan", "kesehatan", "dokter", "medis"],
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
SCREEN_CAPTURE_PROFILES: Final[frozenset[str]] = frozenset(
    {"mutasi", "rekening", "jkn", "iuran", "vaksinasi_1", "vaksinasi_2", "vaksinasi_3"}
)

# Minimal satu keyword per grup harus lolos fuzzy (OR dalam grup, AND antar grup).
PROFILE_ANY_KEYWORD_GROUPS: Final[dict[str, list[list[str]]]] = {
    "jkn": [["info peserta", "faskes"]],
    "bpjs_tk": [
        ["kartu peserta"],
        ["ketenagakerjaan", "bpjs ketenagakerjaan"],
    ],
    "bpjs_kesanggupan": [
        [
            "surat pernyataan kesanggupan",
            "menanggung biaya bpjs kesehatan",
            "menanggung biaya bpjs",
        ],
        [
            "tidak aktif",
            "menanggung secara pribadi",
            "syarat bekerja",
            "peserta mandiri",
            "virtual account",
        ],
    ],
    "iuran": [
        ["info iuran", "info luran"],
        [
            "total tagihan",
            "sisa saldo",
            "tidak memiliki tagihan pribadi",
            "jenis peserta tidak terkategori",
            "batas waktu pembayaran",
        ],
    ],
    "vaksinasi_1": [
        [
            "kartu vaksinasi covid",
            "surat keterangan vaksinasi",
            "covid-19 vaksin",
            "sertifikat vaksinasi covid",
            "international covid-19 vaccination",
        ],
        [
            "vaksin primer 1",
            "dosis pertama",
            "1st dose",
            "vaksin dosis pertama",
            "telah selesai di vaksin 1",
            "untuk dosis pertama",
        ],
    ],
    "vaksinasi_2": [
        [
            "kartu vaksinasi covid",
            "surat keterangan vaksinasi",
            "covid-19 vaksin",
            "sertifikat vaksinasi covid",
            "international covid-19 vaccination",
        ],
        [
            "vaksin primer 2",
            "dosis kedua",
            "2nd dose",
            "vaksin dosis kedua",
            "telah selesai di vaksin 2",
            "untuk dosis kedua",
            "vaksin 2",
        ],
    ],
    "vaksinasi_3": [
        [
            "kartu vaksinasi covid",
            "surat keterangan vaksinasi",
            "covid-19 vaksin",
            "sertifikat vaksinasi covid",
            "international covid-19 vaccination",
        ],
        [
            "vaksin booster",
            "dosis ketiga",
            "3rd dose",
            "booster",
            "vaksin dosis ketiga",
            "telah selesai di vaksin 3",
            "untuk dosis ketiga",
        ],
    ],
}

# Profil yang tidak memvalidasi nama vs expected_name (hanya keyword dokumen).
PROFILES_WITHOUT_IDENTITY: Final[frozenset[str]] = frozenset({"mutasi"})

# Keyword yang membuat profil gagal bila terdeteksi di OCR (fuzzy).
PROFILE_EXCLUDED_KEYWORDS: Final[dict[str, list[str]]] = {
    "rekening": ["e-statement"],
    "jkn": ["info iuran", "total tagihan", "tidak memiliki tagihan pribadi", "kartu indonesia sehat", "bpjs ketenagakerjaan"],
    "iuran": ["faskes 1", "kepesertaan terdaftar", "jenis tampilan"],
    "bpjs": ["info iuran", "kepesertaan terdaftar", "kartu vaksinasi covid", "vaksin booster", "bpjs ketenagakerjaan", "ketenagakerjaan"],
    "bpjs_tk": ["syarat dan ketentuan", "faskes tingkat", "info iuran", "kepesertaan terdaftar"],
    "bpjs_kesanggupan": ["kartu indonesia sehat", "ketenagakerjaan"],
    "vaksinasi_1": ["info peserta", "info iuran"],
    "vaksinasi_2": ["info peserta", "info iuran", "dosis pertama", "1st dose"],
    "vaksinasi_3": ["info peserta", "info iuran", "dosis pertama", "dosis kedua"],
}

PROFILE_LABELS: Final[dict[str, str]] = {
    "ktp": "KTP",
    "npwp": "NPWP",
    "kk": "Kartu Keluarga",
    "rekening": "Rekening",
    "mutasi": "Mutasi",
    "skck": "SKCK",
    "bpjs": "BPJS Kesehatan (KIS)",
    "bpjs_tk": "BPJS Ketenagakerjaan",
    "bpjs_kesanggupan": "Kesanggupan BPJS Kesehatan",
    "jkn": "JKN (Info Peserta)",
    "iuran": "Iuran JKN (Info Iuran)",
    "vaksinasi_1": "Vaksinasi COVID-19 (Dosis 1)",
    "vaksinasi_2": "Vaksinasi COVID-19 (Dosis 2)",
    "vaksinasi_3": "Vaksinasi COVID-19 (Dosis 3 / Booster)",
    "ijasah": "Ijazah",
    "transkrip": "Transkrip Nilai",
    "formulir_okb": "Formulir OKB",
    "formulir_lamaran": "Formulir Lamaran Pekerjaan",
    "surat_lamaran": "Surat Lamaran",
    "pemadanan_npwp": "Pemadanan NPWP",
    "keterangan_kesehatan": "Surat Keterangan Kesehatan",
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
    "kartu keluarga": "kk",
    "rekening bank": "rekening",
    "rekening koran": "rekening",
    "rekening tabungan": "rekening",
    "mutasi rekening": "mutasi",
    "e-statement": "mutasi",
    "surat keterangan catatan kepolisian": "skck",
    "jaminan kesehatan nasional": "jkn",
    "bpjs kesehatan": "bpjs",
    "kartu indonesia sehat": "bpjs",
    "kis": "bpjs",
    "kartu kis": "bpjs",
    "bpjs ketenagakerjaan": "bpjs_tk",
    "bpjs tk": "bpjs_tk",
    "jaminan ketenagakerjaan": "bpjs_tk",
    "kartu peserta bpjs": "bpjs_tk",
    "kesanggupan bpjs": "bpjs_kesanggupan",
    "bpjs kesanggupan": "bpjs_kesanggupan",
    "surat kesanggupan bpjs": "bpjs_kesanggupan",
    "surat pernyataan kesanggupan": "bpjs_kesanggupan",
    "kesanggupan menanggung biaya": "bpjs_kesanggupan",
    "info peserta": "jkn",
    "peserta jkn": "jkn",
    "mobile jkn": "jkn",
    "info iuran": "iuran",
    "iuran jkn": "iuran",
    "tagihan jkn": "iuran",
    "iuran bpjs": "iuran",
    "vaksinasi 1": "vaksinasi_1",
    "vaksinasi covid dosis 1": "vaksinasi_1",
    "sertifikat vaksin covid": "vaksinasi_1",
    "kartu vaksinasi covid": "vaksinasi_1",
    "vaksin dosis pertama": "vaksinasi_1",
    "covid-19 vaksin dosis pertama": "vaksinasi_1",
    "vaksinasi 2": "vaksinasi_2",
    "vaksinasi covid dosis 2": "vaksinasi_2",
    "vaksin dosis kedua": "vaksinasi_2",
    "covid-19 vaksin dosis kedua": "vaksinasi_2",
    "vaksinasi 3": "vaksinasi_3",
    "vaksinasi covid dosis 3": "vaksinasi_3",
    "vaksin booster": "vaksinasi_3",
    "vaksin dosis ketiga": "vaksinasi_3",
    "covid-19 vaksin booster": "vaksinasi_3",
    "ijasah": "ijasah",
    "ijazah": "ijasah",
    "transkrip": "transkrip",
    "transkrip nilai": "transkrip",
    "formulir okb": "formulir_okb",
    "okb": "formulir_okb",
    "formulir lamaran pekerjaan": "formulir_lamaran",
    "formulir lamaran": "formulir_lamaran",
    "surat lamaran": "surat_lamaran",
    "pemadanan npwp": "pemadanan_npwp",
    "keterangan kesehatan": "keterangan_kesehatan",
    "surat keterangan kesehatan": "keterangan_kesehatan",
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
