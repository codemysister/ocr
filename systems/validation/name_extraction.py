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


def extract_jkn_participant_names(ocr_text: str) -> list[str]:
    """Nama peserta dari kartu Mobile JKN (biasanya huruf kapital, bisa beberapa anggota)."""
    seen: set[str] = set()
    out: list[str] = []

    skip_next_value = False
    skip_markers = (
        "faskes",
        "lahir",
        "kelas",
        "kelompok",
        "tagihan",
        "saldo",
        "sisa",
    )
    non_name_tokens = frozenset(
        {
            "aktif",
            "tidak",
            "non",
            "bulan",
            "kelas",
            "lahir",
            "faskes",
            "kelompok",
            "peserta",
            "anak",
            "istri",
            "kepesertaan",
            "info",
            "jenis",
            "tampilan",
            "terdaftar",
            "prolanis",
            "hipertensi",
            "daerah",
            "pemerintah",
            "pbpu",
            "pbi",
            "apbn",
            "pbpu",
            "pega",
            "swasta",
        }
    )

    def _jkn_caps_is_name(raw: str) -> bool:
        s = raw.strip()
        if not s or re.fullmatch(r"[\d\s./:-]+", s):
            return False
        low = _normalize(s)
        words = low.split()
        if any(w in non_name_tokens for w in words):
            return False
        if any(low.startswith(marker) for marker in skip_markers):
            return False
        alpha = sum(c.isalpha() for c in s)
        if alpha < max(3, len(s) * 0.5):
            return False
        if s.upper() == s:
            if len(words) >= 2:
                return True
            if len(s) >= 4:
                return True
        return _looks_like_person_name(s)

    def _add(raw: str) -> None:
        cand = _trim_name_punctuation(raw)
        if not _jkn_caps_is_name(cand):
            return
        key = _normalize(cand)
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    segments = _split_segments(ocr_text)
    for i, seg in enumerate(segments):
        raw = seg.strip()
        low = _normalize(raw)
        if not raw:
            continue
        if any(low.startswith(marker) for marker in skip_markers):
            skip_next_value = True
            continue
        if skip_next_value:
            skip_next_value = False
            continue
        if raw.upper() == raw and re.search(r"[A-Z]{2,}", raw):
            _add(raw)
            if i + 1 < len(segments):
                nxt = segments[i + 1].strip()
                if (
                    nxt.upper() == nxt
                    and re.search(r"[A-Z]{2,}", nxt)
                    and len(nxt.split()) <= 2
                ):
                    _add(f"{raw} {nxt}")
        for m in _CAPS_NAME_RUN_RE.finditer(raw):
            _add(m.group(1))

    return out


def iuran_modal_only_state(ocr_text: str) -> bool:
    """Layar Info Iuran — modal peserta tanpa tagihan pribadi (nama biasanya tidak tampil)."""
    blob = _normalize(ocr_text)
    return (
        "tidak memiliki tagihan pribadi" in blob
        or "jenis peserta tidak terkategori" in blob
    )


def vaksinasi_1_first_dose_signal(ocr_text: str) -> bool:
    """Penanda dosis pertama pada kartu/surat/sertifikat internasional vaksin COVID-19."""
    blob = _normalize(ocr_text)
    compact = blob.replace(" ", "").replace("-", "")
    dose_markers = (
        "vaksin primer 1",
        "dosis pertama",
        "1st dose",
        "vaksin dosis pertama",
        "telah selesai di vaksin 1",
        "untuk dosis pertama",
    )
    if any(m.replace(" ", "") in compact or m in blob for m in dose_markers):
        return True

    intl_markers = (
        "international covid",
        "sertifikat vaksinasi covid-19 internasional",
        "vaccination certificate",
    )
    if not any(m in blob for m in intl_markers):
        return False
    return bool(
        re.search(
            r"(?:dose\s*number|dosis\s*ke|vaccination\s+details|rinician\s+vaksinasi)"
            r"[\s\S]{0,500}?\b1\b",
            ocr_text or "",
            re.I,
        )
    )


def vaksinasi_1_wrong_dose_primary(ocr_text: str) -> bool:
    """Kartu/surat yang utamanya primer 2+, booster, tanpa bukti dosis 1."""
    if vaksinasi_1_first_dose_signal(ocr_text):
        return False
    blob = _normalize(ocr_text)
    compact = blob.replace(" ", "")
    wrong_compact = (
        "vaksinprimer2",
        "vaksinprimer3",
        "vaksinbooster",
        "boosterpertama",
        "vaksinboosterpertama",
    )
    return any(w in compact for w in wrong_compact)


def extract_vaksinasi_1_names(ocr_text: str) -> list[str]:
    """Nama penerima vaksin pada kartu/surat/sertifikat dosis pertama."""
    seen: set[str] = set()
    out: list[str] = []

    def _add(raw: str) -> None:
        cand = _trim_name_punctuation(raw.lstrip(":").strip())
        if not cand:
            return
        key = _normalize(cand)
        if key in seen:
            return
        if _is_probably_field_label(cand):
            return
        if _looks_like_person_name(cand) or (
            cand.upper() == cand and re.search(r"[A-Z]{2,}", cand) and len(cand) >= 4
        ):
            seen.add(key)
            out.append(cand)

    segments = _split_segments(ocr_text)
    for i, seg in enumerate(segments):
        raw = seg.strip()
        if not raw:
            continue
        low = _normalize(raw)
        if low in {"diberikan kepada", "nama lengkap", "full name", "nama"}:
            if i + 1 < len(segments):
                _add(segments[i + 1])
        if low.startswith("diberikan kepada"):
            rest = raw.split("kepada", 1)[-1].strip(" :")
            if rest and rest != raw:
                _add(rest)
        if "diberikan kepada" in low and ":" in raw:
            _add(raw.split(":", 1)[-1])
        if low in {"sertifikat ini diberikan kepada", "this is to certify that"}:
            if i + 1 < len(segments):
                _add(segments[i + 1])
        if ":" in raw and low.split(":", 1)[0].strip() in {"nama lengkap", "nama"}:
            _add(raw.split(":", 1)[1])

    for i, seg in enumerate(segments):
        low = _normalize(seg)
        if low == "program pemerintah" and i + 1 < len(segments):
            _add(segments[i + 1])
            break

    for seg in segments:
        for m in _CAPS_NAME_RUN_RE.finditer(seg):
            _add(m.group(1))

    return out


def extract_iuran_participant_names(ocr_text: str) -> list[str]:
    """Nama peserta pada kartu Info Iuran (huruf kapital, bisa beberapa anggota)."""
    seen: set[str] = set()
    out: list[str] = []

    skip_markers = (
        "sisa saldo",
        "tagihan",
        "total tagihan",
        "batas waktu",
        "jenis peserta",
        "peserta mandiri",
        "kembali",
        "tidak memiliki",
        "info iuran",
    )
    non_name_tokens = frozenset(
        {
            "saldo",
            "tagihan",
            "total",
            "kembali",
            "mandiri",
            "pekerja",
            "bulan",
            "berjalan",
            "tanggal",
            "pembayaran",
            "peserta",
            "bukan",
            "terkategori",
            "pribadi",
            "pbpu",
            "bp",
        }
    )

    def _iuran_caps_is_name(raw: str) -> bool:
        s = raw.strip()
        if not s or re.fullmatch(r"[\d\s./:-]+", s):
            return False
        if re.fullmatch(r"0\d{12}", re.sub(r"\s+", "", s)):
            return False
        low = _normalize(s)
        words = low.split()
        if any(w in non_name_tokens for w in words):
            return False
        if any(low.startswith(marker) for marker in skip_markers):
            return False
        alpha = sum(c.isalpha() for c in s)
        if alpha < max(3, len(s) * 0.5):
            return False
        if s.upper() == s:
            if len(words) >= 2:
                return True
            if len(s) >= 4:
                return True
        return _looks_like_person_name(s)

    def _add(raw: str) -> None:
        cand = _trim_name_punctuation(raw)
        if not _iuran_caps_is_name(cand):
            return
        key = _normalize(cand)
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    skip_next_value = False
    segments = _split_segments(ocr_text)
    for i, seg in enumerate(segments):
        raw = seg.strip()
        low = _normalize(raw)
        if not raw:
            continue
        if any(low.startswith(marker) for marker in skip_markers):
            skip_next_value = True
            continue
        if skip_next_value:
            skip_next_value = False
            continue
        if raw.upper() == raw and re.search(r"[A-Z]{2,}", raw):
            _add(raw)
            if i + 1 < len(segments):
                nxt = segments[i + 1].strip()
                if (
                    nxt.upper() == nxt
                    and re.search(r"[A-Z]{2,}", nxt)
                    and len(nxt.split()) <= 2
                ):
                    _add(f"{raw} {nxt}")
        for m in _CAPS_NAME_RUN_RE.finditer(raw):
            _add(m.group(1))

    return out


def extract_bpjs_kesanggupan_name(ocr_text: str) -> str | None:
    """Nama dari surat pernyataan kesanggupan BPJS (field Nama / tanda tangan)."""
    segments = _split_segments(ocr_text)

    def _clean(raw: str) -> str:
        return _trim_name_punctuation(raw.lstrip(":").strip())

    for i, seg in enumerate(segments):
        raw = seg.strip()
        if not raw:
            continue
        low = _normalize(raw)
        if raw.startswith(":"):
            cand = _clean(raw)
            if cand and (_looks_like_person_name(cand) or len(cand.split()) >= 2):
                return cand
        if low in {"nama lengkap", "nama", "nana"}:
            if i + 1 < len(segments):
                nxt = _clean(segments[i + 1])
                if nxt and not _is_probably_field_label(nxt):
                    return nxt
        if ":" in raw and low.split(":", 1)[0].strip() in {"nama lengkap", "nama"}:
            cand = _clean(raw.split(":", 1)[1])
            if cand:
                return cand

    for seg in reversed(segments):
        raw = seg.strip()
        if not raw:
            continue
        low = _normalize(raw)
        if low in {"hormat saya", "yang menyatakan", "meterai tempel"}:
            continue
        m = re.search(r"\(([^)]+)\)", raw)
        if m:
            cand = _clean(m.group(1).strip("."))
            if cand and _looks_like_person_name(cand):
                return cand
        if _looks_like_person_name(raw) and low not in {"bekasi", "hormat saya"}:
            return raw

    return None


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

    if profile == "bpjs_tk":
        for seg in person_like_segments(ocr_text):
            if seg.upper() == seg and re.search(r"[A-Z]{2,}", seg) and len(seg.split()) >= 1:
                words = seg.split()
                if len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 4):
                    return seg, "bpjs_tk_caps_name"

    if profile == "bpjs_kesanggupan":
        ks_name = extract_bpjs_kesanggupan_name(ocr_text)
        if ks_name:
            return ks_name, "bpjs_kesanggupan_name"

    if profile == "bpjs":
        segments = _split_segments(ocr_text)
        for i, seg in enumerate(segments):
            raw = seg.strip()
            low = _normalize(raw)
            if low in {"nama", "nana", "name", "lama"} and i + 1 < len(segments):
                nxt = segments[i + 1].strip()
                if nxt and not _is_probably_field_label(nxt):
                    merged = re.sub(r"\s+", " ", nxt)
                    if _looks_like_person_name(merged) or (
                        merged.upper() == merged and re.search(r"[A-Z]{2,}", merged)
                    ):
                        return merged, "bpjs_after_nama"
        for seg in person_like_segments(ocr_text):
            if seg.upper() == seg and re.search(r"[A-Z]{2,}", seg) and len(seg.split()) >= 2:
                return seg, "bpjs_caps_name"

    if profile == "jkn":
        jkn_names = extract_jkn_participant_names(ocr_text)
        if jkn_names:
            return jkn_names[0], "jkn_participant_name"
        for seg in person_like_segments(ocr_text):
            if len(seg.split()) >= 2:
                return seg, "jkn_participant_name"

    if profile == "iuran":
        iuran_names = extract_iuran_participant_names(ocr_text)
        if iuran_names:
            return iuran_names[0], "iuran_participant_name"
        for seg in person_like_segments(ocr_text):
            words = seg.split()
            if len(words) >= 2 and seg.upper() == seg:
                return seg, "iuran_participant_name"

    if profile == "vaksinasi_1":
        vax_names = extract_vaksinasi_1_names(ocr_text)
        if vax_names:
            return vax_names[0], "vaksinasi_1_recipient_name"
        for seg in person_like_segments(ocr_text):
            if seg.upper() == seg and re.search(r"[A-Z]{2,}", seg) and len(seg.split()) >= 1:
                return seg, "vaksinasi_1_recipient_name"

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
