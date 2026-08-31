"""Heuristik ekstraksi nama pemilik dari teks OCR KTP/NPWP (tanpa layout bbox)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz


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
_KK_FAMILY_NO_RE = re.compile(r"no\.?\s*(\d{16})", re.I)
_KK_PLAIN_NAME_LINE_RE = re.compile(r"^[A-Z][A-Z\s'.-]{3,}$")
_KK_PLAIN_NAME_SKIP = frozenset(
    {
        "kartu keluarga",
        "kepala keluarga",
        "status hubungan",
        "nama lengkap",
        "dikeluarkan tanggal",
        "tanda taigan",
        "cap jempol",
        "daiam keluarga",
        "no paspor",
        "dekumen imigrasi",
        "jenis pekerjasn",
        "mengurus rumah tangga",
        "pelajar mahasiswa",
        "pelaiarmahasiswa",
        "slta sederajat",
        "sltp sederajat",
        "desa kelurahan",
        "kabupaten kota",
        "kecamatan",
        "provinsi",
        "kode pos",
        "alamat",
        "rt rw",
        "lembar",
    }
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

# Kata/frasa label KTP — bukan nama orang (mis. «Kewarganegaraan WNI»).
_NON_NAME_FIELD_WORDS = frozenset(
    {
        "kewarganegaraan",
        "wni",
        "wna",
        "perkawinan",
        "kawin",
        "bekerja",
        "tidak",
        "islam",
        "kristen",
        "katolik",
        "hindu",
        "buddha",
        "konghucu",
        "perempuan",
        "laki",
        "lakilaki",
        "seumur",
        "hidup",
        "berlaku",
        "hingga",
        "tempat",
        "lahir",
        "agama",
        "status",
        "pekerjaan",
        "alamat",
        "kecamatan",
        "kelurahan",
        "desa",
        "provinsi",
        "kabupaten",
        "belum",
        "cerai",
        "meninggal",
        "dusun",
        "gol",
        "darah",
    }
)


def _contains_field_label_word(seg: str) -> bool:
    for w in _normalize(seg).split():
        if w in _NON_NAME_FIELD_WORDS:
            return True
    return False


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
    if _contains_field_label_word(s):
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
    # KTP: nama satu kata kapital (ANNISA, BUDI) sering 4–7 huruf — jangan paksa ≥8.
    w = words[0]
    letter_chars = sum(1 for c in w if c.isalpha())
    if letter_chars >= 4 and letter_chars >= len(w) * 0.85:
        if w.upper() == w and 4 <= len(w) <= 40:
            return True
    return len(s) >= 8


def extract_kk_family_card_number(ocr_text: str) -> str | None:
    """Nomor KK 16 digit (bukan NIK anggota)."""
    m = _KK_FAMILY_NO_RE.search(ocr_text or "")
    return m.group(1) if m else None


def extract_kk_member_niks_ordered(ocr_text: str) -> list[str]:
    """NIK anggota KK berurutan; nomor kartu keluarga dikecualikan."""
    family_no = extract_kk_family_card_number(ocr_text)
    seen: set[str] = set()
    out: list[str] = []
    for line in (ocr_text or "").splitlines():
        digits = re.sub(r"\D", "", line)
        if len(digits) != 16:
            continue
        if digits == family_no or digits in seen:
            continue
        seen.add(digits)
        out.append(digits)
    if out:
        return out
    for m in re.finditer(r"(?<!\d)(\d{16})(?!\d)", ocr_text or ""):
        digits = m.group(1)
        if digits == family_no or digits in seen:
            continue
        seen.add(digits)
        out.append(digits)
    return out


def extract_kk_member_name_nik_pairs(ocr_text: str) -> list[tuple[str, str]]:
    """Pasangan (nama, nik) dari tabel KK — markdown atau urutan kolom OCR."""
    pairs: list[tuple[str, str]] = []
    for m in _KK_MEMBER_ROW_RE.finditer(ocr_text or ""):
        raw = m.group(1).strip()
        nik = m.group(2)
        if not raw or raw in {"-", "—"} or not _looks_like_person_name(raw):
            continue
        pairs.append((raw, nik))
    if pairs:
        return pairs

    niks = extract_kk_member_niks_ordered(ocr_text)
    names = _extract_kk_member_names_ordered_plain(ocr_text)
    if niks and names and len(niks) == len(names):
        return list(zip(names, niks))
    return []


def _extract_kk_member_names_ordered_plain(ocr_text: str) -> list[str]:
    """Nama anggota berurutan dari OCR Paddle (tanpa pipe markdown)."""
    out: list[str] = []
    seen: set[str] = set()
    for line in (ocr_text or "").splitlines():
        s = line.strip()
        if not s or not _KK_PLAIN_NAME_LINE_RE.match(s):
            continue
        low = _normalize(s)
        if low in _KK_PLAIN_NAME_SKIP:
            continue
        if any(skip in low for skip in _KK_PLAIN_NAME_SKIP):
            continue
        if not _looks_like_person_name(s):
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out


def extract_kk_name_by_nik(ocr_text: str, nik: str) -> str | None:
    """Nama anggota KK yang NIK-nya cocok (jika pasangan nama–NIK terbaca)."""
    digits = re.sub(r"\D", "", nik or "")
    if len(digits) != 16:
        return None
    for name, row_nik in extract_kk_member_name_nik_pairs(ocr_text):
        if row_nik == digits:
            return name
    return None


def extract_kk_member_names(ocr_text: str) -> list[str]:
    """Nama dari tabel anggota Kartu Keluarga (kolom nama lengkap + NIK 16 digit)."""
    seen: set[str] = set()
    out: list[str] = []
    for name, _ in extract_kk_member_name_nik_pairs(ocr_text):
        key = _normalize(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
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
    for raw in _extract_kk_member_names_ordered_plain(ocr_text):
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

_CAPS_NAME_RUN_RE = re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{2,})+)\b")

_NPWP_NUMBER_IN_TEXT_RE = re.compile(r"\d{4}[\s.\-]?\d{4}[\s.\-]?\d{4}[\s.\-]?\d{4}")

_NPWP_LINE_SKIP_FRAGMENTS = (
    "kantor pelayanan",
    "www.",
    "tanggal terdaftar",
    "pajak kita",
    "untuk kita",
    " rt",
    " rw",
    "kab.",
    "jawa barat",
    "djp",
    "npwp",
    "np vp",
)


def _npwp_line_is_noise(line: str) -> bool:
    low = _normalize(line)
    if not low:
        return True
    if any(frag in low for frag in _NPWP_LINE_SKIP_FRAGMENTS):
        return True
    if re.search(r"\d{6,}", line):
        return True
    return False


def _npwp_name_plausible(cand: str) -> bool:
    """Tolak sampah OCR singkat (mis. «OA AH» dari logo/belakang kartu)."""
    s = (cand or "").strip()
    if not s:
        return False
    words = s.split()
    if not words:
        return False
    if all(len(w) <= 2 for w in words):
        return False
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False
    if s.upper() == s:
        if len(words) >= 2 and all(len(w) <= 3 for w in words):
            return False
        if sum(len(w) for w in words) < 8:
            return False
    return _looks_like_person_name(s) or (len(words) >= 2 and sum(len(w) for w in words) >= 10)


def _merge_npwp_adjacent_name_lines(lines: list[str]) -> list[str]:
    """NPWP lama: nama depan+belakang sering di baris terpisah (mis. ACHMAD SAEFUL + RHOMADHONI)."""
    if not lines:
        return lines
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.upper() == line
            and len(line.split()) >= 2
            and nxt
            and nxt.upper() == nxt
            and len(nxt.split()) == 1
            and len(nxt) >= 4
            and not re.search(r"\d", nxt)
            and not _npwp_line_is_noise(nxt)
        ):
            out.append(f"{line} {nxt}")
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def extract_npwp_holder_names(ocr_text: str) -> list[str]:
    """Nama pemilik dari kartu NPWP (baris kapital di bawah/atas nomor 16 digit)."""
    lines = _merge_npwp_adjacent_name_lines(
        [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
    )
    seen: set[str] = set()
    out: list[str] = []

    def _add(raw: str) -> None:
        cand = _trim_name_punctuation(raw)
        if not cand or _npwp_line_is_noise(cand) or not _npwp_name_plausible(cand):
            return
        words = cand.split()
        if cand.upper() == cand:
            if len(words) < 2:
                return
        elif not _looks_like_person_name(cand):
            return
        key = _normalize(cand)
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    for i, line in enumerate(lines):
        if len(re.sub(r"\D", "", line)) != 16:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(lines) and not _npwp_line_is_noise(lines[j]):
                _add(lines[j])

    # NPWP lama: «NAMA DEPAN» lalu NIK 16 digit lalu «NAMA BELAKANG» di baris terpisah.
    for i, line in enumerate(lines):
        if _npwp_line_is_noise(line) or line.upper() != line or len(line.split()) < 2:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            cand = lines[j]
            if len(re.sub(r"\D", "", cand)) == 16:
                continue
            if (
                cand.upper() == cand
                and len(cand.split()) == 1
                and len(cand) >= 4
                and not re.search(r"\d", cand)
                and not _npwp_line_is_noise(cand)
            ):
                _add(f"{line} {cand}")
                break

    for line in lines:
        if _npwp_line_is_noise(line):
            continue
        if line.upper() == line and len(line.split()) >= 2:
            _add(line)
        elif (
            line.upper() == line
            and len(line.split()) == 1
            and len(line) >= 4
            and not re.search(r"\d", line)
        ):
            _add(line)

    for m in _CAPS_NAME_RUN_RE.finditer(ocr_text or ""):
        _add(m.group(1))

    if not out and _NPWP_NUMBER_IN_TEXT_RE.search(ocr_text or ""):
        for seg in person_like_segments(ocr_text):
            if len(seg.split()) >= 2:
                _add(seg)

    return out


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


_CV_LINE_NOISE = frozenset(
    {
        "tentang saya",
        "about me",
        "profil",
        "ringkasan profil",
        "summary",
        "curriculum vitae",
        "cv",
        "daftar riwayat hidup",
        "data pribadi",
        "personal data",
        "kontak",
        "contact",
        "fresh graduate",
        "fresh graduete",
        "fresh graduation",
        "lulusan baru",
        "pendidikan",
        "education",
        "keahlian",
        "skills",
        "pengalaman",
        "pengalaman kerja",
        "organisasi",
        "bahasa",
        "hobi",
        "media sosial",
        "referensi",
        "pelatihan",
        "sertifikasi",
    }
)

_CV_JUNK_CHARS_RE = re.compile(r"[★☆⭐◇◆●○◎✓✗☑☒]")

_CV_SECTION_HEADER_RE = re.compile(
    r"^(pendidikan|keahlian|pengalaman|organisasi|bahasa|hobi|media\s+sosial|"
    r"education|experience|skills|personal\s+data|riwayat\s+pendidikan|"
    r"riwayat\s+pekerjaan|work\s+experience|no\s+data)$",
    re.I,
)

_CV_JOB_NOISE_RE = re.compile(
    r"\b(fresh\s+grad\w*|lulusan\s+baru|mahasiswa|student|internship|magang|undergraduate)\b",
    re.I,
)

_CV_NAME_LABELS = frozenset({"nama", "name", "nama lengkap", "full name"})


def _latin_letter_count(s: str) -> int:
    return sum(1 for c in s if c.isascii() and c.isalpha())


def _cv_line_has_latin_name(s: str) -> bool:
    return _latin_letter_count(s) >= 3 and bool(re.search(r"[A-Za-z]{3,}", s))


def _looks_like_cv_name_line(seg: str, *, expected_name: str = "") -> bool:
    s = (seg or "").strip()
    s = re.sub(r"^[-•*|]+\s*", "", s).strip()
    if not s:
        return False
    if _CV_JUNK_CHARS_RE.search(s):
        return False
    if not _cv_line_has_latin_name(s):
        return False
    low = _normalize(s)
    if low in _CV_LINE_NOISE:
        return False
    if _CV_SECTION_HEADER_RE.match(low):
        return False
    if low in {"status", "email", "telepon", "phone", "alamat", "address", "kontak", "contact"}:
        return False
    if _CV_JOB_NOISE_RE.search(s):
        return False
    if "@" in s or re.search(r"\d{5,}", s):
        return False
    if re.search(r"\b(jl\.?|jalan|street|grand\s+mutiara|kec\.|kel\.|kab\.)\b", low):
        return False

    exp_n = _normalize(expected_name)
    words = s.split()

    # Nama satu kata di header CV (mis. ANNISA) — umum di template Canva/design
    if len(words) == 1 and re.fullmatch(r"[A-Za-z'.-]+", s):
        if exp_n:
            return float(fuzz.WRatio(low, exp_n)) >= 65.0
        if s.upper() == s and 4 <= len(s) <= 30:
            return True

    if not _looks_like_person_name(s):
        return False
    return True


def extract_cv_holder_name(
    ocr_text: str,
    *,
    expected_name: str = "",
) -> tuple[str | None, str]:
    """
    Ekstrak nama dari teks CV (bukan pola label KTP).
    `expected_name` opsional — dipakai untuk memilih kandidat terbaik bila ada beberapa baris mirip nama.
    """
    segments = _split_segments(ocr_text)

    for i, seg in enumerate(segments):
        raw = seg.strip()
        low = _normalize(raw)
        if ":" in raw:
            label_part, _, after = raw.partition(":")
            if _normalize(label_part) in _CV_NAME_LABELS:
                after = after.strip()
                if _looks_like_cv_name_line(after, expected_name=expected_name):
                    return after, "cv_nama_colon"
        if low in _CV_NAME_LABELS and i + 1 < len(segments):
            nxt = segments[i + 1].strip()
            if _looks_like_cv_name_line(nxt, expected_name=expected_name):
                return nxt, "cv_after_nama_label"

    exp_n = _normalize(expected_name)
    candidates: list[tuple[float, str, str]] = []

    for line in re.split(r"[\n|]+", ocr_text or ""):
        line = re.sub(r"^[-•*|]+\s*", "", (line or "").strip()).strip()
        if not _looks_like_cv_name_line(line, expected_name=expected_name):
            continue
        score = 0.0
        words = line.split()
        line_n = _normalize(line)
        if exp_n:
            wr = float(fuzz.WRatio(line_n, exp_n))
            score += wr * 2.0
            if wr >= 90.0:
                score += 25.0
        if line.upper() == line and re.search(r"[A-Z]{2,}", line):
            score += 10.0
        if 2 <= len(words) <= 5:
            score += 6.0
        if len(words) == 1 and 4 <= len(line) <= 30:
            score += 5.0
        candidates.append((score, line, "cv_name_line"))

    for seg in person_like_segments(ocr_text):
        if not _looks_like_cv_name_line(seg, expected_name=expected_name):
            continue
        if any(seg == c[1] for c in candidates):
            continue
        score = 0.0
        if seg.upper() == seg and re.search(r"[A-Z]{2,}", seg):
            score += 12.0
        if exp_n:
            score += float(fuzz.WRatio(_normalize(seg), exp_n))
        candidates.append((score, seg, "cv_person_segment"))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        best = candidates[0]
        return best[1], best[2]

    if exp_n:
        text_n = _normalize(ocr_text)
        if float(fuzz.partial_ratio(exp_n, text_n)) >= 65.0:
            for line in re.split(r"[\n|]+", ocr_text or ""):
                line = re.sub(r"^[-•*|]+\s*", "", (line or "").strip()).strip()
                if not line or len(line) < 4:
                    continue
                if not _cv_line_has_latin_name(line):
                    continue
                if _CV_JUNK_CHARS_RE.search(line):
                    continue
                if float(fuzz.WRatio(_normalize(line), exp_n)) >= 65.0:
                    return line, "cv_fuzzy_line_match"

    return None, "failed"


def extract_holder_name_candidate(ocr_text: str, document_profile_id: str) -> tuple[str | None, str]:
    """
    Mengembalikan (teks nama perkiraan atau None, metode / alasan).
    `document_profile_id` kanonik: ktp, npwp, dll.
    """
    profile = (document_profile_id or "").strip().casefold()
    if profile == "cv":
        return extract_cv_holder_name(ocr_text)

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

    if profile in {"npwp", "pemadanan_npwp"}:
        npwp_names = extract_npwp_holder_names(ocr_text)
        if npwp_names:
            return npwp_names[0], "npwp_name_line"

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
            if not _is_probably_field_label(nxt) and (
                _looks_like_person_name(nxt)
                or (
                    nxt.upper() == nxt
                    and 4 <= len(nxt) <= 40
                    and nxt.replace(" ", "").isalpha()
                )
            ):
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
