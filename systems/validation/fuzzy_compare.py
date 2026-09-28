"""Perbandingan teks hasil OCR dengan nama referensi dan keyword dokumen (RapidFuzz)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from rapidfuzz import fuzz

from systems.validation.bank_rekening import normalize_expected_bank, validate_rekening_bank, BANK_ACCOUNT_PROFILES
from systems.validation.document_profiles import (
    CANONICAL_KEYWORDS,
    any_keyword_groups_for_profile,
    excluded_keywords_for_profile,
    profile_label,
    resolve_keywords,
    skip_identity_validation,
)
from systems.validation.ktp_marital import apply_ktp_helper_marital_gate
from systems.validation.skck_validity import apply_skck_expiry_gate
from systems.validation.bpjs_kesanggupan_stamp import apply_bpjs_kesanggupan_stamp_gate
from systems.validation.jkn_status import apply_jkn_status_extract
from systems.validation.keterangan_kesehatan import apply_keterangan_kesehatan_gates
from systems.validation.user_errors import attach_user_errors
from systems.validation.name_extraction import (
    extract_holder_name_candidate,
    extract_iuran_participant_names,
    extract_jkn_participant_names,
    jkn_masked_name_matches,
    extract_kk_family_card_number,
    extract_kk_member_names,
    extract_kk_member_niks_ordered,
    extract_kk_name_by_nik,
    extract_npwp_holder_names,
    extract_vaksinasi_1_names,
    iuran_modal_only_state,
    person_like_segments,
    sliding_name_windows,
    extract_keterangan_kesehatan_names,
    vaksinasi_1_first_dose_signal,
    vaksinasi_1_wrong_dose_primary,
)

_NIK16_SPACED_RE = re.compile(r"\d{4}[\s.\-]?\d{4}[\s.\-]?\d{4}[\s.\-]?\d{4}")
_NIK_ACCOUNT_SUFFIX_RE = re.compile(r"(\d{16})(?:_(\d{10,13}))?$", re.I)


def parse_nik16_from_filename(filename: str) -> str | None:
    """Ambil NIK 16 digit dari pola `{tipe}_{Nama}_{NIK16}` atau `..._{NIK16}_{NoRek}.ext`."""
    stem = (filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    m = _NIK_ACCOUNT_SUFFIX_RE.search(stem)
    return m.group(1) if m else None


def _rekening_identity_and_account_pass(
    *,
    has_name_ref: bool,
    has_account_ref: bool,
    identity_pass: bool | None,
    account_pass: bool | None,
) -> bool:
    """Nama lengkap DAN nomor rekening — keduanya harus lolos bila referensinya diisi."""
    if has_name_ref and has_account_ref:
        return identity_pass is True and account_pass is True
    if has_name_ref:
        return identity_pass is True
    if has_account_ref:
        return account_pass is True
    return False


def extract_nik16_from_ocr(ocr_text: str) -> list[str]:
    """Kumpulkan NIK/NPWP 16 digit dari teks OCR (spasi/titik diperbolehkan)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _NIK16_SPACED_RE.finditer(ocr_text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) != 16 or digits in seen:
            continue
        seen.add(digits)
        out.append(digits)
    return out


_NPWP_ID_RUN_RE = re.compile(r"\d[\d\s.\-]{12,22}\d")


def extract_npwp_ids_from_ocr(ocr_text: str) -> list[str]:
    """Nomor NPWP 15–16 digit (kartu baru sering 4-4-4-4; OCR kadang hilang 1 digit depan)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _NPWP_ID_RUN_RE.finditer(ocr_text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) not in {15, 16}:
            continue
        if digits in seen:
            continue
        seen.add(digits)
        out.append(digits)
    return out


_KK_NIK_OCR_TRANS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "i": "1",
        "L": "1",
        "l": "1",
        "Z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "B": "8",
        "C": "0",
        "c": "0",
    }
)


def _hamming16(a: str, b: str) -> int:
    if len(a) != 16 or len(b) != 16:
        return 99
    return sum(x != y for x, y in zip(a, b))


def _kk_nik_candidates(ocr_text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(n: str) -> None:
        if len(n) == 16 and n not in seen:
            seen.add(n)
            out.append(n)

    for n in extract_kk_member_niks_ordered(ocr_text):
        add(n)
    for n in extract_nik16_from_ocr(ocr_text):
        add(n)
    mapped = (ocr_text or "").translate(_KK_NIK_OCR_TRANS)
    for m in re.finditer(r"(?<![0-9])([0-9]{16})(?![0-9])", mapped):
        add(m.group(1))
    for m in re.finditer(r"(?<![A-Za-z0-9])([A-Za-z0-9]{16})(?![A-Za-z0-9])", ocr_text or ""):
        add(re.sub(r"\D", "", m.group(1).translate(_KK_NIK_OCR_TRANS)))
    return out


def kk_nik_matches_expected(ocr_text: str, expected_nik: str) -> str:
    """NIK anggota KK: exact, huruf OCR (C/O→0), atau beda 1 digit."""
    exp = re.sub(r"\D", "", expected_nik or "")
    if len(exp) != 16:
        return ""
    family_no = extract_kk_family_card_number(ocr_text)
    cands = [n for n in _kk_nik_candidates(ocr_text) if n != family_no]
    if exp in cands:
        return exp
    for n in cands:
        if _hamming16(n, exp) <= 1:
            return n
    compact = re.sub(r"\D", "", (ocr_text or "").translate(_KK_NIK_OCR_TRANS))
    if exp in compact:
        return exp
    return ""


def npwp_id_matches_expected(ocr_text: str, expected_nik: str) -> str:
    """Cocokkan NIK/NPWP 16 digit referensi dengan nomor di OCR (termasuk 15 digit)."""
    exp = re.sub(r"\D", "", expected_nik or "")
    if len(exp) != 16:
        return ""
    compact = re.sub(r"\D", "", ocr_text or "")
    if exp in compact:
        return exp
    if exp[1:] in compact:
        return exp
    if exp[:15] in compact:
        return exp
    for found in extract_npwp_ids_from_ocr(ocr_text):
        if found == exp or found == exp[1:] or found == exp[:15]:
            return exp
    return ""


def jkn_nik_matches_expected(ocr_text: str, expected_nik: str) -> str:
    """NIK 16 digit di layar JKN / chat Pandawa, termasuk potongan OCR ≥12 digit."""
    exp = re.sub(r"\D", "", expected_nik or "")
    if len(exp) != 16:
        return ""
    compact = re.sub(r"\D", "", ocr_text or "")
    if exp in compact or exp[1:] in compact or exp[:15] in compact:
        return exp
    for m in re.finditer(r"\d{11,16}", ocr_text or ""):
        run = m.group(0)
        if run in exp:
            return exp
    return ""


_OCR_DATE_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b")


def jkn_nik_birthdate_in_ocr(ocr_text: str, expected_nik: str) -> str:
    """Tanggal lahir dari NIK 16 digit (DD+40 untuk perempuan) muncul di layar JKN."""
    exp = re.sub(r"\D", "", expected_nik or "")
    if len(exp) != 16:
        return ""
    day = int(exp[6:8])
    month = int(exp[8:10])
    year2 = int(exp[10:12])
    if day > 40:
        day -= 40
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return ""
    years = {2000 + year2, 1900 + year2}
    for m in _OCR_DATE_RE.finditer(ocr_text or ""):
        d_raw, mo_raw, y_raw = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = y_raw
        if y_raw < 100:
            year = 2000 + y_raw if y_raw <= 30 else 1900 + y_raw
        if d_raw == day and mo_raw == month and year in years:
            return f"{day:02d}-{month:02d}-{year}"
    return ""

try:
    from systems.ocr.mistral_annotation import (
        document_type_profile_from_annotation,
        holder_name_from_annotation,
    )
except ImportError:
    document_type_profile_from_annotation = None  # type: ignore[misc, assignment]
    holder_name_from_annotation = None  # type: ignore[misc, assignment]


def _normalize(s: str) -> str:
    s = (s or "").strip().casefold()
    return re.sub(r"\s+", " ", s)


def _normalize_name(s: str) -> str:
    """Normalisasi untuk perbandingan nama: spasi + tanda hubung diperlakukan sama."""
    s = _normalize(s)
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[:;|]+", " ", s)
    s = re.sub(r"\b(nn|ny|tn|nt|sdr|sdri)\.?\b", " ", s)
    s = re.sub(r"[.…]{2,}$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def _normalize_keyword(s: str) -> str:
    """Sama seperti teks penuh; slash dipasangkan jadi spasi agar 'rt/rw' ~ 'rt rw'."""
    s = (s or "").strip().casefold()
    s = s.replace("/", " ")
    return re.sub(r"\s+", " ", s)


# Alias OCR KTP: label panjang sering rusak, singkatan/frasa pendek masih muncul.
_KTP_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "kewarganegaraan": ("wni", "wna", "seumur hidup", "seumurhidup"),
    "status perkawinan": (
        "belum kawin",
        "belumkawin",
        "belumka",
        "belum ka",
        "belum menikah",
        "kawin",
        "cerai",
    ),
    "agama": ("islam", "kristen", "katolik", "hindu", "buddha", "konghucu"),
    "nama": ("lahir",),
}

_NPWP_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "npwp": ("npvp", "pwp", "np wp"),
    "pajak": ("djp", "perpajakan", "kpp"),
    "kantor pelayanan": ("kpp pratama", "pelayanan pajak", "tanggal terdaftar"),
    "pratama": ("kpp", "tanggal terdaftar"),
}


def _ktp_keyword_alias_scores(keyword: str, ocr_normalized: str) -> dict[str, float]:
    """Naikkan skor keyword bila alias KTP/NPWP umum terbaca di OCR."""
    kw = _normalize_keyword(keyword)
    aliases = _KTP_KEYWORD_ALIASES.get(kw) or _NPWP_KEYWORD_ALIASES.get(kw)
    if not aliases:
        return {}
    compact = re.sub(r"\s+", "", ocr_normalized)
    for alias in aliases:
        alias_n = _normalize_keyword(alias)
        alias_compact = alias_n.replace(" ", "")
        if alias_compact and alias_compact in compact:
            return {
                "partial_ratio": 100.0,
                "token_sort_ratio": 100.0,
                "token_set_ratio": 100.0,
                "wratio": 100.0,
            }
        if alias_n and re.search(rf"\b{re.escape(alias_n)}\b", ocr_normalized):
            return {
                "partial_ratio": 100.0,
                "token_sort_ratio": 100.0,
                "token_set_ratio": 100.0,
                "wratio": 100.0,
            }
    return {}


def _ktp_nik_in_ocr(ocr_n: str) -> bool:
    return bool(re.search(r"\b\d{16}\b", ocr_n or ""))


def _ktp_field_signal_count(ocr_n: str) -> int:
    """Penanda sekunder KTP (bukan keyword panjang) — tahan OCR buruk."""
    compact = re.sub(r"\s+", "", ocr_n or "")
    signals = 0
    if "wni" in compact or "wna" in compact:
        signals += 1
    if "seumurhidup" in compact or re.search(r"\bseumur\s+hidup\b", ocr_n or ""):
        signals += 1
    if re.search(r"\blahir\b", ocr_n or ""):
        signals += 1
    if re.search(r"\b(?:ke)?camatan\b", ocr_n or "") or "ecamatan" in compact:
        signals += 1
    if re.search(r"\b(?:kel|desa)\b", ocr_n or "") or "el/desa" in ocr_n or "keldesa" in compact:
        signals += 1
    if re.search(r"\b(?:rt|rw)\b", ocr_n or ""):
        signals += 1
    if "belumka" in compact or "belumkawin" in compact or re.search(r"\bbelum\s+kawin\b", ocr_n or ""):
        signals += 1
    return signals


def _bpjs_field_signal_count(ocr_n: str) -> int:
    """Penanda sekunder kartu KIS — tahan OCR miring/rusak."""
    compact = re.sub(r"\s+", "", ocr_n or "")
    blob = ocr_n or ""
    signals = 0
    if re.search(r"\bnama\b", blob) or "nama" in compact:
        signals += 1
    if re.search(r"\bnik\b", blob) or "nik" in compact:
        signals += 1
    if "faskes" in compact:
        signals += 1
    if "nomorkartu" in compact or re.search(r"\bnomor\s+kartu\b", blob):
        signals += 1
    if "alamat" in compact:
        signals += 1
    if "kartuindonesiasehat" in compact or re.search(r"\bkartu\s+indonesia\b", blob):
        signals += 1
    return signals


@dataclass(frozen=True)
class NameFuzzyResult:
    ocr_normalized: str
    expected_normalized: str
    scores: dict[str, float]
    best_score: float
    matched: bool
    identity: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ocr_normalized": self.ocr_normalized,
            "expected_normalized": self.expected_normalized,
            "scores": self.scores,
            "best_score": self.best_score,
            "matched": self.matched,
        }
        if self.identity is not None:
            out["identity"] = self.identity
        return out


def compare_ocr_name_to_expected(
    ocr_text: str,
    expected_name: str,
    *,
    threshold: float | None = 85.0,
) -> NameFuzzyResult:
    """
    Bandingkan teks OCR vs nama referensi dengan aturan identitas (cakupan kata),
    bukan skor terbaik dari partial_ratio/WRatio.
    Jika OCR panjang, pilih segmen nama orang dengan skor identitas tertinggi.
    """
    a = _normalize(ocr_text)
    b = _normalize(expected_name)
    empty_scores = {
        "ratio": 0.0,
        "partial_ratio": 0.0,
        "token_sort_ratio": 0.0,
        "token_set_ratio": 0.0,
        "wratio": 0.0,
        "identity_combined": 0.0,
        "token_coverage": 0.0,
    }
    if not a or not b:
        return NameFuzzyResult(
            ocr_normalized=a,
            expected_normalized=b,
            scores=empty_scores,
            best_score=0.0,
            matched=False,
            identity=_empty_identity_scores(a, b),
        )

    candidates: list[str] = [ocr_text]
    long_blob = len(_name_identity_tokens(a)) > 6 or "\n" in (ocr_text or "")
    if long_blob:
        candidates.extend(person_like_segments(ocr_text))
        candidates.extend(sliding_name_windows(ocr_text, expected_name))

    best_identity: dict[str, Any] | None = None
    best_score = -1.0
    seen: set[str] = set()
    for cand in candidates:
        key = _normalize_name(cand)
        if not key or key in seen:
            continue
        seen.add(key)
        ident = compare_extracted_identity_scores(cand, expected_name)
        sc = float(ident["identity_combined_score"])
        if sc > best_score:
            best_score = sc
            best_identity = ident

    ident = best_identity or _empty_identity_scores(a, b)
    scores = {
        "ratio": float(fuzz.ratio(a, b)),
        "partial_ratio": float(fuzz.partial_ratio(a, b)),
        "token_sort_ratio": float(fuzz.token_sort_ratio(a, b)),
        "token_set_ratio": float(fuzz.token_set_ratio(a, b)),
        "wratio": float(fuzz.WRatio(a, b)),
        "identity_combined": float(ident.get("identity_combined_score") or 0.0),
        "token_coverage": float((ident.get("scores") or {}).get("token_coverage") or 0.0),
    }
    combined = float(ident.get("identity_combined_score") or 0.0)
    if threshold is None:
        matched = False
    else:
        matched = combined >= float(threshold) and bool(ident.get("token_gate_pass"))
    return NameFuzzyResult(
        ocr_normalized=a,
        expected_normalized=b,
        scores=scores,
        best_score=combined,
        matched=matched,
        identity=ident,
    )


_NAME_SKIP_TOKENS = frozenset(
    {
        "bin",
        "binti",
        "bte",
        "bt",
        "ibn",
        "als",
        "alias",
        "van",
        "de",
        "der",
        "den",
        "el",
        "al",
        "nn",
        "ny",
        "tn",
        "nt",
        "sdr",
        "sdri",
        "saudara",
        "saudari",
        "bahwa",
    }
)
_TOKEN_MATCH_MIN = 80.0
_TOKEN_MATCH_MIN_SHORT = 90.0
_IDENTITY_TOKEN_GATE_CAP = 45.0


def _name_identity_tokens(s: str) -> list[str]:
    return [t for t in _normalize_name(s).split() if t and t not in _NAME_SKIP_TOKENS]


_OCR_NAME_DIGIT_TRANS = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "2": "z",
        "3": "e",
        "4": "a",
        "5": "s",
        "6": "g",
        "8": "b",
    }
)


def _ocr_fold_name_token(s: str) -> str:
    return (s or "").translate(_OCR_NAME_DIGIT_TRANS)


def _token_pair_score(a: str, b: str) -> float:
    raw = float(fuzz.ratio(a, b))
    folded = float(fuzz.ratio(_ocr_fold_name_token(a), b))
    return max(raw, folded)


def _token_pair_matches(score: float, a: str, b: str) -> bool:
    need = _TOKEN_MATCH_MIN_SHORT if min(len(a), len(b)) <= 3 else _TOKEN_MATCH_MIN
    return score >= need


def _greedy_token_matches(extracted_tokens: list[str], expected_tokens: list[str]) -> list[tuple[int, int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for i, ta in enumerate(extracted_tokens):
        for j, tb in enumerate(expected_tokens):
            pairs.append((_token_pair_score(ta, tb), i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for score, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, score))
    return matched


def _required_expected_token_matches(n_expected: int) -> int:
    """Nama 2 kata harus cocok semua token — cegah lolos hanya karena satu kata berbagi."""
    if n_expected <= 1:
        return 1
    if n_expected == 2:
        return 2
    return max(2, n_expected - 1)


def _empty_identity_scores(extracted_normalized: str, expected_normalized: str) -> dict[str, Any]:
    return {
        "extracted_normalized": extracted_normalized,
        "expected_normalized": expected_normalized,
        "scores": {
            "token_sort_ratio": 0.0,
            "wratio": 0.0,
            "partial_ratio": 0.0,
            "token_coverage": 0.0,
        },
        "identity_combined_score": 0.0,
        "identity_combined_ratio": 0.0,
        "matched_token_count": 0,
        "required_token_matches": 0,
        "token_coverage_expected": 0.0,
        "token_coverage_extracted": 0.0,
        "token_gate_pass": False,
    }


def compare_extracted_identity_scores(extracted: str, expected_name: str) -> dict[str, Any]:
    """
    Identitas: bandingkan nama diekstrak vs nama referensi.

    Skor gabungan memakai token_sort_ratio (urutan kata bebas) dan cakupan token 1-1.
    partial_ratio/WRatio disimpan untuk diagnostik saja — keduanya terlalu longgar
    bila dua orang berbagi satu kata (mis. Anggun Ramadhani vs Rahmalia Anggun Agustin).
    """
    a = _normalize_name(extracted)
    b = _normalize_name(expected_name)
    if not a or not b:
        return _empty_identity_scores(a, b)

    ts = float(fuzz.token_sort_ratio(a, b))
    wr = float(fuzz.WRatio(a, b))
    pr = float(fuzz.partial_ratio(a, b))

    compact_a = re.sub(r"[^a-z]", "", _ocr_fold_name_token(a))
    compact_b = re.sub(r"[^a-z]", "", b)
    glued_same = len(compact_b) >= 8 and compact_a == compact_b
    glued_prefix = (
        len(compact_a) >= 10
        and len(compact_b) >= 10
        and compact_b.startswith(compact_a)
        and (len(compact_a) / len(compact_b)) >= 0.68
    )
    glued_ocr_near = (
        len(compact_a) >= 8
        and len(compact_b) >= 8
        and abs(len(compact_a) - len(compact_b)) <= 3
        and float(fuzz.ratio(compact_a, compact_b)) >= 82.0
    )

    ext_tokens = _name_identity_tokens(a)
    exp_tokens = _name_identity_tokens(b)
    if not ext_tokens or not exp_tokens:
        return _empty_identity_scores(a, b)

    required = _required_expected_token_matches(len(exp_tokens))
    pairs = _greedy_token_matches(ext_tokens, exp_tokens)
    n_match = sum(
        1
        for i, j, sc in pairs
        if _token_pair_matches(sc, ext_tokens[i], exp_tokens[j])
    )
    cov_exp = n_match / len(exp_tokens)
    cov_ext = n_match / len(ext_tokens)
    if cov_exp > 0.0 and cov_ext > 0.0:
        cov_score = 100.0 * (2.0 * cov_exp * cov_ext / (cov_exp + cov_ext))
    else:
        cov_score = 0.0

    token_gate_pass = n_match >= required
    if glued_same or glued_prefix or glued_ocr_near:
        token_gate_pass = True
        ts = max(ts, 96.0)
        cov_score = max(cov_score, 96.0)
        n_match = max(n_match, required)
        cov_exp = 1.0
        cov_ext = 1.0
    elif (
        not token_gate_pass
        and len(exp_tokens) >= 2
        and len(ext_tokens) >= 2
        and exp_tokens[-1] == ext_tokens[-1]
        and len(exp_tokens[-1]) >= 6
    ):
        first_sc = max(_token_pair_score(t, exp_tokens[0]) for t in ext_tokens[:-1])
        if first_sc >= 55.0:
            token_gate_pass = True
            n_match = max(n_match, required)
            ts = max(ts, 80.0)
            cov_score = max(cov_score, 80.0)
    if (
        not token_gate_pass
        and len(exp_tokens) >= 2
        and len(ext_tokens) >= 2
        and len(exp_tokens[-1]) >= 5
    ):
        last_sc = _token_pair_score(ext_tokens[-1], exp_tokens[-1])
        first_sc = max(_token_pair_score(t, exp_tokens[0]) for t in ext_tokens[:-1])
        if last_sc >= 65.0 and first_sc >= 55.0:
            token_gate_pass = True
            n_match = max(n_match, required)
            ts = max(ts, 80.0)
            cov_score = max(cov_score, 80.0)
    if (
        not token_gate_pass
        and len(exp_tokens) >= 3
        and len(ext_tokens) >= 2
    ):
        sc0 = _token_pair_score(ext_tokens[0], exp_tokens[0])
        sc1 = _token_pair_score(ext_tokens[1], exp_tokens[1])
        if sc0 >= 80.0 and sc1 >= 80.0:
            token_gate_pass = True
            n_match = max(n_match, required)
            ts = max(ts, 80.0)
            cov_score = max(cov_score, 80.0)
    combined = (ts + cov_score) / 2.0
    if not token_gate_pass:
        combined = min(combined, _IDENTITY_TOKEN_GATE_CAP)
    else:
        combined = max(combined, 70.0)

    return {
        "extracted_normalized": a,
        "expected_normalized": b,
        "scores": {
            "token_sort_ratio": ts,
            "wratio": wr,
            "partial_ratio": pr,
            "token_coverage": round(cov_score, 4),
        },
        "identity_combined_score": round(combined, 4),
        "identity_combined_ratio": round(combined / 100.0, 6),
        "matched_token_count": n_match,
        "required_token_matches": required,
        "token_coverage_expected": round(cov_exp, 4),
        "token_coverage_extracted": round(cov_ext, 4),
        "token_gate_pass": token_gate_pass,
    }


def _ocr_search_corpora(ocr_normalized: str) -> list[str]:
    """Korpus untuk fuzzy keyword: teks penuh, segmen baris, jendela 2–3 segmen, tanpa spasi."""
    text = (ocr_normalized or "").strip()
    if not text:
        return []

    segments: list[str] = []
    for chunk in re.split(r"[\n|•]+", text):
        part = chunk.strip()
        if part:
            segments.append(part)
    if len(segments) <= 1:
        segments = [s.strip() for s in re.split(r"\s{2,}", text) if s.strip()] or [text]

    corpora: list[str] = [text]
    corpora.extend(segments)
    for i in range(len(segments) - 1):
        corpora.append(f"{segments[i]} {segments[i + 1]}")
    for i in range(len(segments) - 2):
        corpora.append(f"{segments[i]} {segments[i + 1]} {segments[i + 2]}")

    compact = re.sub(r"\s+", "", text)
    if compact:
        corpora.append(compact)

    seen: set[str] = set()
    out: list[str] = []
    for c in corpora:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fuzzy_scores_for_pair(a: str, b: str) -> dict[str, float]:
    return {
        "partial_ratio": float(fuzz.partial_ratio(a, b)),
        "token_sort_ratio": float(fuzz.token_sort_ratio(a, b)),
        "token_set_ratio": float(fuzz.token_set_ratio(a, b)),
        "wratio": float(fuzz.WRatio(a, b)),
    }


_NPWP_OCR_MARKERS = ("npwp", "npvp", "pvp", "nvp", "np vp", "pv p", "djp")


def _ocr_has_npwp_marker(ocr_n: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", (ocr_n or "").casefold())
    if any(m.replace(" ", "") in compact for m in _NPWP_OCR_MARKERS):
        return True
    return bool(re.search(r"\bnp\s*vp\b", ocr_n or "", re.I))


def _npwp_tax_id_in_ocr(ocr_n: str) -> bool:
    """Nomor NPWP 15–16 digit (sering terbaca dengan spasi di OCR)."""
    digits = re.sub(r"\D", "", ocr_n or "")
    if re.search(r"\d{15,16}", digits):
        return True
    return bool(extract_npwp_ids_from_ocr(ocr_n))


def fuzzy_keyword_against_ocr(keyword: str, ocr_normalized: str) -> dict[str, Any]:
    """
    Skor fuzzy keyword terhadap teks OCR: ambil terbaik dari teks penuh, segmen baris,
    jendela segmen berdekatan, dan teks tanpa spasi (untuk label menempel seperti provinsijawabarat).
    """
    kw = _normalize_keyword(keyword)
    if not kw:
        return {
            "keyword_raw": keyword,
            "keyword_normalized": kw,
            "scores": {},
            "best_score": 0.0,
            "score_ratio": 0.0,
            "skipped": True,
        }
    if not ocr_normalized:
        return {
            "keyword_raw": keyword,
            "keyword_normalized": kw,
            "scores": {
                "partial_ratio": 0.0,
                "token_sort_ratio": 0.0,
                "token_set_ratio": 0.0,
                "wratio": 0.0,
            },
            "best_score": 0.0,
            "score_ratio": 0.0,
            "skipped": False,
        }

    corpora = _ocr_search_corpora(ocr_normalized)
    merged: dict[str, float] = {
        "partial_ratio": 0.0,
        "token_sort_ratio": 0.0,
        "token_set_ratio": 0.0,
        "wratio": 0.0,
    }
    for corpus in corpora:
        for metric, value in _fuzzy_scores_for_pair(kw, corpus).items():
            merged[metric] = max(merged[metric], value)

    for alias_scores in (_ktp_keyword_alias_scores(kw, ocr_normalized),):
        for metric, value in alias_scores.items():
            merged[metric] = max(merged[metric], value)

    # Label pendek (nik, npwp): cek juga kecocokan substring pada teks padat.
    if " " not in kw and len(kw) <= 12:
        compact_kw = kw.replace(" ", "")
        for corpus in corpora:
            compact_corpus = re.sub(r"\s+", "", corpus)
            if compact_kw and compact_kw in compact_corpus:
                merged["partial_ratio"] = max(merged["partial_ratio"], 100.0)
                merged["wratio"] = max(merged["wratio"], 100.0)
            if kw == "npwp":
                for alias in ("npvp", "pvp", "nvp"):
                    if alias in compact_corpus:
                        merged["partial_ratio"] = max(merged["partial_ratio"], 100.0)
                        merged["wratio"] = max(merged["wratio"], 100.0)

    best = max(merged.values())
    return {
        "keyword_raw": keyword,
        "keyword_normalized": kw,
        "scores": merged,
        "best_score": best,
        "score_ratio": round(best / 100.0, 6),
        "skipped": False,
    }


_EXTRACTION_METHOD_ID: dict[str, str] = {
    "nama_inline_colon": "teks setelah label 'Nama:' pada baris yang sama",
    "after_nama_segment": "segmen berikut setelah kata 'Nama'",
    "regex_nama": "pola teks di sekitar kata 'nama' dalam OCR",
    "best_person_like_vs_expected": "segmen yang paling mirip nama (heuristik, karena penanda 'Nama' tidak jelas)",
    "person_like_segment": "segmen mirip nama dengan skor identitas tertinggi vs referensi",
    "kk_table_row": "baris tabel anggota Kartu Keluarga (nama lengkap + NIK)",
    "kk_table_first_member": "anggota pertama pada tabel Kartu Keluarga",
    "jkn_caps_name": "nama peserta JKN (huruf kapital di kartu Info Peserta)",
    "jkn_participant_name": "nama peserta JKN pada kartu Mobile JKN",
    "bpjs_after_nama": "nama pada kartu KIS setelah label Nama",
    "bpjs_caps_name": "nama peserta pada kartu KIS (huruf kapital)",
    "bpjs_tk_caps_name": "nama peserta pada kartu BPJS Ketenagakerjaan",
    "bpjs_kesanggupan_name": "nama pada surat pernyataan kesanggupan BPJS",
    "iuran_participant_name": "nama peserta pada kartu Info Iuran",
    "iuran_modal_no_name_displayed": (
        "layar modal Info Iuran tanpa tagihan pribadi — nama tidak ditampilkan di layar"
    ),
    "vaksinasi_1_recipient_name": "nama penerima pada sertifikat/kartu vaksin dosis 1",
    "npwp_name_line": "baris nama kapital pada kartu NPWP (dekat nomor 16 digit)",
    "npwp_nik_identity": "NIK/NPWP 16 digit di OCR vs NIK referensi",
    "failed": "tidak berhasil menarik nama dari OCR",
    "skipped_no_expected_name": "nama referensi kosong — identitas tidak diuji",
    "skipped_profile_no_identity": "profil dokumen tidak memeriksa nama (mis. mutasi)",
}


def _validation_explanation(
    *,
    document_matched: bool,
    document_type_pass: bool,
    document_type_aggregate_pass_ratio: float,
    aggregate_min_pass_ratio: float,
    keyword_results: list[dict[str, Any]],
    want_identity: bool,
    identity_pass: bool | None,
    identity_min_score: float,
    identity: dict[str, Any] | None,
    name_extraction: dict[str, Any],
    expected_name_display: str,
    identity_skip_reason: str | None = None,
) -> dict[str, Any]:
    """Ringkasan manusiawi + kode untuk klien API."""
    req_pct = aggregate_min_pass_ratio * 100.0
    avg_pct = document_type_aggregate_pass_ratio * 100.0
    blockers: list[str] = []
    detail_lines: list[str] = []
    hints: list[str] = []

    kw_non_skipped = [k for k in keyword_results if not k.get("skipped")]
    n_kw = len(kw_non_skipped)

    if document_type_pass:
        dt_reason = (
            f"Profil jenis dokumen lolos: rata-rata skor keyword {avg_pct:.1f}% "
            f"mencapai syarat minimal {req_pct:.0f}% ({n_kw} keyword dihitung)."
        )
    else:
        blockers.append("DOCUMENT_TYPE")
        dt_reason = (
            f"Profil jenis dokumen tidak lolos: rata-rata skor keyword {avg_pct:.1f}% "
            f"di bawah syarat {req_pct:.0f}% ({n_kw} keyword dihitung)."
        )
        worst = sorted(
            (k for k in kw_non_skipped),
            key=lambda x: float(x.get("best_score") or 0.0),
        )[:5]
        if worst:
            low_bits = ", ".join(
                f"«{w.get('keyword_raw', '')}» ~{float(w.get('best_score') or 0):.0f}%"
                for w in worst
            )
            dt_reason += f" Keyword terendah: {low_bits}."
        hints.append(
            "Periksa apakah OCR teks profil (mis. 'REPUBLIK INDONESIA', 'KTP') terbaca utuh; "
            "coba preprocess atau mode OCR lain bila teks keyword putus-putus.",
        )
    detail_lines.append(dt_reason)

    id_reason = ""
    if identity_skip_reason:
        id_reason = identity_skip_reason
    elif identity and identity.get("npwp_nik_identity"):
        exp_nik_disp = str(identity.get("expected_nik") or "")
        got_nik = str(identity.get("ocr_nik_matched") or "")
        if identity_pass is True:
            id_reason = (
                f"Identitas lolos: NIK/NPWP 16 digit di dokumen cocok dengan referensi "
                f"«{got_nik or exp_nik_disp}» (validasi NPWP memakai NIK, bukan nama)."
            )
        else:
            blockers.append("IDENTITY")
            id_reason = (
                "Identitas tidak lolos: NIK/NPWP 16 digit referensi "
                f"«{exp_nik_disp}» tidak terbaca atau tidak cocok di OCR."
            )
            hints.append(
                "Pastikan nomor 16 digit di kartu NPWP (NIK) terbaca utuh di OCR; "
                "kartu baru memakai NIK sebagai nomor NPWP."
            )
    elif identity and (identity.get("kk_or_identity") or identity.get("kk_and_identity")):
        name_ok = identity.get("kk_name_pass") is True
        nik_ok = identity.get("kk_nik_pass") is True
        exp_nik_disp = str(identity.get("expected_nik") or "")
        if identity_pass is True:
            parts = []
            if name_ok:
                parts.append("nama anggota KK cocok")
            if nik_ok:
                parts.append(f"NIK «{exp_nik_disp}» cocok")
            id_reason = "Identitas lolos (OR): " + " dan ".join(parts) + "."
        else:
            blockers.append("IDENTITY")
            bits = []
            if not name_ok:
                bits.append("nama tidak cukup mirip dengan referensi")
            if not nik_ok:
                bits.append(
                    f"NIK «{exp_nik_disp}» tidak ditemukan di daftar anggota KK"
                    if exp_nik_disp
                    else "NIK 16 digit referensi kosong"
                )
            id_reason = "Identitas tidak lolos: " + " dan ".join(bits) + "."
            hints.append(
                "Kartu Keluarga lolos jika nama ATAU NIK anggota cocok (OR)."
            )
    elif not want_identity:
        id_reason = (
            "Identitas tidak diuji karena nama referensi kosong — "
            "`document_matched` hanya mengandalkan profil keyword."
        )
    elif identity_pass is True:
        id_reason = (
            f"Identitas lolos: skor gabungan nama ≥ ambang {identity_min_score:.0f} "
            "(token_sort_ratio + cakupan kata nama; setiap kata referensi harus terwakili, "
            "bukan hanya satu kata yang kebetulan sama)."
        )
    else:
        blockers.append("IDENTITY")
        method = str(name_extraction.get("method") or "")
        method_human = _EXTRACTION_METHOD_ID.get(method, method or "tidak diketahui")
        raw_cand = name_extraction.get("candidate_raw")
        if identity and identity.get("extraction_failed"):
            id_reason = (
                "Identitas tidak lolos: tidak menemukan cuplikan nama yang cukup meyakinkan dari OCR "
                f"(metode ekstraksi: {method_human}). "
                f"Nama referensi yang diharapkan: «{expected_name_display}»."
            )
            hints.append(
                "Pastikan baris 'Nama' di dokumen terbaca jelas di teks OCR; "
                "bandingkan cuplikan teks OCR dengan field nama di KTP/NPWP.",
            )
        elif not raw_cand:
            id_reason = (
                "Identitas tidak lolos: tidak ada kandidat nama terpilih dari OCR "
                f"({method_human}). Nama referensi: «{expected_name_display}»."
            )
            hints.append(
                "Isi nama referensi sama seperti di dokumen (urutan kata, tanpa gelar jika tidak ada di KTP).",
            )
        elif identity:
            isc = float(identity.get("identity_combined_score") or 0.0)
            scores = identity.get("scores") or {}
            ts = float(scores.get("token_sort_ratio") or 0.0)
            wr = float(scores.get("wratio") or 0.0)
            pr = float(scores.get("partial_ratio") or 0.0)
            tc = float(scores.get("token_coverage") or 0.0)
            n_match = int(identity.get("matched_token_count") or 0)
            n_need = int(identity.get("required_token_matches") or 0)
            ext = identity.get("extracted_normalized") or ""
            exp = identity.get("expected_normalized") or ""
            gate_note = ""
            if identity.get("token_gate_pass") is False:
                gate_note = (
                    f" Cakupan kata tidak cukup ({n_match} cocok, perlu ≥ {n_need}) — "
                    "nama yang hanya berbagi satu kata tidak dianggap orang yang sama."
                )
            id_reason = (
                "Identitas tidak lolos: nama yang dipakai dari OCR "
                f"«{raw_cand}» (dinormalisasi: «{ext}»; cara ambil: {method_human}) "
                f"dibandingkan dengan referensi «{expected_name_display}» (dinormalisasi: «{exp}»). "
                f"Skor gabungan {isc:.1f} di bawah ambang {identity_min_score:.0f}. "
                f"Rincian sub-skor — token_sort_ratio: {ts:.1f}, token_coverage: {tc:.1f}, "
                f"WRatio: {wr:.1f}, partial_ratio: {pr:.1f} (dua terakhir diagnostik)."
                f"{gate_note}"
            )
            if isc + 15 < identity_min_score:
                hints.append(
                    "Kemiripan nama rendah: cek salah baca OCR, nama pendek/alias, atau nama referensi yang tidak sama dengan di dokumen.",
                )
            elif ts < wr and ts < pr:
                hints.append(
                    "token_sort_ratio rendah — urutan atau penggalan kata antara OCR dan referensi mungkin berbeda; "
                    "samakan ejaan dan spasi.",
                )
        else:
            id_reason = "Identitas tidak lolos (detail skor tidak tersedia)."
    detail_lines.append(id_reason)

    if document_matched:
        summary_id = "Dokumen dianggap cocok: semua gate yang aktif telah lulus."
        blockers = []
    elif blockers == ["IDENTITY"] and document_type_pass:
        if identity and identity.get("npwp_nik_identity"):
            summary_id = (
                "Dokumen tidak cocok karena identitas: profil keyword/jenis dokumen lolos, "
                "tetapi NIK/NPWP 16 digit di OCR tidak cocok dengan referensi."
            )
        else:
            summary_id = (
                "Dokumen tidak cocok karena identitas: profil keyword/jenis dokumen lolos, "
                "tetapi nama dari OCR tidak cukup mirip dengan nama referensi."
            )
    elif blockers == ["DOCUMENT_TYPE"]:
        summary_id = (
            "Dokumen tidak cocok karena profil jenis dokumen: keyword tidak mencapai ambang agregat."
        )
    elif "DOCUMENT_TYPE" in blockers and "IDENTITY" in blockers:
        summary_id = (
            "Dokumen tidak cocok: profil jenis dokumen dan identitas keduanya tidak memenuhi syarat."
        )
    elif "IDENTITY" in blockers:
        summary_id = "Dokumen tidak cocok karena identitas tidak memenuhi syarat."
    else:
        summary_id = "Dokumen tidak cocok."

    return {
        "locale": "id",
        "document_matched": document_matched,
        "primary_blockers": blockers,
        "summary": summary_id,
        "detail_lines": detail_lines,
        "hints": hints,
        "gates": {
            "document_type": {
                "pass": document_type_pass,
                "aggregate_percent": round(avg_pct, 4),
                "required_percent": round(req_pct, 4),
            },
            "identity": {
                "evaluated": want_identity,
                "pass": identity_pass,
                "min_combined_score": identity_min_score,
            },
        },
    }


def _profile_aggregate_score(ocr_normalized: str, keywords: list[str]) -> tuple[float, list[dict[str, Any]]]:
    keyword_results: list[dict[str, Any]] = []
    ratios: list[float] = []
    for raw_kw in keywords:
        item = fuzzy_keyword_against_ocr(raw_kw, ocr_normalized)
        keyword_results.append(item)
        if item.get("skipped"):
            continue
        ratios.append(float(item["best_score"]) / 100.0)
    aggregate = sum(ratios) / len(ratios) if ratios else 0.0
    return aggregate, keyword_results


def _ocr_has_keyword_signal(ocr_normalized: str, keyword: str, *, min_ratio: float = 0.7) -> bool:
    item = fuzzy_keyword_against_ocr(keyword, ocr_normalized)
    if item.get("skipped"):
        return False
    return float(item["best_score"]) / 100.0 >= min_ratio


def _evaluate_any_keyword_groups(
    ocr_normalized: str,
    groups: list[list[str]],
    *,
    min_ratio: float,
) -> dict[str, Any]:
    """Tiap grup: minimal satu keyword harus mencapai min_ratio (logika OR)."""
    group_results: list[dict[str, Any]] = []
    keyword_results: list[dict[str, Any]] = []
    group_max_ratios: list[float] = []
    all_groups_pass = True

    for gi, group in enumerate(groups):
        best_ratio = 0.0
        best_kw: str | None = None
        members: list[dict[str, Any]] = []
        for raw_kw in group:
            item = fuzzy_keyword_against_ocr(raw_kw, ocr_normalized)
            item = {
                **item,
                "any_group_index": gi,
                "any_group_member": True,
            }
            members.append(item)
            keyword_results.append(item)
            ratio = 0.0 if item.get("skipped") else float(item["best_score"]) / 100.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_kw = raw_kw
        group_pass = best_ratio >= min_ratio
        if not group_pass:
            all_groups_pass = False
        group_max_ratios.append(best_ratio)
        group_results.append(
            {
                "group_index": gi,
                "keywords": list(group),
                "best_keyword": best_kw,
                "best_ratio": round(best_ratio, 6),
                "pass": group_pass,
                "members": members,
            }
        )

    aggregate = sum(group_max_ratios) / len(group_max_ratios) if group_max_ratios else 0.0
    return {
        "aggregate": aggregate,
        "keyword_results": keyword_results,
        "any_groups": group_results,
        "any_groups_pass": all_groups_pass,
    }


def _exclusion_keyword_score(item: dict[str, Any]) -> float:
    """Skor keyword terlarang: abaikan WRatio (terlalu longgar vs frasa yang berbagi 1 kata)."""
    scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
    return float(
        max(
            float(scores.get("token_sort_ratio") or 0.0),
            float(scores.get("token_set_ratio") or 0.0),
            float(scores.get("partial_ratio") or 0.0),
        )
    )


def _evaluate_profile_keyword_match(
    profile_id: str,
    ocr_normalized: str,
    keywords: list[str],
    *,
    exclusion_min_ratio: float = 0.7,
) -> dict[str, Any]:
    aggregate, keyword_results = _profile_aggregate_score(ocr_normalized, keywords)
    flat_ratios = [
        float(item["best_score"]) / 100.0
        for item in keyword_results
        if not item.get("skipped")
    ]

    any_groups = any_keyword_groups_for_profile(profile_id)
    any_group_eval: dict[str, Any] | None = None
    if any_groups:
        any_group_eval = _evaluate_any_keyword_groups(
            ocr_normalized,
            any_groups,
            min_ratio=exclusion_min_ratio,
        )
        keyword_results.extend(any_group_eval["keyword_results"])

    if flat_ratios and any_group_eval:
        flat_aggregate = sum(flat_ratios) / len(flat_ratios)
        group_aggregate = float(any_group_eval["aggregate"])
        aggregate = (flat_aggregate + group_aggregate) / 2.0
    elif any_group_eval and not flat_ratios:
        aggregate = float(any_group_eval["aggregate"])
    elif flat_ratios:
        aggregate = sum(flat_ratios) / len(flat_ratios)

    any_groups_pass = bool(any_group_eval["any_groups_pass"]) if any_group_eval else True
    excluded_results: list[dict[str, Any]] = []
    exclusion_violated = False
    for raw_ex in excluded_keywords_for_profile(profile_id):
        item = fuzzy_keyword_against_ocr(raw_ex, ocr_normalized)
        hit_ratio = 0.0 if item.get("skipped") else _exclusion_keyword_score(item) / 100.0
        hit = hit_ratio >= exclusion_min_ratio
        excluded_results.append(
            {
                **item,
                "exclusion_hit": hit,
                "exclusion_min_ratio": exclusion_min_ratio,
            }
        )
        if hit:
            exclusion_violated = True
            aggregate = 0.0
    return {
        "aggregate": aggregate,
        "keyword_results": keyword_results,
        "excluded_keyword_results": excluded_results,
        "exclusion_violated": exclusion_violated,
        "any_groups": any_group_eval.get("any_groups") if any_group_eval else [],
        "any_groups_pass": any_groups_pass,
        "flat_keyword_count": len(flat_ratios),
    }


def _profile_structural_boost(profile_id: str, ocr_n: str) -> float:
    """Bonus kecil bila ada penanda struktural kuat di OCR (NIK 16 digit, dll.)."""
    pid = (profile_id or "").strip().casefold()
    compact = re.sub(r"\s+", "", ocr_n)
    if pid == "ktp":
        bonus = 0.0
        if _ktp_nik_in_ocr(ocr_n):
            bonus += 0.06
        if re.search(r"\bnik\b", ocr_n) or "nik" in compact:
            bonus += 0.04
        if _ktp_field_signal_count(ocr_n) >= 2:
            bonus += 0.04
        return min(0.12, bonus)
    if pid == "npwp":
        bonus = 0.0
        compact_np = re.sub(r"[^a-z0-9]", "", ocr_n)
        if _ocr_has_npwp_marker(ocr_n):
            bonus += 0.04
        if re.search(r"\d{15,16}", compact_np):
            bonus += 0.05
        if _ocr_has_keyword_signal(ocr_n, "kantor pelayanan"):
            bonus += 0.04
        if re.search(r"\bpratama\b", ocr_n) or "pratama" in compact_np:
            bonus += 0.03
        if re.search(r"\bkpp\b", ocr_n) or "kpp" in compact_np:
            bonus += 0.03
        if re.search(r"\bdjp\b", ocr_n) or "djp" in compact_np:
            bonus += 0.03
        if "tanggalterdaftar" in compact_np or "terdaftar" in compact_np:
            bonus += 0.03
        return min(0.10, bonus)
    if pid == "kk" and "kartukeluarga" in compact:
        return 0.08
    if pid == "mutasi" and _ocr_has_keyword_signal(ocr_n, "e-statement"):
        return 0.08
    if pid == "rekening" and _ocr_has_keyword_signal(ocr_n, "tabungan"):
        if not _ocr_has_keyword_signal(ocr_n, "e-statement"):
            return 0.08
    if pid == "skck" and _ocr_has_keyword_signal(ocr_n, "skck"):
        return 0.08
    if pid == "bpjs":
        bonus = 0.0
        if _ocr_has_keyword_signal(ocr_n, "kartu indonesia sehat"):
            bonus += 0.06
        if re.search(r"\b0\d{12}\b", ocr_n.replace(" ", "")):
            bonus += 0.04
        if re.search(r"\b\d{16}\b", ocr_n):
            bonus += 0.04
        return min(0.10, bonus)
    if pid == "bpjs_tk":
        bonus = 0.0
        if _ocr_has_keyword_signal(ocr_n, "kartu peserta"):
            bonus += 0.05
        if _ocr_has_keyword_signal(ocr_n, "ketenagakerjaan"):
            bonus += 0.05
        compact = re.sub(r"\s+", "", ocr_n)
        if re.search(r"\d{11}", compact):
            bonus += 0.04
        return min(0.10, bonus)
    if pid == "bpjs_kesanggupan":
        bonus = 0.0
        if _ocr_has_keyword_signal(ocr_n, "surat pernyataan kesanggupan"):
            bonus += 0.06
        if _ocr_has_keyword_signal(ocr_n, "tidak aktif"):
            bonus += 0.04
        if _ocr_has_keyword_signal(ocr_n, "meterai tempel") or _ocr_has_keyword_signal(
            ocr_n, "materai tempel"
        ):
            bonus += 0.04
        return min(0.10, bonus)
    if pid == "jkn":
        if _ocr_has_keyword_signal(ocr_n, "info peserta"):
            return 0.08
        if _ocr_has_keyword_signal(ocr_n, "nama peserta") or _ocr_has_keyword_signal(
            ocr_n, "status peserta"
        ):
            return 0.08
        if _ocr_has_keyword_signal(ocr_n, "pandawa"):
            return 0.06
        if _ocr_has_keyword_signal(ocr_n, "kartu indonesia sehat"):
            return 0.08
        if _ocr_has_keyword_signal(ocr_n, "faskes"):
            return 0.06
    if pid == "iuran":
        if _ocr_has_keyword_signal(ocr_n, "info iuran"):
            return 0.08
        if _ocr_has_keyword_signal(ocr_n, "total tagihan"):
            return 0.06
    if pid == "vaksinasi_1":
        bonus = 0.0
        if _ocr_has_keyword_signal(ocr_n, "kartu vaksinasi covid"):
            bonus += 0.05
        if vaksinasi_1_first_dose_signal(ocr_n):
            bonus += 0.05
        return min(0.10, bonus)
    return 0.0


def _document_type_score_boosts(
    profile_id: str,
    ocr_n: str,
    mistral_annotation: dict[str, Any] | None,
) -> dict[str, float]:
    boosts: dict[str, float] = {}
    structural = _profile_structural_boost(profile_id, ocr_n)
    if structural > 0:
        boosts["structural"] = round(structural, 4)

    if mistral_annotation and document_type_profile_from_annotation is not None:
        ann_profile = document_type_profile_from_annotation(mistral_annotation)
        if ann_profile and ann_profile == (profile_id or "").strip().casefold():
            boosts["mistral_document_type"] = 0.10

    return boosts


def _detection_tiebreak_bonus(profile_id: str, ocr_n: str, hint_profile_id: str) -> float:
    """Pecah seri skor: anchor eksklusif profil + preferensi profil yang diminta user."""
    bonus = 0.0
    pid = (profile_id or "").strip().casefold()
    hint = (hint_profile_id or "").strip().casefold()
    compact = re.sub(r"\s+", "", ocr_n)
    if pid == "kk" and "kartu keluarga" in ocr_n:
        bonus += 1.0
    if pid == "npwp" and _ocr_has_npwp_marker(ocr_n):
        bonus += 1.0
    if pid == "ktp" and (re.search(r"\bnik\b", ocr_n) or "nik" in compact):
        bonus += 0.75
    if pid == "ktp" and re.search(r"\b\d{16}\b", ocr_n):
        bonus += 0.5
    if pid == "mutasi" and _ocr_has_keyword_signal(ocr_n, "e-statement"):
        bonus += 1.25
    if pid == "rekening" and _ocr_has_keyword_signal(ocr_n, "tabungan"):
        bonus += 0.75
        if not _ocr_has_keyword_signal(ocr_n, "e-statement"):
            bonus += 1.0
    if pid == "skck" and _ocr_has_keyword_signal(ocr_n, "skck"):
        bonus += 1.0
    if pid == "bpjs":
        if _ocr_has_keyword_signal(ocr_n, "kartu indonesia sehat"):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "bpjs kesehatan"):
            bonus += 1.0
        elif _ocr_has_keyword_signal(ocr_n, "nomor kartu"):
            bonus += 0.75
    if pid == "bpjs_tk":
        if _ocr_has_keyword_signal(ocr_n, "bpjs ketenagakerjaan"):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "ketenagakerjaan"):
            bonus += 1.0
        elif _ocr_has_keyword_signal(ocr_n, "kartu peserta"):
            bonus += 0.75
    if pid == "bpjs_kesanggupan":
        if _ocr_has_keyword_signal(ocr_n, "surat pernyataan kesanggupan"):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "menanggung biaya bpjs"):
            bonus += 1.0
        elif _ocr_has_keyword_signal(ocr_n, "syarat bekerja"):
            bonus += 0.75
    if pid == "jkn":
        if _ocr_has_keyword_signal(ocr_n, "info peserta"):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "nama peserta") or _ocr_has_keyword_signal(
            ocr_n, "status peserta"
        ):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "pandawa"):
            bonus += 1.0
        elif _ocr_has_keyword_signal(ocr_n, "kartu indonesia sehat"):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "faskes"):
            bonus += 1.0
    if pid == "iuran":
        if _ocr_has_keyword_signal(ocr_n, "info iuran"):
            bonus += 1.25
        elif _ocr_has_keyword_signal(ocr_n, "total tagihan"):
            bonus += 1.0
        elif _ocr_has_keyword_signal(ocr_n, "tidak memiliki tagihan pribadi"):
            bonus += 0.75
    if pid == "vaksinasi_1":
        if _ocr_has_keyword_signal(ocr_n, "vaksin primer 1"):
            bonus += 1.25
        elif vaksinasi_1_first_dose_signal(ocr_n):
            bonus += 1.0
        elif _ocr_has_keyword_signal(ocr_n, "kartu vaksinasi covid"):
            bonus += 0.75
    if pid == hint and hint:
        bonus += 0.5
    return bonus


def detect_document_type_from_ocr(
    ocr_text: str,
    *,
    min_aggregate_ratio: float = 0.5,
    hint_profile_id: str = "",
) -> dict[str, Any]:
    """
    Tebak jenis dokumen dari teks OCR dengan membandingkan skor keyword tiap profil.
    `hint_profile_id`: profil yang diminta user — dipakai memecah seri skor (mis. KK vs KTP).
    """
    ocr_n = _normalize(ocr_text)
    min_ratio = max(0.0, min(1.0, float(min_aggregate_ratio)))
    ranked: list[dict[str, Any]] = []

    for profile_id, keywords in CANONICAL_KEYWORDS.items():
        eval_result = _evaluate_profile_keyword_match(
            profile_id,
            ocr_n,
            keywords,
            exclusion_min_ratio=min_ratio,
        )
        aggregate = float(eval_result["aggregate"])
        keyword_results = eval_result["keyword_results"]
        ranked.append(
            {
                "document_profile_id": profile_id,
                "document_type_label": profile_label(profile_id),
                "aggregate_score": round(aggregate, 6),
                "aggregate_percent": round(aggregate * 100.0, 4),
                "keywords": keyword_results,
                "excluded_keywords": eval_result["excluded_keyword_results"],
                "exclusion_violated": eval_result["exclusion_violated"],
                "tiebreak_bonus": round(
                    _detection_tiebreak_bonus(profile_id, ocr_n, hint_profile_id), 4
                ),
            }
        )

    ranked.sort(
        key=lambda x: (float(x["aggregate_score"]), float(x.get("tiebreak_bonus") or 0.0)),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    second_score = float(ranked[1]["aggregate_score"]) if len(ranked) > 1 else 0.0
    detected_profile_id: str | None = None
    confidence = 0.0

    if best and float(best["aggregate_score"]) >= min_ratio:
        detected_profile_id = str(best["document_profile_id"])
        confidence = float(best["aggregate_score"])
        lead = confidence - second_score
        if confidence < min_ratio + 0.08 and lead < 0.06:
            detected_profile_id = None

    return {
        "detected_profile_id": detected_profile_id,
        "detected_document_type": profile_label(detected_profile_id) if detected_profile_id else None,
        "confidence_score": round(confidence * 100.0, 4),
        "confidence_ratio": round(confidence, 6),
        "min_aggregate_ratio": min_ratio,
        "candidates": ranked,
    }


def build_document_verdict(
    *,
    requested_document_type: str,
    requested_profile_id: str,
    detection: dict[str, Any],
    expected_name: str,
    identity_pass: bool | None,
    identity: dict[str, Any] | None,
    identity_min_score: float,
    document_type_pass: bool,
    document_matched: bool,
    name_extraction: dict[str, Any],
    ownership_checked: bool | None = None,
) -> dict[str, Any]:
    """Ringkasan singkat untuk klien: jenis dokumen saat ini + milik user atau tidak."""
    if ownership_checked is None:
        ownership_checked = bool(expected_name.strip())
    detected_profile_id = detection.get("detected_profile_id")
    detected_label = detection.get("detected_document_type")
    requested_label = profile_label(requested_profile_id) or requested_document_type.strip()

    if detected_profile_id:
        document_type_matches_request = detected_profile_id == requested_profile_id.casefold()
    else:
        document_type_matches_request = bool(document_type_pass)

    if not ownership_checked:
        is_own_document: bool | None = None
    elif identity_pass is True:
        is_own_document = True
    else:
        is_own_document = False

    identity_score = None
    if identity and ownership_checked:
        identity_score = float(identity.get("identity_combined_score") or 0.0)

    extracted_name = name_extraction.get("candidate_raw")
    if isinstance(extracted_name, str):
        extracted_name = extracted_name.strip() or None

    if ownership_checked and is_own_document is True and document_type_matches_request:
        summary = f"Dokumen ini adalah {detected_label or requested_label} milik user saat ini."
    elif ownership_checked and is_own_document is True and not document_type_matches_request:
        summary = (
            f"Nama cocok dengan user saat ini, tetapi jenis dokumen terdeteksi "
            f"{detected_label or 'tidak jelas'} (diminta {requested_label})."
        )
    elif ownership_checked and is_own_document is False and document_type_matches_request:
        summary = (
            f"Dokumen terdeteksi {detected_label or requested_label}, "
            "tetapi nama di dokumen tidak cocok dengan user saat ini."
        )
    elif ownership_checked and is_own_document is False:
        summary = (
            f"Dokumen terdeteksi {detected_label or 'tidak jelas'} "
            "dan bukan milik user saat ini (nama tidak cocok)."
        )
    elif not ownership_checked and document_type_matches_request:
        summary = (
            f"Dokumen terdeteksi {detected_label or requested_label}. "
            "Kepemilikan belum dicek karena nama referensi kosong."
        )
    elif not ownership_checked:
        summary = (
            f"Jenis dokumen terdeteksi {detected_label or 'tidak jelas'} "
            f"(diminta {requested_label}). Kepemilikan belum dicek."
        )
    else:
        summary = "Hasil verifikasi dokumen."

    return {
        "document_type_current": detected_profile_id,
        "document_type_current_label": detected_label,
        "document_type_requested": requested_document_type.strip(),
        "document_type_requested_label": requested_label,
        "document_type_matches_request": document_type_matches_request,
        "document_type_detection_confidence": detection.get("confidence_score"),
        "ownership_checked": ownership_checked,
        "is_own_document": is_own_document,
        "expected_name": expected_name.strip() or None,
        "extracted_name": extracted_name,
        "ownership_confidence_score": identity_score,
        "ownership_min_score": identity_min_score if ownership_checked else None,
        "document_matched": document_matched,
        "summary": summary,
    }


def validate_document_ocr(
    ocr_text: str,
    *,
    document_type: str = "",
    document_profile_id: str = "",
    keywords: list[str] | None = None,
    expected_name: str = "",
    expected_nik: str = "",
    expected_bank: str = "",
    expected_account: str = "",
    job_position: str = "",
    aggregate_min_pass_ratio: float = 0.7,
    identity_min_score: float = 65.0,
    mistral_annotation: dict[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """
    - Tipe dokumen: rata-rata skor keyword vs OCR penuh >= aggregate_min_pass_ratio.
    - Identitas (jika nama referensi diisi): penanda nama di OCR; jika gagal, pilih segmen
      yang mirip nama orang lalu yang paling tinggi skor identitas vs nama referensi.
    - document_matched = document_type_pass AND (identitas tidak dicek | identity_pass)
      AND (bukan Helper KTP | status perkawinan BELUM KAWIN).
    """
    keywords = keywords or []
    ocr_n = _normalize(ocr_text)
    ratio_req = max(0.0, min(1.0, float(aggregate_min_pass_ratio)))
    id_min = max(0.0, min(100.0, float(identity_min_score)))
    profile = (document_profile_id or "").strip().casefold()
    if not profile:
        resolved = resolve_keywords(document_type)
        if resolved:
            profile = resolved[0]
            if not keywords:
                keywords = list(resolved[1])
    elif not keywords:
        resolved = resolve_keywords(profile)
        if resolved:
            keywords = list(resolved[1])

    profile_eval = _evaluate_profile_keyword_match(
        profile,
        ocr_n,
        keywords,
        exclusion_min_ratio=ratio_req,
    )
    keyword_results = profile_eval["keyword_results"]
    excluded_keyword_results = profile_eval["excluded_keyword_results"]
    exclusion_violated = bool(profile_eval["exclusion_violated"])
    kw_ratios = [
        float(item["best_score"]) / 100.0
        for item in keyword_results
        if not item.get("skipped") and not item.get("any_group_member")
    ]
    any_groups_pass = bool(profile_eval.get("any_groups_pass", True))
    has_any_groups = bool(profile_eval.get("any_groups"))
    if profile == "vaksinasi_1" and vaksinasi_1_wrong_dose_primary(ocr_text):
        exclusion_violated = True
        profile_eval["exclusion_violated"] = True
        profile_eval["aggregate"] = 0.0
    if (
        profile == "iuran"
        and has_any_groups
        and not any_groups_pass
        and not profile_eval.get("exclusion_violated")
        and iuran_modal_only_state(ocr_text)
    ):
        groups_meta = profile_eval.get("any_groups") or []
        if len(groups_meta) >= 2 and groups_meta[1].get("pass"):
            any_groups_pass = True
    if (
        profile == "vaksinasi_1"
        and has_any_groups
        and not any_groups_pass
        and not profile_eval.get("exclusion_violated")
        and vaksinasi_1_first_dose_signal(ocr_text)
        and not vaksinasi_1_wrong_dose_primary(ocr_text)
    ):
        groups_meta = profile_eval.get("any_groups") or []
        if len(groups_meta) >= 2 and groups_meta[0].get("pass"):
            any_groups_pass = True

    document_type_boosts = _document_type_score_boosts(profile, ocr_n, mistral_annotation)
    boost_total = min(0.15, sum(document_type_boosts.values()))

    if exclusion_violated:
        keyword_aggregate_raw = 0.0
        document_type_aggregate_pass_ratio = 0.0
        document_type_pass = False
    elif has_any_groups and not kw_ratios:
        keyword_aggregate_raw = float(profile_eval["aggregate"])
        document_type_aggregate_pass_ratio = min(1.0, keyword_aggregate_raw + boost_total)
        document_type_pass = any_groups_pass and document_type_aggregate_pass_ratio >= ratio_req
    elif has_any_groups and kw_ratios:
        flat_aggregate = sum(kw_ratios) / len(kw_ratios)
        keyword_aggregate_raw = float(profile_eval["aggregate"])
        document_type_aggregate_pass_ratio = min(1.0, keyword_aggregate_raw + boost_total)
        document_type_pass = (
            flat_aggregate >= ratio_req
            and any_groups_pass
            and document_type_aggregate_pass_ratio >= ratio_req
        )
    elif not kw_ratios:
        keyword_aggregate_raw = 0.0
        document_type_aggregate_pass_ratio = 0.0
        document_type_pass = False
    else:
        keyword_aggregate_raw = sum(kw_ratios) / len(kw_ratios)
        document_type_aggregate_pass_ratio = min(1.0, keyword_aggregate_raw + boost_total)
        document_type_pass = document_type_aggregate_pass_ratio >= ratio_req

    if (
        profile in {"npwp", "pemadanan_npwp"}
        and not exclusion_violated
        and not document_type_pass
        and _ocr_has_npwp_marker(ocr_n)
        and _npwp_tax_id_in_ocr(ocr_n)
    ):
        document_type_pass = True
        document_type_aggregate_pass_ratio = max(
            document_type_aggregate_pass_ratio,
            ratio_req,
        )
        document_type_boosts["npwp_tax_id_anchor"] = round(
            max(0.0, ratio_req - keyword_aggregate_raw),
            4,
        )

    document_type_components_count = len(kw_ratios) + len(profile_eval.get("any_groups") or [])

    skip_identity = skip_identity_validation(profile)
    exp_nik = re.sub(r"\D", "", expected_nik or "")
    npwp_nik_gate = profile in {"npwp", "pemadanan_npwp"} and len(exp_nik) == 16
    if profile in {"npwp", "pemadanan_npwp"}:
        want_identity = npwp_nik_gate and not skip_identity
    elif profile == "kk":
        want_identity = (bool(expected_name.strip()) or len(exp_nik) == 16) and not skip_identity
    else:
        want_identity = bool(expected_name.strip()) and not skip_identity
    name_extraction: dict[str, Any] = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_no_expected_name",
    }
    identity: dict[str, Any] | None = None
    identity_pass: bool | None = None
    identity_skip_reason: str | None = None
    if skip_identity and (expected_name.strip() or npwp_nik_gate):
        identity_skip_reason = (
            f"Identitas tidak diuji untuk profil {profile_label(profile)} — "
            "validasi hanya memeriksa keyword dokumen."
        )
        name_extraction["method"] = "skipped_profile_no_identity"
    elif profile in {"npwp", "pemadanan_npwp"} and not npwp_nik_gate:
        identity_skip_reason = (
            "Identitas NPWP tidak diuji karena NIK/NPWP 16 digit referensi kosong — "
            "validasi hanya mengandalkan profil keyword."
        )
        name_extraction["method"] = "skipped_no_expected_nik"

    if want_identity and npwp_nik_gate:
        matched_nik = npwp_id_matches_expected(ocr_text, exp_nik)
        if not matched_nik and mistral_annotation:
            ann_nik = re.sub(r"\D", "", str(mistral_annotation.get("identity_number") or ""))
            if ann_nik == exp_nik or (len(ann_nik) == 15 and exp_nik.endswith(ann_nik)):
                matched_nik = exp_nik
        identity_pass = bool(matched_nik)
        identity = {
            "expected_nik": exp_nik,
            "ocr_nik_matched": matched_nik or "",
            "npwp_nik_identity": True,
            "token_gate_pass": identity_pass,
            "identity_combined_score": 100.0 if identity_pass else 0.0,
            "identity_combined_ratio": 1.0 if identity_pass else 0.0,
            "note": "Identitas NPWP dinilai dari NIK/nomor 16 digit, bukan dari nama.",
        }
        name_extraction = {
            "candidate_raw": matched_nik or None,
            "candidate_normalized": matched_nik,
            "method": "npwp_nik_identity",
        }
        if (
            not exclusion_violated
            and not document_type_pass
            and identity_pass is True
            and (
                _npwp_tax_id_in_ocr(ocr_n)
                or bool(matched_nik)
            )
        ):
            document_type_pass = True
            document_type_aggregate_pass_ratio = max(
                document_type_aggregate_pass_ratio,
                ratio_req,
            )
            document_type_boosts["npwp_identity_tax_anchor"] = round(
                max(0.0, ratio_req - keyword_aggregate_raw),
                4,
            )
        document_matched = bool(document_type_pass and identity_pass)
    elif want_identity:
        ann_name: str | None = None
        if mistral_annotation and holder_name_from_annotation is not None:
            ann_name = holder_name_from_annotation(mistral_annotation)

        if ann_name:
            struct_cand, struct_method = ann_name, "mistral_document_annotation"
        else:
            struct_cand, struct_method = extract_holder_name_candidate(ocr_text, profile)

        scored: list[tuple[float, str, str]] = []
        if struct_cand:
            scv = float(
                compare_extracted_identity_scores(struct_cand, expected_name)[
                    "identity_combined_score"
                ]
            )
            scored.append((scv, struct_cand, struct_method))
        if profile == "kk" and not ann_name:
            exp_nik_kk = re.sub(r"\D", "", expected_nik or "")
            if len(exp_nik_kk) == 16:
                nik_name = extract_kk_name_by_nik(ocr_text, exp_nik_kk)
                if nik_name:
                    scv = float(
                        compare_extracted_identity_scores(nik_name, expected_name)[
                            "identity_combined_score"
                        ]
                    )
                    scored.append((scv, nik_name, "kk_nik_row"))
            for name in extract_kk_member_names(ocr_text):
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, name, "kk_table_row"))
        jkn_names = extract_jkn_participant_names(ocr_text) if profile == "jkn" else []
        jkn_best = 0.0
        if profile == "jkn" and not ann_name:
            for name in jkn_names:
                ident_sc = compare_extracted_identity_scores(name, expected_name)
                scv = float(ident_sc["identity_combined_score"])
                if jkn_masked_name_matches(name, expected_name):
                    scv = max(scv, 92.0)
                jkn_best = max(jkn_best, scv)
                scored.append((scv, name, "jkn_participant_name"))

        iuran_names = extract_iuran_participant_names(ocr_text) if profile == "iuran" else []
        if profile == "iuran" and not ann_name:
            for name in iuran_names:
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, name, "iuran_participant_name"))

        vax_names = extract_vaksinasi_1_names(ocr_text) if profile == "vaksinasi_1" else []
        if profile == "vaksinasi_1" and not ann_name:
            for name in vax_names:
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, name, "vaksinasi_1_recipient_name"))

        health_names = (
            extract_keterangan_kesehatan_names(ocr_text)
            if profile == "keterangan_kesehatan"
            else []
        )
        if profile == "keterangan_kesehatan" and not ann_name:
            for name in health_names:
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, name, "keterangan_kesehatan_nama_field"))

        npwp_names = (
            extract_npwp_holder_names(ocr_text)
            if profile in {"npwp", "pemadanan_npwp"}
            else []
        )
        if profile in {"npwp", "pemadanan_npwp"} and not ann_name:
            for name in npwp_names:
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, name, "npwp_name_line"))

        skip_person_like = (
            (profile == "jkn" and jkn_best >= id_min)
            or (profile == "iuran" and iuran_names)
            or (profile == "iuran" and iuran_modal_only_state(ocr_text))
            or (profile == "vaksinasi_1" and vax_names)
            or (profile in {"npwp", "pemadanan_npwp"} and npwp_names)
        )
        if not ann_name and not skip_person_like:
            window_sources: list[tuple[str, str]] = [(ocr_text, "person_like_segment")]
            for seg in person_like_segments(ocr_text):
                window_sources.append((seg, "person_like_segment"))
            seen_window: set[str] = set()
            for source_text, method in window_sources:
                for window in sliding_name_windows(source_text, expected_name):
                    nk = _normalize_name(window)
                    if nk in seen_window:
                        continue
                    seen_window.add(nk)
                    scv = float(
                        compare_extracted_identity_scores(window, expected_name)[
                            "identity_combined_score"
                        ]
                    )
                    scored.append((scv, window, method))
            for seg in person_like_segments(ocr_text):
                nk = _normalize_name(seg)
                if nk in seen_window:
                    continue
                scv = float(
                    compare_extracted_identity_scores(seg, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, seg, "person_like_segment"))

        by_norm: dict[str, tuple[float, str, str]] = {}
        for scv, text, meth in scored:
            nk = _normalize_name(text)
            if nk not in by_norm or scv > by_norm[nk][0]:
                by_norm[nk] = (scv, text, meth)

        if by_norm:
            best_scv, raw_cand, method = max(by_norm.values(), key=lambda t: t[0])
        else:
            raw_cand = None
            method = "failed"

        name_extraction = {
            "candidate_raw": raw_cand,
            "candidate_normalized": _normalize_name(raw_cand) if raw_cand else "",
            "method": method,
        }
        if profile == "iuran" and iuran_modal_only_state(ocr_text) and document_type_pass:
            identity_pass = True
            identity = {
                "extracted_normalized": "",
                "expected_normalized": _normalize(expected_name),
                "scores": {"token_sort_ratio": 0.0, "wratio": 0.0, "partial_ratio": 0.0},
                "identity_combined_score": 0.0,
                "identity_combined_ratio": 0.0,
                "extraction_failed": False,
                "iuran_modal_skip": True,
                "note": (
                    "Layar modal Info Iuran tanpa tagihan pribadi — nama tidak ditampilkan; "
                    "identitas dianggap lolos bila profil keyword iuran lulus."
                ),
            }
            name_extraction["method"] = "iuran_modal_no_name_displayed"
        elif raw_cand:
            identity = compare_extracted_identity_scores(raw_cand, expected_name)
            identity_pass = float(identity["identity_combined_score"]) >= id_min and bool(
                identity.get("token_gate_pass", True)
            )
            if (
                not identity_pass
                and profile == "jkn"
                and jkn_masked_name_matches(raw_cand, expected_name)
            ):
                identity_pass = True
                identity = {
                    **identity,
                    "token_gate_pass": True,
                    "jkn_masked_name_pass": True,
                    "identity_combined_score": max(
                        float(identity.get("identity_combined_score") or 0.0),
                        92.0,
                    ),
                    "note": (
                        "Nama peserta JKN pada chat Pandawa tersensor (X), "
                        "tetapi awalan kata cocok dengan nama referensi."
                    ),
                }
        else:
            identity = {
                "extracted_normalized": "",
                "expected_normalized": _normalize(expected_name),
                "scores": {"token_sort_ratio": 0.0, "wratio": 0.0, "partial_ratio": 0.0},
                "identity_combined_score": 0.0,
                "identity_combined_ratio": 0.0,
                "extraction_failed": True,
            }
            identity_pass = False

        if profile == "kk":
            nik_hit = kk_nik_matches_expected(ocr_text, exp_nik)
            nik_ok = bool(nik_hit)
            if not nik_ok and mistral_annotation and len(exp_nik) == 16:
                ann_nik = re.sub(
                    r"\D", "", str(mistral_annotation.get("identity_number") or "")
                )
                nik_ok = ann_nik == exp_nik
                if nik_ok:
                    nik_hit = exp_nik
            name_ok = bool(identity_pass)
            compact_ocr = re.sub(r"[^a-z0-9]", "", _normalize_name(ocr_text))
            compact_exp = re.sub(r"[^a-z0-9]", "", _normalize_name(expected_name))
            if not name_ok and len(compact_exp) >= 5 and compact_exp in compact_ocr:
                name_ok = True
            identity_pass = name_ok or nik_ok
            identity = {
                **(identity or {}),
                "kk_or_identity": True,
                "kk_name_pass": name_ok,
                "kk_nik_pass": nik_ok,
                "expected_nik": exp_nik,
                "ocr_nik_matched": nik_hit if nik_ok else "",
                "note": "Identitas KK: nama atau NIK anggota cukup salah satu yang cocok (OR).",
            }

        compact_ocr = re.sub(r"[^a-z0-9]", "", _normalize_name(ocr_text))
        compact_exp = re.sub(r"[^a-z0-9]", "", _normalize_name(expected_name))
        if (
            not identity_pass
            and profile in {"ktp", "npwp", "pemadanan_npwp", "rekening", "jkn", "bpjs", "keterangan_kesehatan"}
            and len(compact_exp) >= 8
            and compact_exp in compact_ocr
        ):
            identity_pass = True
            identity = {
                **(identity or {}),
                "expected_normalized": _normalize(expected_name),
                "ocr_compact_name_fallback": True,
                "token_gate_pass": True,
                "identity_combined_score": max(
                    float((identity or {}).get("identity_combined_score") or 0.0),
                    80.0,
                ),
                "note": (
                    "Nama referensi (tanpa spasi) ditemukan utuh di teks OCR KTP, "
                    "meski kandidat baris 'Nama' tidak terpilih dengan benar."
                ),
            }

        if (
            not identity_pass
            and profile == "rekening"
            and len(compact_exp) >= 10
        ):
            compact_letters = re.sub(r"[^a-z]", "", compact_ocr)
            prefix_hit = ""
            for n in range(len(compact_exp), 9, -1):
                pref = compact_exp[:n]
                if pref in compact_letters and (n / len(compact_exp)) >= 0.68:
                    prefix_hit = pref
                    break
            if prefix_hit:
                identity_pass = True
                identity = {
                    **(identity or {}),
                    "expected_normalized": _normalize(expected_name),
                    "ocr_compact_name_prefix_fallback": True,
                    "token_gate_pass": True,
                    "identity_combined_score": max(
                        float((identity or {}).get("identity_combined_score") or 0.0),
                        80.0,
                    ),
                    "note": (
                        "Nama di kartu rekening terpotong OCR (ellipsis) tetapi "
                        "awalan nama cocok dengan referensi."
                    ),
                }
            name_extraction["method"] = "ktp_ocr_compact_name_fallback"

        exp_nik = re.sub(r"\D", "", expected_nik or "")
        if (
            not identity_pass
            and profile in {"npwp", "pemadanan_npwp", "ktp", "jkn", "bpjs", "keterangan_kesehatan"}
            and document_type_pass
            and len(exp_nik) == 16
        ):
            if profile in {"npwp", "pemadanan_npwp"}:
                matched_nik = npwp_id_matches_expected(ocr_text, exp_nik)
            elif profile in {"jkn", "bpjs"}:
                matched_nik = jkn_nik_matches_expected(ocr_text, exp_nik)
                if not matched_nik and profile == "bpjs":
                    ocr_niks = extract_nik16_from_ocr(ocr_text)
                    matched_nik = exp_nik if exp_nik in ocr_niks else ""
            else:
                ocr_niks = extract_nik16_from_ocr(ocr_text)
                matched_nik = exp_nik if exp_nik in ocr_niks else ""
            ann_nik = ""
            if mistral_annotation:
                ann_nik = re.sub(r"\D", "", str(mistral_annotation.get("identity_number") or ""))
            if not matched_nik and ann_nik == exp_nik:
                matched_nik = exp_nik
            if matched_nik:
                identity_pass = True
                nik_name = None
                identity = {
                    **(identity or {}),
                    "expected_normalized": _normalize(expected_name),
                    "nik_match_fallback": True,
                    "expected_nik": exp_nik,
                    "ocr_nik_matched": matched_nik,
                    "note": (
                        "Nama OCR tidak cukup mirip, tetapi NIK 16 digit di dokumen "
                        "cocok dengan referensi — identitas dianggap lolos."
                    ),
                }
                if nik_name:
                    identity["extracted_normalized"] = _normalize(nik_name)
                    identity["ocr_name_by_nik"] = nik_name
                if not name_extraction.get("candidate_raw"):
                    if profile == "ktp":
                        name_extraction["method"] = "ktp_nik_fallback"
                    elif profile == "kk":
                        name_extraction["method"] = "kk_nik_fallback"
                    elif profile == "jkn":
                        name_extraction["method"] = "jkn_nik_fallback"
                    elif profile == "bpjs":
                        name_extraction["method"] = "bpjs_nik_fallback"
                    else:
                        name_extraction["method"] = "npwp_nik_fallback"

        if (
            not identity_pass
            and profile == "bpjs"
            and document_type_pass
        ):
            raw_one = str(name_extraction.get("candidate_raw") or "")
            ext_toks = _name_identity_tokens(_normalize_name(raw_one))
            exp_toks = _name_identity_tokens(_normalize_name(expected_name))
            if (
                len(ext_toks) == 1
                and len(exp_toks) >= 2
                and len(ext_toks[0]) >= 4
                and _token_pair_score(ext_toks[0], exp_toks[0]) >= 90.0
            ):
                identity_pass = True
                identity = {
                    **(identity or {}),
                    "expected_normalized": _normalize(expected_name),
                    "bpjs_truncated_first_name": True,
                    "token_gate_pass": True,
                    "identity_combined_score": max(
                        float((identity or {}).get("identity_combined_score") or 0.0),
                        80.0,
                    ),
                    "note": (
                        "Nama pada kartu KIS terpotong OCR, tetapi nama depan "
                        "cocok dengan referensi."
                    ),
                }
                name_extraction["method"] = "bpjs_truncated_first_name"

        if (
            not identity_pass
            and profile == "keterangan_kesehatan"
            and document_type_pass
        ):
            raw_one = str(name_extraction.get("candidate_raw") or "")
            ext_toks = _name_identity_tokens(_normalize_name(raw_one))
            exp_toks = _name_identity_tokens(_normalize_name(expected_name))
            if (
                len(ext_toks) == 1
                and len(exp_toks) >= 1
                and len(ext_toks[0]) >= 3
                and _token_pair_score(ext_toks[0], exp_toks[0]) >= 88.0
            ):
                identity_pass = True
                identity = {
                    **(identity or {}),
                    "expected_normalized": _normalize(expected_name),
                    "health_truncated_first_name": True,
                    "token_gate_pass": True,
                    "identity_combined_score": max(
                        float((identity or {}).get("identity_combined_score") or 0.0),
                        80.0,
                    ),
                    "note": (
                        "Nama pada surat sehat terpotong OCR, tetapi nama depan "
                        "cocok dengan referensi."
                    ),
                }
                name_extraction["method"] = "keterangan_kesehatan_truncated_first_name"

        if (
            not identity_pass
            and profile == "jkn"
            and document_type_pass
            and len(exp_nik) == 16
        ):
            dob_hit = jkn_nik_birthdate_in_ocr(ocr_text, exp_nik)
            if dob_hit:
                identity_pass = True
                identity = {
                    **(identity or {}),
                    "expected_normalized": _normalize(expected_name),
                    "jkn_nik_birthdate_fallback": True,
                    "expected_nik": exp_nik,
                    "ocr_birthdate_matched": dob_hit,
                    "token_gate_pass": True,
                    "identity_combined_score": max(
                        float((identity or {}).get("identity_combined_score") or 0.0),
                        80.0,
                    ),
                    "note": (
                        "Nama OCR tidak cukup mirip, tetapi tanggal lahir dari NIK "
                        "referensi muncul di layar Info Peserta."
                    ),
                }
                if not name_extraction.get("candidate_raw"):
                    name_extraction["method"] = "jkn_nik_birthdate_fallback"

        if (
            profile in {"npwp", "pemadanan_npwp"}
            and not exclusion_violated
            and not document_type_pass
            and identity_pass is True
            and (
                _npwp_tax_id_in_ocr(ocr_n)
                or npwp_id_matches_expected(ocr_text, exp_nik)
            )
        ):
            document_type_pass = True
            document_type_aggregate_pass_ratio = max(
                document_type_aggregate_pass_ratio,
                ratio_req,
            )
            document_type_boosts["npwp_identity_tax_anchor"] = round(
                max(0.0, ratio_req - keyword_aggregate_raw),
                4,
            )

        if (
            profile == "ktp"
            and not exclusion_violated
            and not document_type_pass
            and _ktp_nik_in_ocr(ocr_n)
            and _ktp_field_signal_count(ocr_n) >= 2
            and (identity_pass is True or _ktp_field_signal_count(ocr_n) >= 3)
        ):
            document_type_pass = True
            document_type_aggregate_pass_ratio = max(
                document_type_aggregate_pass_ratio,
                ratio_req,
            )
            document_type_boosts["ktp_nik_anchor"] = round(
                max(0.0, ratio_req - keyword_aggregate_raw),
                4,
            )

        if (
            profile == "bpjs"
            and not exclusion_violated
            and not document_type_pass
            and identity_pass is True
            and _bpjs_field_signal_count(ocr_n) >= 2
        ):
            document_type_pass = True
            document_type_aggregate_pass_ratio = max(
                document_type_aggregate_pass_ratio,
                ratio_req,
            )
            document_type_boosts["bpjs_identity_field_anchor"] = round(
                max(0.0, ratio_req - keyword_aggregate_raw),
                4,
            )

        if (
            profile == "jkn"
            and not exclusion_violated
            and not document_type_pass
            and identity_pass is True
            and (
                _ocr_has_keyword_signal(ocr_n, "kartu indonesia sehat")
                or _ocr_has_keyword_signal(ocr_n, "info peserta")
                or _ocr_has_keyword_signal(ocr_n, "nama peserta")
            )
        ):
            document_type_pass = True
            document_type_aggregate_pass_ratio = max(
                document_type_aggregate_pass_ratio,
                ratio_req,
            )
            document_type_boosts["jkn_kis_or_info_anchor"] = round(
                max(0.0, ratio_req - keyword_aggregate_raw),
                4,
            )

        document_matched = bool(document_type_pass and identity_pass)
    elif skip_identity:
        document_matched = bool(document_type_pass)
    else:
        if name_extraction.get("method") not in {
            "skipped_no_expected_nik",
            "skipped_profile_no_identity",
        }:
            name_extraction["method"] = "skipped_no_expected_name"
        document_matched = bool(document_type_pass)

    bank_detection: dict[str, Any] | None = None
    bank_pass: bool | None = None
    account_pass: bool | None = None
    identity_or_account_pass: bool | None = None
    if profile in BANK_ACCOUNT_PROFILES:
        bank_detection = validate_rekening_bank(
            ocr_text,
            expected_bank=expected_bank,
            expected_account=expected_account,
        )
        bank_pass = bank_detection.get("bank_pass")
        account_pass = bank_detection.get("account_pass")

        exp_bank = normalize_expected_bank(expected_bank or "")
        if not exp_bank:
            bank_pass = False
            bank_detection["bank_pass"] = False
            bank_detection["note"] = (
                f"expected_bank wajib untuk validasi {profile_label(profile).casefold()} "
                "(mandiri | mas)."
            )
        elif bank_pass is None:
            bank_pass = False
            bank_detection["bank_pass"] = False

        if profile == "rekening":
            has_name_ref = bool(expected_name.strip())
            has_account_ref = bool(bank_detection.get("expected_account"))
            identity_or_account_pass = _rekening_identity_and_account_pass(
                has_name_ref=has_name_ref,
                has_account_ref=has_account_ref,
                identity_pass=identity_pass,
                account_pass=account_pass,
            )
            document_matched = bool(
                document_type_pass
                and bank_pass is True
                and identity_or_account_pass is True
            )
            if (
                not document_matched
                and not exclusion_violated
                and bank_pass is True
                and identity_or_account_pass is True
            ):
                document_type_pass = True
                document_matched = True
                document_type_boosts["rekening_identity_account_anchor"] = round(
                    max(0.0, ratio_req - keyword_aggregate_raw),
                    4,
                )
                document_type_aggregate_pass_ratio = max(
                    document_type_aggregate_pass_ratio,
                    ratio_req,
                )
        else:
            if not bank_detection.get("expected_account"):
                account_pass = False
                bank_detection["account_pass"] = False
                bank_detection.setdefault(
                    "note",
                    "expected_account wajib untuk validasi mutasi.",
                )
            document_matched = bool(
                document_type_pass
                and bank_pass is True
                and account_pass is True
            )
            if (
                not document_matched
                and not exclusion_violated
                and bank_pass is True
                and account_pass is True
            ):
                document_type_pass = True
                document_matched = True
                document_type_boosts["mutasi_account_anchor"] = round(
                    max(0.0, ratio_req - keyword_aggregate_raw),
                    4,
                )
                document_type_aggregate_pass_ratio = max(
                    document_type_aggregate_pass_ratio,
                    ratio_req,
                )

    explanation = _validation_explanation(
        document_matched=document_matched,
        document_type_pass=document_type_pass,
        document_type_aggregate_pass_ratio=document_type_aggregate_pass_ratio,
        aggregate_min_pass_ratio=ratio_req,
        keyword_results=keyword_results,
        want_identity=want_identity,
        identity_pass=identity_pass,
        identity_min_score=id_min,
        identity=identity,
        name_extraction=name_extraction,
        expected_name_display=expected_name.strip(),
        identity_skip_reason=identity_skip_reason,
    )
    if exclusion_violated:
        hits = [
            str(item.get("keyword_raw") or "")
            for item in excluded_keyword_results
            if item.get("exclusion_hit")
        ]
        blocked_by = ", ".join(f"«{h}»" for h in hits if h) or "keyword terlarang"
        exclusion_line = (
            f"Profil ditolak karena teks mengandung keyword yang dilarang: {blocked_by}."
        )
        explanation["detail_lines"] = [exclusion_line, *list(explanation.get("detail_lines") or [])]
        explanation["summary"] = (
            "Dokumen tidak cocok karena profil jenis dokumen: keyword terlarang terdeteksi di OCR."
        )
        blockers = list(explanation.get("primary_blockers") or [])
        if "DOCUMENT_TYPE" not in blockers:
            blockers.insert(0, "DOCUMENT_TYPE")
        explanation["primary_blockers"] = blockers
        if explanation.get("gates", {}).get("document_type"):
            explanation["gates"]["document_type"]["pass"] = False

    if profile in BANK_ACCOUNT_PROFILES and bank_detection:
        doc_lbl = profile_label(profile) or profile
        exp_lbl = bank_detection.get("expected_bank_label") or bank_detection.get("expected_bank")
        det_lbl = bank_detection.get("detected_bank_label")
        if bank_pass is True and exp_lbl:
            bank_line = f"Bank: cocok dengan {exp_lbl}"
            if det_lbl and det_lbl != exp_lbl:
                bank_line = f"Bank: cocok dengan {exp_lbl} (terdeteksi: {det_lbl})"
        elif exp_lbl:
            bank_line = f"Bank: tidak cocok — diharapkan {exp_lbl}"
            if det_lbl:
                bank_line += f", terdeteksi {det_lbl}"
            else:
                bank_line += " (bank tidak teridentifikasi di OCR)"
        else:
            bank_line = "Bank: expected_bank wajib (mandiri | mas)."
        explanation["detail_lines"] = [bank_line, *list(explanation.get("detail_lines") or [])]
        if bank_pass is False:
            explanation["summary"] = (
                f"Dokumen {doc_lbl} tidak cocok: bank bukan {exp_lbl}."
                if exp_lbl
                else f"Dokumen {doc_lbl} tidak cocok: expected_bank wajib."
            )
            blockers = list(explanation.get("primary_blockers") or [])
            if "BANK" not in blockers:
                blockers.insert(0, "BANK")
            explanation["primary_blockers"] = blockers
            gates = dict(explanation.get("gates") or {})
            gates["bank"] = {"pass": False}
            explanation["gates"] = gates

    if profile in BANK_ACCOUNT_PROFILES and bank_detection and bank_detection.get("expected_account"):
        exp_acct = bank_detection.get("expected_account")
        if account_pass is True:
            acct_line = f"Nomor rekening: cocok dengan {exp_acct}"
        else:
            acct_line = f"Nomor rekening: tidak cocok — diharapkan {exp_acct}"
            det = bank_detection.get("account_detection") or {}
            found = det.get("detected_accounts") or []
            if found:
                acct_line += f", terbaca di OCR: {', '.join(str(x) for x in found[:3])}"
            else:
                acct_line += " (nomor rekening tidak terbaca di OCR)"
        explanation["detail_lines"] = [acct_line, *list(explanation.get("detail_lines") or [])]
        if profile == "mutasi" and account_pass is False and bank_pass is not False:
            explanation["summary"] = "Dokumen mutasi tidak cocok: nomor rekening tidak sesuai."
            blockers = list(explanation.get("primary_blockers") or [])
            if "ACCOUNT" not in blockers:
                blockers.insert(0, "ACCOUNT")
            explanation["primary_blockers"] = blockers
            gates = dict(explanation.get("gates") or {})
            gates["account"] = {"pass": False}
            explanation["gates"] = gates

    if profile == "rekening" and identity_or_account_pass is not None:
        has_name_ref = bool(expected_name.strip())
        has_account_ref = bool((bank_detection or {}).get("expected_account"))
        if identity_or_account_pass is True:
            if has_name_ref and has_account_ref:
                or_line = "Nama dan nomor rekening: keduanya cocok (AND)."
            elif has_name_ref:
                or_line = "Nama: cocok dengan referensi."
            else:
                or_line = "Nomor rekening: cocok dengan referensi."
        else:
            if has_name_ref and has_account_ref:
                bits = []
                if identity_pass is not True:
                    bits.append("nama lengkap tidak cocok")
                if account_pass is not True:
                    bits.append("nomor rekening tidak terlihat/tidak cocok")
                or_line = "Identitas rekening tidak lolos: " + " dan ".join(bits) + "."
            elif has_name_ref:
                or_line = "Nama tidak cocok dengan referensi."
            else:
                or_line = "Nomor rekening tidak cocok dengan referensi."
        explanation["detail_lines"] = [or_line, *list(explanation.get("detail_lines") or [])]
        if identity_or_account_pass is False and bank_pass is not False:
            explanation["summary"] = or_line
            blockers = list(explanation.get("primary_blockers") or [])
            if "IDENTITY_AND_ACCOUNT" not in blockers:
                blockers.insert(0, "IDENTITY_AND_ACCOUNT")
            explanation["primary_blockers"] = blockers
            gates = dict(explanation.get("gates") or {})
            gates["identity_or_account"] = {"pass": False}
            explanation["gates"] = gates
        elif identity_or_account_pass is False and account_pass is False:
            blockers = list(explanation.get("primary_blockers") or [])
            if "ACCOUNT" not in blockers:
                blockers.append("ACCOUNT")
            explanation["primary_blockers"] = blockers

    detection = detect_document_type_from_ocr(
        ocr_text,
        min_aggregate_ratio=ratio_req,
        hint_profile_id=profile,
    )
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=expected_name,
        identity_pass=identity_pass,
        identity=identity,
        identity_min_score=id_min,
        document_type_pass=document_type_pass,
        document_matched=document_matched,
        name_extraction=name_extraction,
        ownership_checked=(
            (profile in {"npwp", "pemadanan_npwp"} and len(exp_nik) == 16)
            or bool(expected_name.strip())
        )
        and not skip_identity,
    )
    if profile == "rekening" and identity_or_account_pass is not None:
        verdict["is_own_document"] = identity_or_account_pass
        if identity_or_account_pass is True and document_matched:
            verdict["summary"] = (
                f"Dokumen rekening {bank_detection.get('expected_bank_label') or ''} "
                "cocok (bank + nama lengkap + nomor rekening)."
            ).strip()
    elif profile == "mutasi" and account_pass is not None:
        verdict["is_own_document"] = account_pass
        if account_pass is True and document_matched:
            verdict["summary"] = (
                f"Dokumen mutasi {bank_detection.get('expected_bank_label') or ''} "
                "cocok (bank + nomor rekening)."
            ).strip()

    result = {
        "document_type": (document_type or "").strip(),
        "document_profile_id": profile,
        "mistral_annotation": mistral_annotation,
        "ocr_normalized": ocr_n,
        "aggregate_min_pass_ratio": ratio_req,
        "identity_min_score": id_min,
        "document_type_pass": document_type_pass,
        "document_type_keyword_aggregate_raw": round(keyword_aggregate_raw, 6),
        "document_type_boosts": document_type_boosts,
        "document_type_aggregate_pass_ratio": round(document_type_aggregate_pass_ratio, 6),
        "document_type_components_count": document_type_components_count,
        "aggregate_pass_ratio": round(document_type_aggregate_pass_ratio, 6),
        "components_count": document_type_components_count,
        "name_extraction": name_extraction,
        "identity": identity,
        "identity_pass": identity_pass,
        "keywords": keyword_results,
        "any_groups": profile_eval.get("any_groups") or [],
        "any_groups_pass": any_groups_pass,
        "excluded_keywords": excluded_keyword_results,
        "exclusion_violated": exclusion_violated,
        "document_matched": document_matched,
        "explanation": explanation,
        "document_type_detection": detection,
        "verdict": verdict,
        "is_own_document": verdict["is_own_document"],
        "document_type_current": verdict["document_type_current"],
        "document_type_current_label": verdict["document_type_current_label"],
        "bank_detection": bank_detection,
        "bank_pass": bank_pass,
        "account_pass": account_pass,
        "identity_or_account_pass": identity_or_account_pass,
        "job_position": (job_position or "").strip() or None,
        "marital_pass": None,
        "marital_status": None,
    }
    return attach_user_errors(
        apply_keterangan_kesehatan_gates(
            apply_jkn_status_extract(
                apply_bpjs_kesanggupan_stamp_gate(
                    apply_skck_expiry_gate(
                        apply_ktp_helper_marital_gate(
                            result,
                            ocr_text=ocr_text,
                            job_position=job_position,
                            profile_id=profile,
                        ),
                        ocr_text=ocr_text,
                        profile_id=profile,
                        today=today,
                    ),
                    ocr_text=ocr_text,
                    profile_id=profile,
                ),
                ocr_text=ocr_text,
                profile_id=profile,
                expected_name=expected_name,
            ),
            ocr_text=ocr_text,
            profile_id=profile,
            today=today,
        )
    )
