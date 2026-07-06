"""Perbandingan teks hasil OCR dengan nama referensi dan keyword dokumen (RapidFuzz)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from systems.validation.document_profiles import (
    CANONICAL_KEYWORDS,
    any_keyword_groups_for_profile,
    excluded_keywords_for_profile,
    profile_label,
    skip_identity_validation,
)
from systems.validation.name_extraction import (
    extract_holder_name_candidate,
    extract_iuran_participant_names,
    extract_jkn_participant_names,
    extract_kk_member_names,
    extract_vaksinasi_1_names,
    iuran_modal_only_state,
    person_like_segments,
    sliding_name_windows,
    vaksinasi_1_first_dose_signal,
    vaksinasi_1_wrong_dose_primary,
)

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
    return re.sub(r"\s+", " ", s).strip()


def _normalize_keyword(s: str) -> str:
    """Sama seperti teks penuh; slash dipasangkan jadi spasi agar 'rt/rw' ~ 'rt rw'."""
    s = (s or "").strip().casefold()
    s = s.replace("/", " ")
    return re.sub(r"\s+", " ", s)


@dataclass(frozen=True)
class NameFuzzyResult:
    ocr_normalized: str
    expected_normalized: str
    scores: dict[str, float]
    best_score: float
    matched: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "ocr_normalized": self.ocr_normalized,
            "expected_normalized": self.expected_normalized,
            "scores": self.scores,
            "best_score": self.best_score,
            "matched": self.matched,
        }


def compare_ocr_name_to_expected(
    ocr_text: str,
    expected_name: str,
    *,
    threshold: float | None = 85.0,
) -> NameFuzzyResult:
    """
    Menggabungkan beberapa metrik fuzzy; keputusan `matched` memakai skor terbaik
    (cocok untuk kesalahan OCR, urutan kata berbeda, atau teks OCR lebih panjang).
    Dipakai endpoint compare-names (boleh memakai teks panjang).
    """
    a = _normalize(ocr_text)
    b = _normalize(expected_name)

    if not a or not b:
        scores = {
            "ratio": 0.0,
            "partial_ratio": 0.0,
            "token_sort_ratio": 0.0,
            "token_set_ratio": 0.0,
            "wratio": 0.0,
        }
        return NameFuzzyResult(
            ocr_normalized=a,
            expected_normalized=b,
            scores=scores,
            best_score=0.0,
            matched=False,
        )

    scores = {
        "ratio": float(fuzz.ratio(a, b)),
        "partial_ratio": float(fuzz.partial_ratio(a, b)),
        "token_sort_ratio": float(fuzz.token_sort_ratio(a, b)),
        "token_set_ratio": float(fuzz.token_set_ratio(a, b)),
        "wratio": float(fuzz.WRatio(a, b)),
    }
    best = max(scores.values())
    if threshold is None:
        matched = False
    else:
        matched = best >= float(threshold)
    return NameFuzzyResult(
        ocr_normalized=a,
        expected_normalized=b,
        scores=scores,
        best_score=best,
        matched=matched,
    )


def compare_extracted_identity_scores(extracted: str, expected_name: str) -> dict[str, Any]:
    """
    Identitas: hanya bandingkan segmen nama yang diekstrak vs nama referensi.
    Skor gabungan = rata-rata token_sort_ratio, WRatio, dan partial_ratio pada **dua string nama**
    (partial di sini tidak memakai teks dokumen penuh, jadi tidak membesar skor palsu).
    """
    a = _normalize_name(extracted)
    b = _normalize_name(expected_name)
    if not a or not b:
        return {
            "extracted_normalized": a,
            "expected_normalized": b,
            "scores": {
                "token_sort_ratio": 0.0,
                "wratio": 0.0,
                "partial_ratio": 0.0,
            },
            "identity_combined_score": 0.0,
            "identity_combined_ratio": 0.0,
        }
    ts = float(fuzz.token_sort_ratio(a, b))
    wr = float(fuzz.WRatio(a, b))
    pr = float(fuzz.partial_ratio(a, b))
    combined = (ts + wr + pr) / 3.0
    return {
        "extracted_normalized": a,
        "expected_normalized": b,
        "scores": {"token_sort_ratio": ts, "wratio": wr, "partial_ratio": pr},
        "identity_combined_score": round(combined, 4),
        "identity_combined_ratio": round(combined / 100.0, 6),
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

    # Label pendek (nik, npwp): cek juga kecocokan substring pada teks padat.
    if " " not in kw and len(kw) <= 12:
        compact_kw = kw.replace(" ", "")
        for corpus in corpora:
            compact_corpus = re.sub(r"\s+", "", corpus)
            if compact_kw and compact_kw in compact_corpus:
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
    elif not want_identity:
        id_reason = (
            "Identitas tidak diuji karena nama referensi kosong — "
            "`document_matched` hanya mengandalkan profil keyword."
        )
    elif identity_pass is True:
        id_reason = (
            f"Identitas lolos: skor gabungan nama ≥ ambang {identity_min_score:.0f} "
            "(rata-rata token_sort_ratio, WRatio, partial_ratio pada nama vs referensi)."
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
            ext = identity.get("extracted_normalized") or ""
            exp = identity.get("expected_normalized") or ""
            id_reason = (
                "Identitas tidak lolos: nama yang dipakai dari OCR "
                f"«{raw_cand}» (dinormalisasi: «{ext}»; cara ambil: {method_human}) "
                f"dibandingkan dengan referensi «{expected_name_display}» (dinormalisasi: «{exp}»). "
                f"Skor gabungan {isc:.1f} di bawah ambang {identity_min_score:.0f}. "
                f"Rincian sub-skor — token_sort_ratio: {ts:.1f}, WRatio: {wr:.1f}, partial_ratio: {pr:.1f}. "
                "Skor gabungan adalah rata-rata ketiga angka tersebut."
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
        hit_ratio = 0.0 if item.get("skipped") else float(item["best_score"]) / 100.0
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
        if re.search(r"\b\d{16}\b", ocr_n):
            bonus += 0.06
        if re.search(r"\bnik\b", ocr_n) or "nik" in compact:
            bonus += 0.04
        return min(0.10, bonus)
    if pid == "npwp":
        if re.search(r"\bnpwp\b", ocr_n) or "npwp" in compact:
            return 0.06
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
        return min(0.10, bonus)
    if pid == "jkn":
        if _ocr_has_keyword_signal(ocr_n, "info peserta"):
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
    if pid == "npwp" and re.search(r"\bnpwp\b", ocr_n):
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
    aggregate_min_pass_ratio: float = 0.7,
    identity_min_score: float = 65.0,
    mistral_annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    - Tipe dokumen: rata-rata skor keyword vs OCR penuh >= aggregate_min_pass_ratio.
    - Identitas (jika nama referensi diisi): penanda nama di OCR; jika gagal, pilih segmen
      yang mirip nama orang lalu yang paling tinggi skor identitas vs nama referensi.
    - document_matched = document_type_pass AND (identitas tidak dicek | identity_pass).
    """
    keywords = keywords or []
    ocr_n = _normalize(ocr_text)
    ratio_req = max(0.0, min(1.0, float(aggregate_min_pass_ratio)))
    id_min = max(0.0, min(100.0, float(identity_min_score)))
    profile = (document_profile_id or "").strip().casefold()

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

    document_type_components_count = len(kw_ratios) + len(profile_eval.get("any_groups") or [])

    skip_identity = skip_identity_validation(profile)
    want_identity = bool(expected_name.strip()) and not skip_identity
    name_extraction: dict[str, Any] = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_no_expected_name",
    }
    identity: dict[str, Any] | None = None
    identity_pass: bool | None = None
    identity_skip_reason: str | None = None
    if skip_identity and expected_name.strip():
        identity_skip_reason = (
            f"Identitas tidak diuji untuk profil {profile_label(profile)} — "
            "validasi hanya memeriksa keyword dokumen."
        )
        name_extraction["method"] = "skipped_profile_no_identity"

    if want_identity:
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
            for name in extract_kk_member_names(ocr_text):
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
                scored.append((scv, name, "kk_table_row"))
        jkn_names = extract_jkn_participant_names(ocr_text) if profile == "jkn" else []
        if profile == "jkn" and not ann_name:
            for name in jkn_names:
                scv = float(
                    compare_extracted_identity_scores(name, expected_name)[
                        "identity_combined_score"
                    ]
                )
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

        skip_person_like = (
            (profile == "jkn" and jkn_names)
            or (profile == "iuran" and iuran_names)
            or (profile == "iuran" and iuran_modal_only_state(ocr_text))
            or (profile == "vaksinasi_1" and vax_names)
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
            identity_pass = float(identity["identity_combined_score"]) >= id_min
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

        document_matched = bool(document_type_pass and identity_pass)
    elif skip_identity:
        document_matched = bool(document_type_pass)
    else:
        name_extraction["method"] = "skipped_no_expected_name"
        document_matched = bool(document_type_pass)

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
        ownership_checked=bool(expected_name.strip()) and not skip_identity,
    )

    return {
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
    }
