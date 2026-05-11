"""Perbandingan teks hasil OCR dengan nama referensi dan keyword dokumen (RapidFuzz)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from systems.validation.name_extraction import extract_holder_name_candidate, person_like_segments


def _normalize(s: str) -> str:
    s = (s or "").strip().casefold()
    return re.sub(r"\s+", " ", s)


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
    a = _normalize(extracted)
    b = _normalize(expected_name)
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


def fuzzy_keyword_against_ocr(keyword: str, ocr_normalized: str) -> dict[str, Any]:
    """
    Skor fuzzy keyword terhadap teks OCR penuh (partial, token, WRatio); tidak ada ambang per keyword.
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

    scores = {
        "partial_ratio": float(fuzz.partial_ratio(kw, ocr_normalized)),
        "token_sort_ratio": float(fuzz.token_sort_ratio(kw, ocr_normalized)),
        "token_set_ratio": float(fuzz.token_set_ratio(kw, ocr_normalized)),
        "wratio": float(fuzz.WRatio(kw, ocr_normalized)),
    }
    best = max(scores.values())
    return {
        "keyword_raw": keyword,
        "keyword_normalized": kw,
        "scores": scores,
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
    "failed": "tidak berhasil menarik nama dari OCR",
    "skipped_no_expected_name": "nama referensi kosong — identitas tidak diuji",
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
    if not want_identity:
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


def validate_document_ocr(
    ocr_text: str,
    *,
    document_type: str = "",
    document_profile_id: str = "",
    keywords: list[str] | None = None,
    expected_name: str = "",
    aggregate_min_pass_ratio: float = 0.7,
    identity_min_score: float = 65.0,
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

    keyword_results: list[dict[str, Any]] = []
    kw_ratios: list[float] = []
    for raw_kw in keywords:
        item = fuzzy_keyword_against_ocr(raw_kw, ocr_n)
        keyword_results.append(item)
        if item.get("skipped"):
            continue
        kw_ratios.append(float(item["best_score"]) / 100.0)

    if not kw_ratios:
        document_type_aggregate_pass_ratio = 0.0
        document_type_pass = False
    else:
        document_type_aggregate_pass_ratio = sum(kw_ratios) / len(kw_ratios)
        document_type_pass = document_type_aggregate_pass_ratio >= ratio_req

    document_type_components_count = len(kw_ratios)

    want_identity = bool(expected_name.strip())
    name_extraction: dict[str, Any] = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_no_expected_name",
    }
    identity: dict[str, Any] | None = None
    identity_pass: bool | None = None

    if want_identity:
        struct_cand, struct_method = extract_holder_name_candidate(ocr_text, profile)
        scored: list[tuple[float, str, str]] = []
        if struct_cand:
            scv = float(
                compare_extracted_identity_scores(struct_cand, expected_name)[
                    "identity_combined_score"
                ]
            )
            scored.append((scv, struct_cand, struct_method))
        for seg in person_like_segments(ocr_text):
            scv = float(
                compare_extracted_identity_scores(seg, expected_name)["identity_combined_score"]
            )
            scored.append((scv, seg, "person_like_segment"))

        by_norm: dict[str, tuple[float, str, str]] = {}
        for scv, text, meth in scored:
            nk = _normalize(text)
            if nk not in by_norm or scv > by_norm[nk][0]:
                by_norm[nk] = (scv, text, meth)

        if by_norm:
            best_scv, raw_cand, method = max(by_norm.values(), key=lambda t: t[0])
        else:
            raw_cand = None
            method = "failed"

        name_extraction = {
            "candidate_raw": raw_cand,
            "candidate_normalized": _normalize(raw_cand) if raw_cand else "",
            "method": method,
        }
        if raw_cand:
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
    )

    return {
        "document_type": (document_type or "").strip(),
        "document_profile_id": profile,
        "ocr_normalized": ocr_n,
        "aggregate_min_pass_ratio": ratio_req,
        "identity_min_score": id_min,
        "document_type_pass": document_type_pass,
        "document_type_aggregate_pass_ratio": round(document_type_aggregate_pass_ratio, 6),
        "document_type_components_count": document_type_components_count,
        "aggregate_pass_ratio": round(document_type_aggregate_pass_ratio, 6),
        "components_count": document_type_components_count,
        "name_extraction": name_extraction,
        "identity": identity,
        "identity_pass": identity_pass,
        "keywords": keyword_results,
        "document_matched": document_matched,
        "explanation": explanation,
    }
