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
        raw_cand, method = extract_holder_name_candidate(ocr_text, profile)
        if raw_cand is None:
            best_seg: str | None = None
            best_sc = -1.0
            for seg in person_like_segments(ocr_text):
                sc = compare_extracted_identity_scores(seg, expected_name)
                scv = float(sc["identity_combined_score"])
                if scv > best_sc:
                    best_sc = scv
                    best_seg = seg
            if best_seg is not None:
                raw_cand = best_seg
                method = "best_person_like_vs_expected"
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
    }
