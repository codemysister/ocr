"""Perbandingan teks hasil OCR dengan nama referensi dan keyword dokumen (RapidFuzz)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz


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
    threshold: float = 85.0,
) -> NameFuzzyResult:
    """
    Menggabungkan beberapa metrik fuzzy; keputusan `matched` memakai skor terbaik
    (cocok untuk kesalahan OCR, urutan kata berbeda, atau teks OCR lebih panjang).
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
    matched = best >= float(threshold)
    return NameFuzzyResult(
        ocr_normalized=a,
        expected_normalized=b,
        scores=scores,
        best_score=best,
        matched=matched,
    )


def fuzzy_keyword_against_ocr(
    keyword: str,
    ocr_normalized: str,
    *,
    threshold: float,
) -> dict[str, Any]:
    """
    Apakah teks OCR (sudah dinormalisasi) mengandung pola yang fuzzy-mirip keyword.
    Memakai gabungan metrik: substring/sejajar (partial), token, WRatio.
    """
    kw = _normalize_keyword(keyword)
    if not kw:
        return {
            "keyword_raw": keyword,
            "keyword_normalized": kw,
            "scores": {},
            "best_score": 0.0,
            "matched": False,
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
            "matched": False,
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
        "matched": best >= float(threshold),
        "skipped": False,
    }


def validate_document_ocr(
    ocr_text: str,
    *,
    document_type: str = "",
    keywords: list[str] | None = None,
    expected_name: str = "",
    threshold: float = 85.0,
) -> dict[str, Any]:
    """
    Validasi gabungan: tiap keyword fuzzy terhadap OCR penuh; nama memakai compare_ocr_name_to_expected.
    `document_matched`: semua keyword (yang tidak di-skip) lolos dan bagian nama (jika diisi) lolos.
    """
    keywords = keywords or []
    ocr_n = _normalize(ocr_text)

    keyword_results: list[dict[str, Any]] = []
    all_kw_ok = True
    for raw_kw in keywords:
        item = fuzzy_keyword_against_ocr(raw_kw, ocr_n, threshold=threshold)
        keyword_results.append(item)
        if item.get("skipped"):
            continue
        if not item["matched"]:
            all_kw_ok = False

    has_keywords = any(not k.get("skipped") for k in keyword_results)
    if not keywords:
        all_kw_ok = True

    name_result: dict[str, Any] | None = None
    name_ok = True
    if expected_name.strip():
        nr = compare_ocr_name_to_expected(ocr_text, expected_name, threshold=threshold)
        name_result = nr.as_dict()
        name_ok = nr.matched
    else:
        name_ok = True

    if has_keywords:
        document_matched = all_kw_ok and name_ok
    elif expected_name.strip():
        document_matched = name_ok
    else:
        document_matched = False

    return {
        "document_type": (document_type or "").strip(),
        "ocr_normalized": ocr_n,
        "threshold": float(threshold),
        "keywords": keyword_results,
        "keywords_all_matched": all_kw_ok if has_keywords else None,
        "name": name_result,
        "name_matched": name_result["matched"] if name_result is not None else None,
        "document_matched": document_matched,
    }
