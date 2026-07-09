"""Match CV pada 3 dimensi: nama, pendidikan, pengalaman."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from systems.validation.fuzzy_compare import compare_extracted_identity_scores
from systems.validation.name_extraction import _looks_like_cv_name_line, extract_cv_holder_name

PENDIDIKAN_KEYWORDS: tuple[str, ...] = ("pendidikan", "education")
PENGALAMAN_KEYWORDS: tuple[str, ...] = ("pengalaman", "experience", "pengalaman kerja")

NAME_MIN_SCORE = 65.0
KEYWORD_MIN_SCORE = 70.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().casefold())


def _chunks_text(chunks: list[dict[str, Any]], kind: str) -> str:
    parts: list[str] = []
    for c in chunks:
        if c.get("section_kind") != kind:
            continue
        path = " / ".join(c.get("section_path") or [])
        body = (c.get("content") or "").strip()
        block = f"{path}\n{body}".strip() if path else body
        if block:
            parts.append(block)
    return "\n".join(parts).strip()


def _keyword_dimension(
    text: str,
    *,
    base_keywords: tuple[str, ...],
    extra_query: str = "",
) -> dict[str, Any]:
    keywords = list(base_keywords)
    q = (extra_query or "").strip()
    if q:
        keywords.append(q)
    if not text.strip():
        return {
            "percent": 0.0,
            "pass": False,
            "keywords_checked": keywords,
            "keywords_hit": [],
            "snippet": "",
        }
    text_n = _norm(text)
    best = 0.0
    hit: list[str] = []
    for kw in keywords:
        kn = _norm(kw)
        if not kn:
            continue
        sc = float(fuzz.partial_ratio(kn, text_n))
        if sc > best:
            best = sc
        if sc >= KEYWORD_MIN_SCORE:
            hit.append(kw)
    return {
        "percent": round(best, 1),
        "pass": best >= KEYWORD_MIN_SCORE,
        "keywords_checked": keywords,
        "keywords_hit": hit,
        "snippet": text.strip()[:280],
    }


def _best_name_line_for_expected(text: str, expected: str) -> tuple[str | None, str]:
    """Cari baris di teks CV yang paling mirip expected_name (benchmark dataset)."""
    exp_n = _norm(expected)
    if not exp_n:
        return None, ""
    best_line: str | None = None
    best_score = 0.0
    for line in re.split(r"[\n|]+", text or ""):
        line = re.sub(r"^[-•*|]+\s*", "", (line or "").strip()).strip()
        if not _looks_like_cv_name_line(line, expected_name=expected):
            continue
        sc = float(fuzz.WRatio(_norm(line), exp_n))
        if sc > best_score:
            best_score = sc
            best_line = line
    if best_line and best_score >= NAME_MIN_SCORE:
        return best_line, "cv_best_expected_line"
    return None, ""


def _name_dimension(
    chunks: list[dict[str, Any]],
    *,
    expected_name: str,
    full_fallback: str,
) -> dict[str, Any]:
    expected = (expected_name or "").strip()
    if not expected:
        return {
            "percent": None,
            "pass": None,
            "skipped": True,
            "expected": "",
            "extracted": None,
            "method": None,
        }

    nama_text = _chunks_text(chunks, "nama") or full_fallback
    extracted, method = extract_cv_holder_name(nama_text, expected_name=expected)
    if not extracted and full_fallback and full_fallback != nama_text:
        extracted, method = extract_cv_holder_name(full_fallback, expected_name=expected)

    if not extracted:
        extracted, method = _best_name_line_for_expected(full_fallback, expected)

    if not extracted:
        return {
            "percent": 0.0,
            "pass": False,
            "skipped": False,
            "expected": expected,
            "extracted": None,
            "method": method or "failed",
        }

    identity = compare_extracted_identity_scores(extracted, expected)
    score = float(identity.get("identity_combined_score") or 0.0)

    if score < NAME_MIN_SCORE:
        alt, alt_method = _best_name_line_for_expected(full_fallback, expected)
        if alt:
            alt_identity = compare_extracted_identity_scores(alt, expected)
            alt_score = float(alt_identity.get("identity_combined_score") or 0.0)
            if alt_score > score:
                extracted, method = alt, alt_method
                identity = alt_identity
                score = alt_score

    return {
        "percent": round(score, 1),
        "pass": score >= NAME_MIN_SCORE,
        "skipped": False,
        "expected": expected,
        "extracted": extracted,
        "method": method,
        "scores": identity.get("scores"),
    }


def match_cv_chunks(
    chunks: list[dict[str, Any]],
    *,
    expected_name: str = "",
    education_query: str = "",
    experience_query: str = "",
) -> dict[str, Any]:
    full_text = "\n".join(c.get("content") or "" for c in chunks).strip()

    pendidikan_text = _chunks_text(chunks, "pendidikan") or full_text
    pengalaman_text = _chunks_text(chunks, "pengalaman") or full_text

    nama = _name_dimension(chunks, expected_name=expected_name, full_fallback=full_text)
    pendidikan = _keyword_dimension(
        pendidikan_text,
        base_keywords=PENDIDIKAN_KEYWORDS,
        extra_query=education_query,
    )
    pengalaman = _keyword_dimension(
        pengalaman_text,
        base_keywords=PENGALAMAN_KEYWORDS,
        extra_query=experience_query,
    )

    percents: list[float] = [pendidikan["percent"], pengalaman["percent"]]
    if nama.get("percent") is not None:
        percents.append(float(nama["percent"]))

    overall = round(sum(percents) / len(percents), 1) if percents else 0.0

    gates: list[bool] = [pendidikan["pass"], pengalaman["pass"]]
    if nama.get("pass") is not None:
        gates.append(bool(nama["pass"]))

    matched = all(gates)

    summary_parts = []
    if nama.get("skipped"):
        summary_parts.append("nama tidak dicek")
    else:
        summary_parts.append(f"nama {'✓' if nama.get('pass') else '✗'} ({nama.get('percent')}%)")
    summary_parts.append(
        f"pendidikan {'✓' if pendidikan['pass'] else '✗'} ({pendidikan['percent']}%)"
    )
    summary_parts.append(
        f"pengalaman {'✓' if pengalaman['pass'] else '✗'} ({pengalaman['percent']}%)"
    )

    return {
        "matched": matched,
        "overall_percent": overall,
        "dimensions": {
            "nama": nama,
            "pendidikan": pendidikan,
            "pengalaman": pengalaman,
        },
        "summary": " · ".join(summary_parts),
    }
