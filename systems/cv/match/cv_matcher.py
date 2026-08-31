"""Match CV pada 3 dimensi: nama, pendidikan, pengalaman."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from systems.validation.fuzzy_compare import compare_extracted_identity_scores
from systems.validation.name_extraction import (
    _cv_normalize_spaced_text,
    _looks_like_cv_name_line,
    extract_cv_holder_name,
)

PENDIDIKAN_KEYWORDS: tuple[str, ...] = (
    "pendidikan",
    "education",
    "riwayat pendidikan",
)
PENGALAMAN_KEYWORDS: tuple[str, ...] = (
    "pengalaman",
    "experience",
    "pengalaman kerja",
    "riwayat pekerjaan",
)
PENDIDIKAN_STRUCTURE_MARKERS: tuple[str, ...] = (
    "sekolah",
    "universitas",
    "kuliah",
    "jurusan",
    "fakultas",
    "lulusan",
    "smk",
    "sma",
    "sd negeri",
    "institut",
    "akademi",
    "politeknik",
)
PENGALAMAN_STRUCTURE_MARKERS: tuple[str, ...] = (
    "pengalaman kerja",
    "riwayat pekerjaan",
    "praktik kerja",
    "magang",
    "internship",
    "pkl",
    "work experience",
    "pekerjaan",
    "organisasi",
    "dunia kerja",
    "siap bekerja",
    "latihan kerja",
    "balai latihan",
    "pelatihan kerja",
)

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


def _structure_keyword_pass(text: str, markers: tuple[str, ...]) -> bool:
    low = _norm(text)
    if not low:
        return False
    return any(marker in low for marker in markers)


def _experience_optional_pass(text: str) -> bool:
    """
    CV lulusan baru sering tidak punya section pengalaman — loloskan bila tidak ada
    judul pengalaman tetapi ada sinyal siap kerja / fresh graduate.
    """
    low = _norm(text)
    if not low:
        return False
    if re.search(r"\b(pengalaman|experience|riwayat pekerjaan|work experience)\b", low):
        return False
    return any(
        marker in low
        for marker in (
            "fresh graduate",
            "lulusan baru",
            "lulusan sma",
            "dunia kerja",
            "siap bekerja",
            "tantangan baru",
            "lamaran kerja",
            "mengajukan diri",
            "personalia",
            "latihan kerja",
            "balai latihan",
        )
    )


def _keyword_dimension(
    text: str,
    *,
    base_keywords: tuple[str, ...],
    extra_query: str = "",
    structural_markers: tuple[str, ...] = (),
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
            "structure_pass": False,
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
    structure_pass = _structure_keyword_pass(text, structural_markers)
    passed = best >= KEYWORD_MIN_SCORE or structure_pass
    if structure_pass and best < KEYWORD_MIN_SCORE:
        best = max(best, KEYWORD_MIN_SCORE)
    return {
        "percent": round(best, 1),
        "pass": passed,
        "keywords_checked": keywords,
        "keywords_hit": hit,
        "snippet": text.strip()[:280],
        "structure_pass": structure_pass,
    }


def _best_name_line_for_expected(text: str, expected: str) -> tuple[str | None, str]:
    """Cari baris di teks CV yang paling mirip expected_name (benchmark dataset)."""
    exp_n = _norm(expected)
    if not exp_n:
        return None, ""
    best_line: str | None = None
    best_score = 0.0
    for line in re.split(r"[\n|]+", _cv_normalize_spaced_text(text or "")):
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
    document_text: str = "",
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

    doc_text = (document_text or full_fallback or "").strip()
    nama_text = _chunks_text(chunks, "nama") or doc_text

    best_line, best_method = _best_name_line_for_expected(doc_text, expected)
    extracted, method = extract_cv_holder_name(nama_text, expected_name=expected)
    if not extracted and doc_text and doc_text != nama_text:
        extracted, method = extract_cv_holder_name(doc_text, expected_name=expected)

    if best_line:
        best_identity = compare_extracted_identity_scores(best_line, expected)
        best_score = float(best_identity.get("identity_combined_score") or 0.0)
        if not extracted:
            extracted, method = best_line, best_method
        else:
            cur_identity = compare_extracted_identity_scores(extracted, expected)
            cur_score = float(cur_identity.get("identity_combined_score") or 0.0)
            if best_score > cur_score:
                extracted, method = best_line, best_method

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

    if score < NAME_MIN_SCORE and doc_text:
        alt, alt_method = _best_name_line_for_expected(doc_text, expected)
        if alt:
            alt_identity = compare_extracted_identity_scores(alt, expected)
            alt_score = float(alt_identity.get("identity_combined_score") or 0.0)
            if alt_score > score:
                extracted, method = alt, alt_method
                identity = alt_identity
                score = alt_score

    if score < NAME_MIN_SCORE and doc_text:
        exp_n = _norm(expected)
        doc_n = _norm(_cv_normalize_spaced_text(doc_text))
        presence = float(fuzz.partial_ratio(exp_n, doc_n))
        if presence >= NAME_MIN_SCORE:
            if best_line:
                extracted, method = best_line, best_method or "cv_document_presence"
            else:
                method = "cv_document_presence"
            score = max(score, presence)

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
    document_text: str = "",
) -> dict[str, Any]:
    chunk_text = "\n".join(c.get("content") or "" for c in chunks).strip()
    doc_text = (document_text or chunk_text).strip()

    pendidikan_text = _chunks_text(chunks, "pendidikan") or doc_text
    pengalaman_text = _chunks_text(chunks, "pengalaman") or doc_text

    nama = _name_dimension(
        chunks,
        expected_name=expected_name,
        full_fallback=chunk_text,
        document_text=doc_text,
    )
    pendidikan = _keyword_dimension(
        pendidikan_text,
        base_keywords=PENDIDIKAN_KEYWORDS,
        extra_query=education_query,
        structural_markers=PENDIDIKAN_STRUCTURE_MARKERS,
    )
    pengalaman = _keyword_dimension(
        pengalaman_text,
        base_keywords=PENGALAMAN_KEYWORDS,
        extra_query=experience_query,
        structural_markers=PENGALAMAN_STRUCTURE_MARKERS,
    )
    if not pengalaman["pass"] and _experience_optional_pass(doc_text):
        pengalaman = {
            **pengalaman,
            "pass": True,
            "optional_pass": True,
            "percent": max(float(pengalaman["percent"]), KEYWORD_MIN_SCORE),
        }

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
