"""Pipeline preprocess → OCR → validasi (dipakai HTTP API & benchmark dataset)."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2

from systems.ocr.fast_runner import run_paddleocr_fast
from systems.ocr.mistral_annotation import parse_document_annotation
from systems.ocr.mistral_runner import run_mistral_ocr
from systems.ocr.vl_runner import run_paddleocr_vl
from systems.preprocessing.pipeline import decode_image_bytes_bgr, preprocess_image_bytes
from systems.validation.document_profiles import (
    is_cv_ingest_profile,
    is_image_only_profile,
    resolve_keywords,
)
from systems.validation.fuzzy_compare import validate_document_ocr
from systems.validation.portrait_photo_validate import validate_foto_profile

OcrMode = Literal["mistral", "fast", "vl"]


@dataclass
class PipelineTiming:
    preprocess_s: float = 0.0
    ocr_s: float = 0.0
    validation_s: float = 0.0
    total_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess": round(self.preprocess_s, 3),
            "ocr": round(self.ocr_s, 3),
            "validation": round(self.validation_s, 3),
            "total": round(self.total_s, 3),
        }


@dataclass
class PipelineResult:
    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    document_matched: bool | None = None

    @property
    def http_status_hint(self) -> int:
        if self.error_kind == "empty_file":
            return 400
        if self.error_kind == "unknown_document_type":
            return 400
        if self.error_kind == "cv_unavailable":
            return 503
        if self.error_kind == "opencv_unavailable":
            return 503
        if self.error_kind in ("ocr_unavailable", "mistral_unavailable"):
            return 503
        if self.error_kind == "empty_ocr":
            return 400
        return 500


def _ocr_text_from_payload(payload: dict) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    markdown = payload.get("markdown")
    if isinstance(markdown, str):
        return markdown.strip()
    return ""


def _run_ocr_inner(
    png_bytes: bytes,
    ocr_mode: OcrMode,
    *,
    full_json: bool = False,
    pp_ocr_tier: str | None = None,
) -> dict:
    if ocr_mode == "mistral":
        return run_mistral_ocr(png_bytes)
    if ocr_mode == "vl":
        return run_paddleocr_vl(png_bytes, include_full_json=full_json)
    return run_paddleocr_fast(png_bytes, pp_ocr_tier=pp_ocr_tier)


def _ocr_timing_s(ocr_payload: dict | None, fallback: float) -> float:
    if not ocr_payload:
        return fallback
    timing = ocr_payload.get("timing")
    if isinstance(timing, dict):
        wall = timing.get("total_wall_s")
        if isinstance(wall, (int, float)):
            return float(wall)
    return fallback


def run_pipeline_bytes(
    raw: bytes,
    *,
    document_type: str,
    expected_name: str = "",
    filename: str = "upload",
    ocr_mode: OcrMode = "fast",
    pp_ocr_tier: str | None = "medium",
    include_preprocessed_image: bool = False,
    full_json: bool = False,
    cv_education_query: str = "",
    cv_experience_query: str = "",
) -> PipelineResult:
    """Jalankan pipeline lengkap; tidak raise HTTPException."""
    t_total0 = time.perf_counter()
    timing = PipelineTiming()
    doc_type = document_type.strip()
    name_ref = expected_name.strip()

    if not raw:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="File kosong.",
            error_kind="empty_file",
            timing=timing,
        )
    if not doc_type:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="document_type tidak boleh kosong.",
            error_kind="empty_document_type",
            timing=timing,
        )

    resolved = resolve_keywords(doc_type)
    if not resolved:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="Jenis dokumen tidak dikenal.",
            error_kind="unknown_document_type",
            timing=timing,
        )

    canonical_id, keywords = resolved

    if is_cv_ingest_profile(canonical_id):
        try:
            from systems.cv.ingest.pipeline import ingest_cv_bytes
        except ImportError:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(
                ok=False,
                error="Subsistem CV belum terpasang.",
                error_kind="cv_unavailable",
                timing=timing,
            )

        cv_name = name_ref
        t_cv0 = time.perf_counter()
        try:
            ingest_result = ingest_cv_bytes(
                raw,
                filename=filename,
                expected_name=cv_name,
                education_query=cv_education_query.strip(),
                experience_query=cv_experience_query.strip(),
            )
        except ValueError as e:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="value_error", timing=timing)
        except Exception as e:
            from systems.cv.index.opensearch_client import (
                OpenSearchUnavailableError,
                is_opensearch_connection_error,
            )

            if isinstance(e, OpenSearchUnavailableError) or is_opensearch_connection_error(e):
                timing.total_s = time.perf_counter() - t_total0
                return PipelineResult(
                    ok=False,
                    error=str(e),
                    error_kind="opencv_unavailable",
                    timing=timing,
                )
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="internal", timing=timing)

        timing.validation_s = time.perf_counter() - t_cv0
        cv_match = ingest_result.get("cv_match") or {}
        matched = bool(cv_match.get("matched", True))
        timing.total_s = time.perf_counter() - t_total0

        payload = {
            "success": True,
            "valid": matched,
            "ocr_mode": None,
            "pp_ocr_tier": None,
            "validation_mode": "cv",
            "preprocess": {"skipped": True, "reason": "cv_ingest"},
            "ocr": None,
            "validation": {
                "document_profile_id": canonical_id,
                "keywords_from_profile": keywords,
                "document_matched": matched,
            },
            "cv_ingest": ingest_result,
            "cv_match": cv_match,
            "verdict": {
                "summary": cv_match.get("summary")
                or (
                    f"CV terindeks: {ingest_result.get('chunk_count', 0)} chunk "
                    f"({ingest_result.get('parse_mode', '?')})."
                ),
                "is_own_document": (
                    cv_match.get("dimensions", {}).get("nama", {}).get("pass")
                    if cv_name
                    else None
                ),
                "document_type_current": canonical_id,
                "document_type_current_label": "CV",
            },
            "is_own_document": None,
            "document_type_current": canonical_id,
            "document_type_current_label": "CV",
            "timing": timing.as_dict(),
        }
        return PipelineResult(
            ok=True,
            payload=payload,
            timing=timing,
            document_matched=matched,
        )

    if is_image_only_profile(canonical_id):
        t_pre0 = time.perf_counter()
        try:
            bgr, decode_meta = decode_image_bytes_bgr(raw)
        except Exception as e:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="decode_error", timing=timing)
        timing.preprocess_s = time.perf_counter() - t_pre0

        t_val0 = time.perf_counter()
        validation_detail = validate_foto_profile(
            bgr,
            document_type=doc_type,
            expected_name=name_ref,
        )
        timing.validation_s = time.perf_counter() - t_val0
        analysis_bgr = validation_detail.pop("_analysis_bgr", None)
        matched = bool(validation_detail.get("document_matched"))

        payload = {
            "success": True,
            "valid": matched,
            "ocr_mode": None,
            "pp_ocr_tier": None,
            "validation_mode": "image",
            "preprocess": {"skipped": True, "reason": "image_only_profile", **decode_meta},
            "ocr": None,
            "validation": {
                "document_profile_id": canonical_id,
                "keywords_from_profile": keywords,
                **validation_detail,
            },
            "verdict": validation_detail.get("verdict"),
            "is_own_document": validation_detail.get("is_own_document"),
            "document_type_current": validation_detail.get("document_type_current"),
            "document_type_current_label": validation_detail.get("document_type_current_label"),
            "timing": timing.as_dict(),
        }
        if include_preprocessed_image:
            preview_bgr = analysis_bgr if analysis_bgr is not None else bgr
            ok, encoded = cv2.imencode(".png", preview_bgr)
            if ok:
                payload["preprocessed_image"] = {
                    "mime": "image/png",
                    "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }

        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=True,
            payload=payload,
            timing=timing,
            document_matched=matched,
        )

    t_pre0 = time.perf_counter()
    try:
        png_bytes, pre_meta = preprocess_image_bytes(raw, document_profile_id=canonical_id)
    except ValueError as e:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind="preprocess_error", timing=timing)
    timing.preprocess_s = time.perf_counter() - t_pre0

    tier_arg = pp_ocr_tier if ocr_mode == "fast" else None
    t_ocr0 = time.perf_counter()
    try:
        ocr_payload = _run_ocr_inner(
            png_bytes,
            ocr_mode,
            full_json=full_json,
            pp_ocr_tier=tier_arg,
        )
    except ValueError as e:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind="ocr_value_error", timing=timing)
    except RuntimeError as e:
        msg = str(e).lower()
        kind = "ocr_unavailable"
        if "mistral_api_key" in msg or "pasang sdk mistral" in msg:
            kind = "mistral_unavailable"
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind=kind, timing=timing)
    except Exception as e:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind="ocr_error", timing=timing)

    timing.ocr_s = _ocr_timing_s(ocr_payload, time.perf_counter() - t_ocr0)
    ocr_text = _ocr_text_from_payload(ocr_payload)
    if not ocr_text:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="OCR tidak menghasilkan teks.",
            error_kind="empty_ocr",
            timing=timing,
        )

    mistral_ann: dict | None = None
    if ocr_mode == "mistral":
        mistral_ann = ocr_payload.get("document_annotation_parsed")
        if mistral_ann is None:
            mistral_ann = parse_document_annotation(ocr_payload.get("document_annotation"))

    t_val0 = time.perf_counter()
    validation_detail = validate_document_ocr(
        ocr_text,
        document_type=doc_type,
        document_profile_id=canonical_id,
        keywords=keywords,
        expected_name=name_ref,
        mistral_annotation=mistral_ann,
    )
    timing.validation_s = time.perf_counter() - t_val0
    matched = bool(validation_detail.get("document_matched"))
    timing.total_s = time.perf_counter() - t_total0

    payload = {
        "success": True,
        "ocr_mode": ocr_mode,
        "pp_ocr_tier": tier_arg if ocr_mode == "fast" else None,
        "preprocess": pre_meta,
        "ocr": ocr_payload,
        "validation": {
            "document_profile_id": canonical_id,
            "keywords_from_profile": keywords,
            **validation_detail,
        },
        "verdict": validation_detail.get("verdict"),
        "is_own_document": validation_detail.get("is_own_document"),
        "document_type_current": validation_detail.get("document_type_current"),
        "document_type_current_label": validation_detail.get("document_type_current_label"),
        "timing": timing.as_dict(),
    }
    if include_preprocessed_image:
        payload["preprocessed_image"] = {
            "mime": "image/png",
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
        }

    return PipelineResult(
        ok=True,
        payload=payload,
        timing=timing,
        document_matched=matched,
    )
