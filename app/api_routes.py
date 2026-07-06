"""API publik untuk integrasi frontend (React, Firebase Hosting, dll.)."""

from __future__ import annotations

import base64
import os
from typing import Literal

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from systems.ocr.fast_runner import list_pp_ocr_tiers, run_paddleocr_fast
from systems.ocr.mistral_annotation import parse_document_annotation
from systems.ocr.mistral_runner import run_mistral_ocr
from systems.ocr.runtime_hint import ocr_inference_unavailable_detail
from systems.ocr.vl_runner import run_paddleocr_vl
from systems.observability.last_tuning_log import (
    log_safe_failure,
    summarize_text_fields,
    summarize_validation_result,
    write_last_tuning_log,
)
from systems.preprocessing.pipeline import decode_image_bytes_bgr, preprocess_image_bytes
from systems.validation.document_profiles import (
    is_cv_ingest_profile,
    is_image_only_profile,
    list_supported_document_types,
    resolve_keywords,
)
from systems.validation.fuzzy_compare import validate_document_ocr
from systems.validation.portrait_photo_validate import validate_foto_profile

router = APIRouter(prefix="/api/v1", tags=["api"])

OcrMode = Literal["mistral", "fast", "vl"]
PpOcrTier = Literal["balanced", "medium", "small", "tiny"]


def _cors_note() -> str:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return "Semua origin diizinkan (dev). Untuk production set CORS_ORIGINS ke domain Firebase."
    return f"Origin diizinkan: {raw}"


@router.get("")
def api_info() -> dict:
    """Daftar endpoint untuk integrasi frontend."""
    return {
        "version": "1.0.0",
        "cors": _cors_note(),
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "preprocess": {
                "method": "POST",
                "path": "/api/v1/preprocess",
                "content_type": "multipart/form-data",
                "fields": {"file": "binary (required)"},
                "query": {"format": "image | json (default image)"},
            },
            "pipeline": {
                "method": "POST",
                "path": "/api/v1/pipeline",
                "content_type": "multipart/form-data",
                "fields": {
                    "file": "binary (required)",
                    "document_type": "string (required, e.g. KTP)",
                    "expected_name": "string (optional)",
                },
                "query": {
                    "ocr_mode": "fast (default PP-OCRv6) | mistral | vl",
                    "pp_ocr_tier": "balanced | medium | small | tiny — hanya ocr_mode=fast",
                    "include_preprocessed_image": "true | false (default false)",
                    "full_json": "true | false — hanya ocr_mode=vl",
                },
                "description": "Preprocess → OCR (pilihan engine) → validasi dokumen dalam satu request. Untuk cv: ingest + opsional search.",
                "ocr_modes": {
                    "fast": "Lokal PP-OCRv6 (default pipeline, pp_ocr_tier)",
                    "mistral": "Cloud Mistral OCR + document_annotation (opsional)",
                    "vl": "Lokal PaddleOCR-VL-1.6 (layout + markdown)",
                },
                "pp_ocr_tiers": list_pp_ocr_tiers(),
                "response_fields": {
                    "verdict.is_own_document": "true | false | null — milik user saat ini?",
                    "verdict.document_type_current_label": "jenis dokumen terdeteksi dari OCR",
                    "verdict.summary": "ringkasan bahasa Indonesia",
                },
            },
            "ocr_mistral": {
                "method": "POST",
                "path": "/systems/ocr/api/v1/ocr-mistral",
                "content_type": "multipart/form-data",
                "fields": {"file": "binary (required)"},
            },
            "ocr_fast": {
                "method": "POST",
                "path": "/systems/ocr/api/v1/ocr-fast",
                "query": {"pp_ocr_tier": "balanced | medium | small | tiny"},
            },
            "ocr_vl": {
                "method": "POST",
                "path": "/systems/ocr/api/v1/ocr",
            },
            "validate_document": {
                "method": "POST",
                "path": "/systems/validation/api/v1/validate-document",
                "content_type": "application/json",
                "body": {
                    "ocr_text": "string (required)",
                    "document_type": "string (required)",
                    "expected_name": "string (optional)",
                    "mistral_annotation": "object (opsional, dari Mistral OCR)",
                },
            },
            "validate_foto_profile": {
                "method": "POST",
                "path": "/systems/validation/api/v1/validate-foto-profile",
                "content_type": "multipart/form-data",
                "fields": {
                    "file": "binary (required)",
                    "document_type": "string (default foto_profile)",
                    "expected_name": "string (optional)",
                },
                "description": "Validasi foto profil berlatar biru (tanpa OCR).",
            },
            "compare_names": {
                "method": "POST",
                "path": "/systems/validation/api/v1/compare-names",
                "content_type": "application/json",
            },
        },
        "document_types": list_supported_document_types(),
        "docs": "/docs",
    }


def _run_ocr(
    png_bytes: bytes,
    ocr_mode: OcrMode,
    *,
    full_json: bool = False,
    pp_ocr_tier: str | None = None,
) -> dict:
    try:
        if ocr_mode == "mistral":
            return run_mistral_ocr(png_bytes)
        if ocr_mode == "vl":
            return run_paddleocr_vl(png_bytes, include_full_json=full_json)
        return run_paddleocr_fast(png_bytes, pp_ocr_tier=pp_ocr_tier)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e).lower()
        if "mistral_api_key" in msg or "pasang sdk mistral" in msg:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MISTRAL_OCR_UNAVAILABLE",
                    "message": str(e),
                    "install": "pip install -r requirements-mistral.txt",
                    "env": ["MISTRAL_API_KEY", "MISTRAL_OCR_MODEL"],
                },
            ) from e
        if any(x in msg for x in ("paddlepaddle", "dependency", "engine", "unavailable", "paddleocr")):
            paddle_mode = "vl" if ocr_mode == "vl" else "fast"
            det = ocr_inference_unavailable_detail(mode=paddle_mode, exc=e)
            raise HTTPException(status_code=503, detail=det) from e
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _ocr_text_from_payload(payload: dict) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    markdown = payload.get("markdown")
    if isinstance(markdown, str):
        return markdown.strip()
    return ""


@router.post("/pipeline")
async def api_pipeline(
    file: UploadFile = File(..., description="Gambar dokumen"),
    document_type: str = Form(..., description="Jenis dokumen, mis. KTP / NPWP"),
    expected_name: str = Form("", description="Nama referensi (opsional)"),
    ocr_mode: OcrMode = Query(
        "fast",
        description="Engine OCR: fast (default, PP-OCRv6 lokal), mistral (cloud), vl (PaddleOCR-VL)",
    ),
    pp_ocr_tier: PpOcrTier = Query(
        "medium",
        description="Tier PP-OCRv6 jika ocr_mode=fast (default medium = akurasi maksimal)",
    ),
    include_preprocessed_image: bool = Query(
        False,
        description="Sertakan image_base64 hasil preprocess di respons",
    ),
    full_json: bool = Query(False, description="Untuk ocr_mode=vl: lampirkan result_json lengkap"),
    cv_search_query: str = Query(
        "",
        description="(Legacy) Alias expected_name untuk match CV",
    ),
    cv_education_query: str = Query(
        "",
        description="Hanya document_type=cv: kata kunci tambahan section pendidikan",
    ),
    cv_experience_query: str = Query(
        "",
        description="Hanya document_type=cv: kata kunci tambahan section pengalaman",
    ),
) -> JSONResponse:
    """
    Pipeline lengkap: preprocess gambar → OCR (pilihan model) → validasi dokumen.

    Cocok dipanggil dari React (Firebase Hosting) dengan satu fetch multipart.
    """
    path = "/api/v1/pipeline"
    raw = await file.read()
    doc_type = document_type.strip()
    name_ref = expected_name.strip()

    if not raw:
        log_safe_failure(
            subsystem="pipeline",
            method="POST",
            path=path,
            http_status=400,
            detail="File kosong.",
        )
        raise HTTPException(status_code=400, detail="File kosong.")
    if not doc_type:
        log_safe_failure(
            subsystem="pipeline",
            method="POST",
            path=path,
            http_status=400,
            detail="document_type kosong.",
        )
        raise HTTPException(status_code=400, detail="document_type tidak boleh kosong.")

    resolved = resolve_keywords(doc_type)
    if not resolved:
        detail = {
            "message": "Jenis dokumen tidak dikenal.",
            "document_type": doc_type,
            "supported": list_supported_document_types(),
        }
        log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=detail)
        raise HTTPException(status_code=400, detail=detail)

    canonical_id, keywords = resolved

    if is_cv_ingest_profile(canonical_id):
        try:
            from systems.cv.ingest.pipeline import ingest_cv_bytes
        except ImportError as e:
            detail = {
                "code": "CV_SUBSYSTEM_UNAVAILABLE",
                "message": "Subsistem CV belum terpasang.",
                "install": "pip install -r requirements-cv.txt",
                "opensearch": "docker compose -f docker-compose.cv.yml up -d",
            }
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=503, detail=detail)
            raise HTTPException(status_code=503, detail=detail) from e

        filename = file.filename or "cv"
        cv_name = (name_ref or cv_search_query).strip()
        try:
            ingest_result = ingest_cv_bytes(
                raw,
                filename=filename,
                expected_name=cv_name,
                education_query=cv_education_query.strip(),
                experience_query=cv_experience_query.strip(),
            )
        except ValueError as e:
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            from systems.cv.index.opensearch_client import (
                OpenSearchUnavailableError,
                is_opensearch_connection_error,
                opensearch_unavailable_detail,
            )

            if isinstance(e, OpenSearchUnavailableError) or is_opensearch_connection_error(e):
                detail = opensearch_unavailable_detail(exc=e)
                log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=503, detail=detail)
                raise HTTPException(status_code=503, detail=detail) from e
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=500, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

        cv_match = ingest_result.get("cv_match") or {}
        matched = bool(cv_match.get("matched", True))

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
        }

        write_last_tuning_log(
            {
                "success": True,
                "subsystem": "pipeline",
                "method": "POST",
                "path": path,
                "input_bytes": len(raw),
                "query": {
                    "expected_name": cv_name or None,
                    "cv_education_query": cv_education_query or None,
                    "cv_experience_query": cv_experience_query or None,
                },
                "request": {"document_type": doc_type, "filename": filename},
                "preprocess": payload["preprocess"],
                "ocr": {"skipped": True, "reason": "cv"},
                "cv_ingest": {
                    "doc_id": ingest_result.get("doc_id"),
                    "chunk_count": ingest_result.get("chunk_count"),
                    "parse_mode": ingest_result.get("parse_mode"),
                },
                "cv_match": {
                    "matched": matched,
                    "overall_percent": cv_match.get("overall_percent"),
                },
            }
        )
        return JSONResponse(content=jsonable_encoder(payload))

    if is_image_only_profile(canonical_id):
        try:
            bgr, decode_meta = decode_image_bytes_bgr(raw)
        except Exception as e:
            log_safe_failure(
                subsystem="pipeline",
                method="POST",
                path=path,
                http_status=400,
                detail=str(e),
            )
            raise HTTPException(status_code=400, detail=str(e)) from e

        validation_detail = validate_foto_profile(
            bgr,
            document_type=doc_type,
            expected_name=name_ref,
        )
        analysis_bgr = validation_detail.pop("_analysis_bgr", None)
        payload: dict = {
            "success": True,
            "valid": validation_detail.get("document_matched"),
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
        }
        if include_preprocessed_image:
            preview_bgr = analysis_bgr if analysis_bgr is not None else bgr
            ok, encoded = cv2.imencode(".png", preview_bgr)
            if ok:
                payload["preprocessed_image"] = {
                    "mime": "image/png",
                    "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }

        write_last_tuning_log(
            {
                "success": True,
                "subsystem": "pipeline",
                "method": "POST",
                "path": path,
                "input_bytes": len(raw),
                "query": {
                    "ocr_mode": ocr_mode,
                    "pp_ocr_tier": pp_ocr_tier if ocr_mode == "fast" else None,
                    "include_preprocessed_image": include_preprocessed_image,
                    "full_json": full_json,
                },
                "request": {
                    "document_type": doc_type,
                    "expected_name_chars": len(name_ref),
                },
                "preprocess": payload["preprocess"],
                "ocr": {"skipped": True, "reason": "foto_profile"},
                "validation": summarize_validation_result(payload["validation"]),
            }
        )
        return JSONResponse(content=jsonable_encoder(payload))

    try:
        png_bytes, pre_meta = preprocess_image_bytes(raw, document_profile_id=canonical_id)
    except ValueError as e:
        log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    tier_arg = pp_ocr_tier if ocr_mode == "fast" else None
    ocr_payload = _run_ocr(png_bytes, ocr_mode, full_json=full_json, pp_ocr_tier=tier_arg)
    ocr_text = _ocr_text_from_payload(ocr_payload)
    if not ocr_text:
        log_safe_failure(
            subsystem="pipeline",
            method="POST",
            path=path,
            http_status=400,
            detail="OCR tidak menghasilkan teks.",
        )
        raise HTTPException(status_code=400, detail="OCR tidak menghasilkan teks.")

    mistral_ann: dict | None = None
    if ocr_mode == "mistral":
        mistral_ann = ocr_payload.get("document_annotation_parsed")
        if mistral_ann is None:
            mistral_ann = parse_document_annotation(ocr_payload.get("document_annotation"))

    validation_detail = validate_document_ocr(
        ocr_text,
        document_type=doc_type,
        document_profile_id=canonical_id,
        keywords=keywords,
        expected_name=name_ref,
        mistral_annotation=mistral_ann,
    )

    payload: dict = {
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
    }
    if include_preprocessed_image:
        payload["preprocessed_image"] = {
            "mime": "image/png",
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
        }

    write_last_tuning_log(
        {
            "success": True,
            "subsystem": "pipeline",
            "method": "POST",
            "path": path,
            "input_bytes": len(raw),
            "query": {
                "ocr_mode": ocr_mode,
                "pp_ocr_tier": tier_arg,
                "include_preprocessed_image": include_preprocessed_image,
                "full_json": full_json,
            },
            "request": {
                "document_type": doc_type,
                "expected_name_chars": len(name_ref),
            },
            "preprocess": pre_meta,
            "ocr": summarize_text_fields(ocr_payload),
            "validation": summarize_validation_result(payload["validation"]),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))
