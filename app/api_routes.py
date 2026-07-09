"""API publik untuk integrasi frontend (React, Firebase Hosting, dll.)."""

from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.dataset_benchmark import (
    BenchmarkConfig,
    BenchmarkSelection,
    dataset_root,
    list_dataset_folders,
    resolve_dataset_file,
    run_benchmark,
)
from app.pipeline_runner import OcrMode, PipelineResult, run_pipeline_bytes
from systems.ocr.fast_runner import list_pp_ocr_tiers
from systems.ocr.runtime_hint import ocr_inference_unavailable_detail
from systems.observability.last_tuning_log import (
    log_safe_failure,
    summarize_text_fields,
    summarize_validation_result,
    write_last_tuning_log,
)
from systems.validation.document_profiles import list_supported_document_types, resolve_keywords

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
                    "enable_preprocess": "true | false (default false) — crop/rotate sebelum OCR",
                    "skip_passthrough": "true | false (default false) — raw upload ke OCR, tanpa resize/grayscale",
                    "full_json": "true | false — hanya ocr_mode=vl",
                },
                "description": "OCR (pilihan engine) → validasi dokumen; preprocessing opsional (enable_preprocess=true).",
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
            "dataset_types": {
                "method": "GET",
                "path": "/api/v1/dataset/types",
                "description": "Daftar folder dataset + pemetaan document_type.",
            },
            "dataset_file": {
                "method": "GET",
                "path": "/api/v1/dataset/file",
                "query": {"folder": "string", "file": "string"},
                "description": "Unduh/preview file dataset (untuk review benchmark).",
            },
            "dataset_benchmark": {
                "method": "POST",
                "path": "/api/v1/dataset/benchmark",
                "content_type": "application/json",
                "description": "Benchmark pipeline terhadap file dataset (response NDJSON stream).",
                "body_note": "Per selection: pakai limit/offset batch, atau isi `files` untuk uji file spesifik.",
            },
        },
        "document_types": list_supported_document_types(),
        "docs": "/docs",
    }


def _pipeline_http_error(result: PipelineResult) -> HTTPException:
    """Map PipelineResult gagal ke HTTPException."""
    kind = result.error_kind or ""
    detail = result.error
    if kind == "unknown_document_type":
        detail = {
            "message": result.error,
            "supported": list_supported_document_types(),
        }
    elif kind == "cv_unavailable":
        detail = {
            "code": "CV_SUBSYSTEM_UNAVAILABLE",
            "message": result.error,
            "install": "pip install -r requirements-cv.txt",
            "opensearch": "docker compose -f docker-compose.cv.yml up -d",
        }
    elif kind == "mistral_unavailable":
        detail = {
            "code": "MISTRAL_OCR_UNAVAILABLE",
            "message": result.error,
            "install": "pip install -r requirements-mistral.txt",
            "env": ["MISTRAL_API_KEY", "MISTRAL_OCR_MODEL"],
        }
    elif kind == "ocr_unavailable":
        detail = ocr_inference_unavailable_detail(mode="fast", exc=RuntimeError(result.error or ""))
    elif kind == "opencv_unavailable":
        from systems.cv.index.opensearch_client import opensearch_unavailable_detail

        detail = opensearch_unavailable_detail(exc=RuntimeError(result.error or ""))
    return HTTPException(status_code=result.http_status_hint, detail=detail)


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
    enable_preprocess: bool = Query(
        False,
        description="Jalankan preprocessing gambar sebelum OCR (crop, rotate, dll.)",
    ),
    skip_passthrough: bool = Query(
        False,
        description="Lewati passthrough ringan; kirim bytes upload langsung ke OCR (hanya bila enable_preprocess=false)",
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
    Pipeline lengkap: OCR (pilihan model) → validasi dokumen. Preprocess opsional (`enable_preprocess=true`).

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

    result = run_pipeline_bytes(
        raw,
        document_type=doc_type,
        expected_name=(name_ref or cv_search_query).strip(),
        filename=file.filename or "upload",
        ocr_mode=ocr_mode,
        pp_ocr_tier=pp_ocr_tier if ocr_mode == "fast" else None,
        include_preprocessed_image=include_preprocessed_image,
        enable_preprocess=enable_preprocess,
        skip_passthrough=skip_passthrough,
        full_json=full_json,
        cv_education_query=cv_education_query.strip(),
        cv_experience_query=cv_experience_query.strip(),
    )

    if not result.ok:
        log_safe_failure(
            subsystem="pipeline",
            method="POST",
            path=path,
            http_status=result.http_status_hint,
            detail=result.error,
        )
        raise _pipeline_http_error(result)

    payload = result.payload or {}
    tier_arg = pp_ocr_tier if ocr_mode == "fast" else None

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
                "enable_preprocess": enable_preprocess,
                "skip_passthrough": skip_passthrough,
                "full_json": full_json,
                "expected_name": (name_ref or cv_search_query).strip() or None,
                "cv_education_query": cv_education_query or None,
                "cv_experience_query": cv_experience_query or None,
            },
            "request": {
                "document_type": doc_type,
                "expected_name_chars": len(name_ref),
                "filename": file.filename,
            },
            "preprocess": payload.get("preprocess"),
            "ocr": (
                {"skipped": True, "reason": payload.get("validation_mode")}
                if payload.get("ocr") is None
                else summarize_text_fields(payload.get("ocr") or {})
            ),
            "validation": summarize_validation_result(payload.get("validation") or {}),
            "timing": payload.get("timing"),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))


class DatasetSelectionBody(BaseModel):
    folder: str
    enabled: bool = False
    limit: int = Field(default=20, ge=0, le=500)
    offset: int = Field(default=0, ge=0, le=10000)
    files: list[str] = Field(
        default_factory=list,
        description="Daftar nama file spesifik (abaikan limit/offset bila diisi). Boleh `folder/file.jpg`.",
    )


class DatasetBenchmarkBody(BaseModel):
    selections: list[DatasetSelectionBody]
    ocr_mode: OcrMode = "fast"
    pp_ocr_tier: PpOcrTier = "medium"
    use_expected_name: bool = True
    enable_preprocess: bool = False
    skip_passthrough: bool = False


@router.get("/dataset/types")
def api_dataset_types() -> dict:
    """Daftar subfolder dataset beserta pemetaan document_type."""
    folders = list_dataset_folders()
    return {
        "dataset_root": str(dataset_root()),
        "folders": [
            {
                "folder": f.folder,
                "file_count": f.file_count,
                "document_type": f.document_type,
                "supported": f.supported,
                "label": f.label,
            }
            for f in folders
        ],
    }


@router.get("/dataset/file")
def api_dataset_file(
    folder: str = Query(..., description="Nama subfolder dataset, mis. ktp"),
    file: str = Query(..., description="Nama file, mis. ktp_Nama_3216....jpg"),
) -> FileResponse:
    """Serve file dataset untuk preview di UI benchmark."""
    try:
        path = resolve_dataset_file(folder.strip(), file.strip())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }.get(path.suffix.casefold(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/dataset/benchmark")
def api_dataset_benchmark(body: DatasetBenchmarkBody) -> StreamingResponse:
    """Jalankan benchmark dataset; respons stream NDJSON."""
    config = BenchmarkConfig(
        selections=[
            BenchmarkSelection(
                folder=s.folder,
                enabled=s.enabled,
                limit=s.limit,
                offset=s.offset,
                files=list(s.files or []),
            )
            for s in body.selections
        ],
        ocr_mode=body.ocr_mode,
        pp_ocr_tier=body.pp_ocr_tier,
        use_expected_name=body.use_expected_name,
        enable_preprocess=body.enable_preprocess,
        skip_passthrough=body.skip_passthrough,
    )

    def stream():
        yield from run_benchmark(config)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
