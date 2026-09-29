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
from systems.ocr.remote_client import default_ocr_mode
from systems.ocr.runtime_hint import ocr_inference_unavailable_detail, paddle_gpu_healthcheck
from systems.observability.last_tuning_log import (
    log_safe_failure,
    summarize_text_fields,
    summarize_validation_result,
    write_last_tuning_log,
)
from systems.observability.llm_fallback_log import (
    clear_llm_fallback_logs,
    count_llm_fallback_logs,
    log_stats,
    read_llm_fallback_logs,
)
from systems.observability.validation_failure_log import (
    clear_validation_failures,
    count_validation_failures,
    failure_log_stats,
    read_validation_failures,
)
from systems.observability.validation_success_log import (
    clear_validation_successes,
    count_validation_successes,
    read_validation_successes,
    success_log_stats,
)
from systems.validation.bank_rekening import (
    list_supported_banks,
    normalize_expected_bank,
    parse_account_from_filename,
)
from systems.validation.document_profiles import list_supported_document_types, resolve_keywords
from systems.validation.user_errors import (
    HttpUserErrorBody,
    PipelineFrontendResponse,
    frontend_error_usage,
    pipeline_http_detail,
    user_error_http_detail,
)

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
            "healthcheck": {
                "method": "GET",
                "path": "/api/v1/healthcheck",
                "description": "Status GPU terdeteksi vs device PaddleOCR (gpu/cpu).",
            },
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
                    "ocr_mode": "fast | vl | mistral (kosong = OCR_DEFAULT_MODE, default fast / PP-OCRv6)",
                    "pp_ocr_tier": "balanced | medium | small | tiny — hanya ocr_mode=fast",
                    "include_preprocessed_image": "true | false (default false)",
                    "enable_preprocess": "true | false (default false) — crop/rotate sebelum OCR",
                    "skip_passthrough": "true | false (default false) — raw upload ke OCR, tanpa resize/grayscale",
                    "full_json": "true | false — hanya ocr_mode=vl",
                },
                "description": (
                    "OCR → validasi. HTTP 200 juga jika dokumen tidak lolos: "
                    "baca error_message, jangan explanation.summary."
                ),
                "ocr_modes": {
                    "fast": "PP-OCRv6 (default pipeline, pp_ocr_tier)",
                    "vl": "PaddleOCR-VL-1.6",
                    "mistral": "Cloud Mistral OCR + document_annotation (opsional)",
                },
                "pp_ocr_tiers": list_pp_ocr_tiers(),
                "response_fields": {
                    "document_matched": "true = VALID. false = tampilkan error_message (HTTP tetap 200)",
                    "error_code": "kode tetap, mis. ktp.name_mismatch. null jika lolos",
                    "error_message": "kalimat siap tampil di UI. null jika lolos",
                    "error_cases": "array {code, message}; >1 item jika gagal lebih dari satu case",
                    "verdict.is_own_document": "true | false | null — milik user saat ini?",
                    "verdict.document_type_current_label": "jenis dokumen terdeteksi dari OCR",
                    "verdict.summary": "ringkasan debug (jangan ditampilkan ke user)",
                },
                "error_catalog": "/systems/validation/api/v1/error-catalog",
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
            "error_catalog": {
                "method": "GET",
                "path": "/systems/validation/api/v1/error-catalog",
                "description": "Katalog error_code + error_message per dokumen/case + usage frontend.",
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
            "health_letter_ai": {
                "method": "POST",
                "path": "/api/v1/health-letter-ai",
                "content_type": "multipart/form-data",
                "description": "Tes OCR AI Ornith: ekstraksi + validasi surat keterangan kesehatan.",
            },
            "validation_failures": {
                "method": "GET",
                "path": "/api/v1/validation-failures",
                "description": "Daftar dokumen gagal validasi/pipeline (JSONL di server).",
            },
            "validation_failures_clear": {
                "method": "DELETE",
                "path": "/api/v1/validation-failures",
                "description": "Hapus semua log kegagalan.",
            },
            "validation_successes": {
                "method": "GET",
                "path": "/api/v1/validation-successes",
                "description": "Daftar dokumen lolos validasi (JSONL di server).",
            },
            "validation_successes_clear": {
                "method": "DELETE",
                "path": "/api/v1/validation-successes",
                "description": "Hapus semua log keberhasilan.",
            },
        },
        "document_types": list_supported_document_types(),
        "docs": "/docs",
        "frontend_errors": frontend_error_usage(),
    }


@router.get("/healthcheck")
def api_healthcheck() -> dict:
    """Status GPU (nvidia-smi / Paddle CUDA) dan device yang dipakai OCR."""
    return paddle_gpu_healthcheck()


def _pipeline_http_error(result: PipelineResult) -> HTTPException:
    """Map PipelineResult gagal ke HTTPException dengan field error frontend."""
    kind = result.error_kind or ""
    extra: dict = {"kind": kind, "message": result.error}
    if kind == "unknown_document_type":
        extra["supported"] = list_supported_document_types()
    elif kind == "cv_unavailable":
        extra.update(
            {
                "code": "CV_SUBSYSTEM_UNAVAILABLE",
                "install": "pip install -r requirements-cv.txt",
                "opensearch": "docker compose -f docker-compose.cv.yml up -d",
            }
        )
    elif kind == "mistral_unavailable":
        extra.update(
            {
                "code": "MISTRAL_OCR_UNAVAILABLE",
                "install": "pip install -r requirements-mistral.txt",
                "env": ["MISTRAL_API_KEY", "MISTRAL_OCR_MODEL"],
            }
        )
    elif kind == "ocr_unavailable":
        extra.update(ocr_inference_unavailable_detail(mode="fast", exc=RuntimeError(result.error or "")))
    elif kind == "health_ai_unavailable":
        extra.update(
            {
                "code": "HEALTH_AI_UNAVAILABLE",
                "engine": "ornith",
                "env": ["LLM_HEALTH_AI_BASE_URL", "LLM_HEALTH_AI_MODEL"],
            }
        )
    elif kind == "opencv_unavailable":
        from systems.cv.index.opensearch_client import opensearch_unavailable_detail

        extra.update(opensearch_unavailable_detail(exc=RuntimeError(result.error or "")))
    detail = pipeline_http_detail(kind, extra=extra)
    return HTTPException(status_code=result.http_status_hint, detail=detail)


@router.post(
    "/pipeline",
    response_description=(
        "HTTP 200 = pipeline selesai (lolos atau tidak). "
        "Jika document_matched=false, tampilkan error_message ke user."
    ),
    responses={
        200: {
            "model": PipelineFrontendResponse,
            "description": (
                "Selesai. Field ocr/validation/verdict tetap ada. "
                "Untuk UI cukup document_matched + error_message + error_cases."
            ),
            "content": {
                "application/json": {
                    "examples": {
                        "lolos": {
                            "summary": "Dokumen lolos",
                            "value": {
                                "success": True,
                                "document_matched": True,
                                "error_code": None,
                                "error_message": None,
                                "error_cases": [],
                            },
                        },
                        "gagal_nama": {
                            "summary": "Nama tidak cocok — tampilkan error_message",
                            "value": {
                                "success": True,
                                "document_matched": False,
                                "error_code": "ktp.name_mismatch",
                                "error_message": "Nama pada KTP tidak cocok.",
                                "error_cases": [
                                    {
                                        "code": "ktp.name_mismatch",
                                        "message": "Nama pada KTP tidak cocok.",
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        },
        400: {
            "model": HttpUserErrorBody,
            "description": "Request tidak valid. Baca detail.error_message.",
        },
        503: {
            "model": HttpUserErrorBody,
            "description": "OCR/CV tidak tersedia. Baca detail.error_message.",
        },
    },
)
async def api_pipeline(
    file: UploadFile = File(..., description="Gambar dokumen"),
    document_type: str = Form(..., description="Jenis dokumen, mis. KTP / NPWP"),
    expected_name: str = Form("", description="Nama referensi (opsional)"),
    ocr_mode: OcrMode | None = Query(
        None,
        description="Engine OCR: fast (PP-OCRv6, default), vl (PaddleOCR-VL), mistral. Kosong = OCR_DEFAULT_MODE.",
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
    expected_bank: str = Form(
        "",
        description="Wajib untuk document_type=rekening: bank yang diharapkan (mandiri | mas)",
    ),
    expected_account: str = Form(
        "",
        description="Untuk document_type=rekening: nomor rekening referensi (opsional jika ada di filename)",
    ),
    job_position: str = Form(
        "",
        description="Posisi kerja. Untuk KTP + Helper: status perkawinan harus BELUM KAWIN.",
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
        raise HTTPException(status_code=400, detail=user_error_http_detail("pipeline.empty_file"))
    if not doc_type:
        log_safe_failure(
            subsystem="pipeline",
            method="POST",
            path=path,
            http_status=400,
            detail="document_type kosong.",
        )
        raise HTTPException(
            status_code=400,
            detail=user_error_http_detail("pipeline.empty_document_type"),
        )

    resolved = resolve_keywords(doc_type)
    if not resolved:
        detail = user_error_http_detail(
            "pipeline.unknown_document_type",
            extra={
                "document_type": doc_type,
                "supported": list_supported_document_types(),
            },
        )
        log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=detail)
        raise HTTPException(status_code=400, detail=detail)

    canonical_id, _keywords = resolved
    upload_name = file.filename or "upload"
    if canonical_id == "rekening":
        if not normalize_expected_bank(expected_bank):
            detail = user_error_http_detail(
                "pipeline.missing_expected_bank",
                extra={
                    "expected_bank": expected_bank.strip(),
                    "supported": list_supported_banks(),
                },
            )
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=detail)
            raise HTTPException(status_code=400, detail=detail)
        acct_ref = (
            expected_account.strip()
            or parse_account_from_filename(upload_name)
            or ""
        )
        if not (name_ref or cv_search_query).strip() and not acct_ref:
            detail = user_error_http_detail("pipeline.missing_expected_identity")
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=detail)
            raise HTTPException(status_code=400, detail=detail)

    if canonical_id == "mutasi":
        if not normalize_expected_bank(expected_bank):
            detail = user_error_http_detail(
                "pipeline.missing_expected_bank",
                extra={
                    "expected_bank": expected_bank.strip(),
                    "supported": list_supported_banks(),
                },
            )
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=detail)
            raise HTTPException(status_code=400, detail=detail)
        acct_ref = (
            expected_account.strip()
            or parse_account_from_filename(upload_name)
            or ""
        )
        if not acct_ref:
            detail = user_error_http_detail("pipeline.missing_expected_account")
            log_safe_failure(subsystem="pipeline", method="POST", path=path, http_status=400, detail=detail)
            raise HTTPException(status_code=400, detail=detail)

    resolved_ocr_mode: OcrMode = ocr_mode or default_ocr_mode()  # type: ignore[assignment]
    result = run_pipeline_bytes(
        raw,
        document_type=doc_type,
        expected_name=(name_ref or cv_search_query).strip(),
        filename=upload_name,
        ocr_mode=resolved_ocr_mode,
        pp_ocr_tier=pp_ocr_tier if resolved_ocr_mode == "fast" else None,
        include_preprocessed_image=include_preprocessed_image,
        enable_preprocess=enable_preprocess,
        skip_passthrough=skip_passthrough,
        full_json=full_json,
        cv_education_query=cv_education_query.strip(),
        cv_experience_query=cv_experience_query.strip(),
        expected_bank=expected_bank.strip(),
        expected_account=expected_account.strip(),
        job_position=job_position.strip(),
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
    tier_arg = pp_ocr_tier if resolved_ocr_mode == "fast" else None

    write_last_tuning_log(
        {
            "success": True,
            "subsystem": "pipeline",
            "method": "POST",
            "path": path,
            "input_bytes": len(raw),
            "query": {
                "ocr_mode": resolved_ocr_mode,
                "pp_ocr_tier": tier_arg,
                "include_preprocessed_image": include_preprocessed_image,
                "enable_preprocess": enable_preprocess,
                "skip_passthrough": skip_passthrough,
                "full_json": full_json,
                "expected_name": (name_ref or cv_search_query).strip() or None,
                "cv_education_query": cv_education_query or None,
                "cv_experience_query": cv_experience_query or None,
                "expected_bank": expected_bank.strip() or None,
                "job_position": job_position.strip() or None,
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
    expected_bank: str = Field(
        "",
        description="Folder rekening: mandiri | mas (kosong = pakai manifest per file bila use_expected_bank)",
    )
    job_position: str = Field(
        "",
        description="Posisi kerja untuk folder KTP (mis. helper → status perkawinan wajib BELUM KAWIN).",
    )


class DatasetBenchmarkBody(BaseModel):
    selections: list[DatasetSelectionBody]
    ocr_mode: OcrMode | None = None
    pp_ocr_tier: PpOcrTier = "medium"
    use_expected_name: bool = True
    use_expected_bank: bool = True
    use_expected_account: bool = True
    enable_preprocess: bool = False
    skip_passthrough: bool = False
    job_position: str = Field(
        "",
        description="Posisi kerja. Untuk folder KTP + Helper: status perkawinan harus BELUM KAWIN.",
    )


@router.get("/dataset/types")
def api_dataset_types() -> dict:
    """Daftar subfolder dataset beserta pemetaan document_type."""
    folders = list_dataset_folders()
    root = dataset_root()
    return {
        "dataset_root": str(root),
        "exists": root.is_dir(),
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
                expected_bank=(s.expected_bank or "").strip() or None,
                job_position=(s.job_position or "").strip() or None,
            )
            for s in body.selections
        ],
        ocr_mode=(body.ocr_mode or default_ocr_mode()),  # type: ignore[arg-type]
        pp_ocr_tier=body.pp_ocr_tier,
        use_expected_name=body.use_expected_name,
        use_expected_bank=body.use_expected_bank,
        use_expected_account=body.use_expected_account,
        enable_preprocess=body.enable_preprocess,
        skip_passthrough=body.skip_passthrough,
        job_position=(body.job_position or "").strip(),
    )

    def stream():
        yield from run_benchmark(config)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/llm-fallback-logs")
def api_llm_fallback_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    rescued_only: bool = Query(False),
    failed_only: bool = Query(False),
) -> dict:
    """Daftar log kegagalan Paddle yang memicu fallback AI vision."""
    entries = read_llm_fallback_logs(limit=limit, offset=offset)
    if rescued_only:
        entries = [e for e in entries if e.get("rescued")]
    elif failed_only:
        entries = [e for e in entries if not e.get("final_matched")]
    stats = log_stats()
    return {
        "entries": entries,
        "stats": stats,
        "limit": limit,
        "offset": offset,
        "total": stats["total"],
        "filtered_count": len(entries),
    }


@router.delete("/llm-fallback-logs")
def api_clear_llm_fallback_logs() -> dict:
    """Hapus semua entri log fallback AI."""
    cleared = clear_llm_fallback_logs()
    return {"cleared": cleared, "total": count_llm_fallback_logs()}


@router.get("/validation-failures")
def api_validation_failures(
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    kind: str | None = Query(None, description="validation_fail | pipeline_error"),
) -> dict:
    """Daftar dokumen yang gagal validasi atau error pipeline."""
    entries = read_validation_failures(limit=limit, offset=offset)
    kind_n = (kind or "").strip()
    if kind_n == "validation_fail":
        entries = [e for e in entries if (e.get("failure_kind") or "validation_fail") == "validation_fail"]
    elif kind_n == "pipeline_error":
        entries = [e for e in entries if (e.get("failure_kind") or "") != "validation_fail"]
    stats = failure_log_stats()
    return {
        "entries": entries,
        "stats": stats,
        "limit": limit,
        "offset": offset,
        "total": stats["total"],
        "filtered_count": len(entries),
    }


@router.delete("/validation-failures")
def api_clear_validation_failures() -> dict:
    """Hapus semua log kegagalan (halaman /dataset-failures)."""
    cleared = clear_validation_failures()
    return {"cleared": cleared, "total": count_validation_failures()}


@router.get("/validation-successes")
def api_validation_successes(
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    document_type: str | None = Query(None, description="Filter profil dokumen, mis. ktp"),
) -> dict:
    """Daftar dokumen yang lolos validasi."""
    entries = read_validation_successes(limit=limit, offset=offset)
    type_n = (document_type or "").strip()
    if type_n:
        entries = [
            e
            for e in entries
            if str(e.get("document_type") or "").strip().casefold() == type_n.casefold()
        ]
    stats = success_log_stats()
    return {
        "entries": entries,
        "stats": stats,
        "limit": limit,
        "offset": offset,
        "total": stats["total"],
        "filtered_count": len(entries),
    }


@router.delete("/validation-successes")
def api_clear_validation_successes() -> dict:
    """Hapus semua log keberhasilan (halaman /dataset-successes)."""
    cleared = clear_validation_successes()
    return {"cleared": cleared, "total": count_validation_successes()}


@router.get("/health-letter-ai/config")
def api_health_letter_ai_config() -> dict:
    """Default server Ornith, daftar model, dan prompt template surat kesehatan."""
    from systems.validation.keterangan_kesehatan_ai import health_ai_config

    return health_ai_config()


@router.post("/health-letter-ai")
async def api_health_letter_ai(
    file: UploadFile = File(..., description="Foto surat keterangan kesehatan"),
    expected_name: str = Form(""),
    expected_nik: str = Form(""),
    llm_base_url: str = Form(""),
    llm_model: str = Form(""),
    user_prompt: str = Form(""),
    today: str = Form(""),
) -> JSONResponse:
    """Kirim gambar ke LLM lokal multimodal, ekstrak field, lalu validasi."""
    from datetime import date as date_cls

    from systems.validation.keterangan_kesehatan_ai import analyze_health_letter_image

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=user_error_http_detail("pipeline.empty_file"))
    as_of = None
    today_s = (today or "").strip()
    if today_s:
        try:
            as_of = date_cls.fromisoformat(today_s[:10])
        except ValueError as e:
            raise HTTPException(status_code=400, detail="today harus YYYY-MM-DD") from e
    mime = file.content_type or "image/jpeg"
    try:
        result = analyze_health_letter_image(
            raw,
            image_mime=mime,
            expected_name=expected_name,
            expected_nik=expected_nik,
            user_prompt=user_prompt or None,
            llm_base_url=llm_base_url.strip() or None,
            llm_model=llm_model.strip() or None,
            today=as_of,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(jsonable_encoder(result))
