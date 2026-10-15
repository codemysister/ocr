"""Aplikasi gabungan: beberapa subsistem di bawah `systems/`."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api_routes import router as public_api_router
from systems.ocr.remote_client import default_ocr_mode, ocr_remote_base_url, ocr_remote_enabled
from systems.ocr.routes import router as ocr_router
from systems.validation.routes import router as validation_router
from systems.preprocessing.routes import api_router as preprocessing_api_router
from systems.preprocessing.routes import ui_router as preprocessing_ui_router

try:
    from systems.cv.routes import router as cv_router

    _CV_AVAILABLE = True
except ImportError:
    cv_router = None  # type: ignore[assignment,misc]
    _CV_AVAILABLE = False

HUB_PATH = Path(__file__).resolve().parent / "static" / "hub.html"
PIPELINE_PATH = Path(__file__).resolve().parent / "static" / "pipeline.html"
DATASET_TEST_PATH = Path(__file__).resolve().parent / "static" / "dataset_test.html"
DATASET_FAILURES_PATH = Path(__file__).resolve().parent / "static" / "dataset_failures.html"
DATASET_SUCCESSES_PATH = Path(__file__).resolve().parent / "static" / "dataset_successes.html"
HEALTHCHECK_PATH = Path(__file__).resolve().parent / "static" / "healthcheck.html"
LLM_FALLBACK_LOG_PATH = Path(__file__).resolve().parent / "static" / "llm_fallback_log.html"
PADDLE_VL_PATH = Path(__file__).resolve().parent / "static" / "paddle_vl.html"
HEALTH_LETTER_AI_PATH = Path(__file__).resolve().parent / "static" / "health_letter_ai.html"


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _cors_settings() -> tuple[list[str], bool]:
    origins = _cors_origins()
    # Browser menolak credentials + wildcard origin bersamaan.
    allow_credentials = "*" not in origins
    return origins, allow_credentials


_cors_origins_list, _allow_credentials = _cors_settings()

app = FastAPI(
    title="OCR Platform",
    description=(
        "Preprocessing, OCR, dan validasi dokumen.\n\n"
        "**Frontend:** pada `POST /api/v1/pipeline`, HTTP 200 berarti proses selesai "
        "(bukan otomatis lolos). Baca `document_matched`. Jika `false`, tampilkan "
        "`error_message` (kode: `error_code`, daftar: `error_cases`). "
        "HTTP 4xx/5xx: baca `detail.error_message`. "
        "`keterangan_kesehatan` memakai Ornith AI (bukan Paddle); gagal AI → 503 "
        "`pipeline.health_ai_unavailable`. "
        "`npwp` memakai Ornith AI; `format_lama` true jika nomor NPWP bukan NIK 16 digit. "
        "`pemadanan_npwp` memakai Ornith AI; kartu NPWP 16 digit atau Ikhtisar Profil, nama boleh terpotong, NIK filename bisa lolos. "
        "`rekening` memakai Ornith AI; nama boleh sebagian atau tersensor. "
        "`bpjs` (Kartu Indonesia Sehat) memakai Ornith AI. "
        "`iuran` memakai Ornith AI; modal tanpa tagihan pribadi lolos meski nama tidak tampil. "
        "`vaksinasi_1` memakai Ornith AI; dosis pertama tetap lolos bila tahap 2 juga tercantum. "
        "`vaksinasi_2` memakai Ornith AI; dosis kedua tetap lolos bila tahap 1 juga tercantum. "
        "`vaksinasi_3` memakai Ornith AI; booster tetap lolos bila tahap 1 atau 2 juga tercantum. "
        "Katalog: `GET /systems/validation/api/v1/error-catalog` · ringkasan: `GET /api/v1` · "
        "spesifikasi lengkap: file `API.md` di root repo."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_list,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(public_api_router)
app.include_router(preprocessing_ui_router)
app.include_router(preprocessing_api_router)
app.include_router(ocr_router)
app.include_router(validation_router)
if _CV_AVAILABLE and cv_router is not None:
    app.include_router(cv_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "api": "/api/v1",
        "docs": "/docs",
        "cors_origins": _cors_origins_list,
        "pipeline_ocr": (
            f"default {default_ocr_mode()} "
            "(ocr_mode=fast|mistral|vl; pp_ocr_tier hanya untuk fast)"
        ),
        "ocr_default_mode": default_ocr_mode(),
        "ocr_remote_base_url": ocr_remote_base_url() if ocr_remote_enabled() else None,
        "ocr_remote_enabled": ocr_remote_enabled(),
        "systems": {
            "preprocessing": "/systems/preprocessing/health",
            "ocr": "/systems/ocr/health",
            "validation": "/systems/validation/health",
            **(
                {"cv": "/systems/cv/health"}
                if _CV_AVAILABLE
                else {}
            ),
        },
        "cv_search": _CV_AVAILABLE,
        "last_tuning_log": {
            "path": "logs/last_tuning.json",
            "absolute_note": "Di root repo; override dengan env LAST_TUNING_LOG_PATH.",
        },
    }


@app.get("/")
def hub() -> FileResponse:
    if not HUB_PATH.is_file():
        raise HTTPException(404, "hub.html tidak ditemukan.")
    # Hindari beranda tertahan cache setelah kita mengganti tautan subsistem.
    return FileResponse(
        HUB_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/pipeline")
def pipeline_page() -> FileResponse:
    if not PIPELINE_PATH.is_file():
        raise HTTPException(404, "pipeline.html tidak ditemukan.")
    return FileResponse(
        PIPELINE_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/dataset-test")
def dataset_test_page() -> FileResponse:
    if not DATASET_TEST_PATH.is_file():
        raise HTTPException(404, "dataset_test.html tidak ditemukan.")
    return FileResponse(
        DATASET_TEST_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/dataset-failures")
def dataset_failures_page() -> FileResponse:
    if not DATASET_FAILURES_PATH.is_file():
        raise HTTPException(404, "dataset_failures.html tidak ditemukan.")
    return FileResponse(
        DATASET_FAILURES_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/dataset-successes")
def dataset_successes_page() -> FileResponse:
    if not DATASET_SUCCESSES_PATH.is_file():
        raise HTTPException(404, "dataset_successes.html tidak ditemukan.")
    return FileResponse(
        DATASET_SUCCESSES_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/healthcheck")
def healthcheck_page() -> FileResponse:
    if not HEALTHCHECK_PATH.is_file():
        raise HTTPException(404, "healthcheck.html tidak ditemukan.")
    return FileResponse(
        HEALTHCHECK_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/paddle-vl")
def paddle_vl_page() -> FileResponse:
    if not PADDLE_VL_PATH.is_file():
        raise HTTPException(404, "paddle_vl.html tidak ditemukan.")
    return FileResponse(
        PADDLE_VL_PATH,
        headers={
            "Cache-Control": "no-store, max-age=0, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/health-letter-ai")
def health_letter_ai_page() -> FileResponse:
    if not HEALTH_LETTER_AI_PATH.is_file():
        raise HTTPException(404, "health_letter_ai.html tidak ditemukan.")
    return FileResponse(
        HEALTH_LETTER_AI_PATH,
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


@app.get("/llm-fallback-log")
def llm_fallback_log_page() -> FileResponse:
    if not LLM_FALLBACK_LOG_PATH.is_file():
        raise HTTPException(404, "llm_fallback_log.html tidak ditemukan.")
    return FileResponse(
        LLM_FALLBACK_LOG_PATH,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/preprocess")
def legacy_preprocess() -> RedirectResponse:
    return RedirectResponse(url="/systems/preprocessing/", status_code=307)
