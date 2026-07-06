"""Aplikasi gabungan: beberapa subsistem di bawah `systems/`."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api_routes import router as public_api_router
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
    description="Preprocessing, OCR, dan validasi nama (RapidFuzzy) dalam satu proses.",
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
        "pipeline_ocr": "default fast (PP-OCRv6); query ocr_mode=fast|mistral|vl, pp_ocr_tier untuk fast",
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


@app.get("/preprocess")
def legacy_preprocess() -> RedirectResponse:
    return RedirectResponse(url="/systems/preprocessing/", status_code=307)
