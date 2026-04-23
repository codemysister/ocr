"""Aplikasi gabungan: beberapa subsistem di bawah `systems/`."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from systems.ai_parsing.routes import router as ai_parsing_router
from systems.ocr.routes import router as ocr_router
from systems.preprocessing.routes import api_router as preprocessing_api_router
from systems.preprocessing.routes import ui_router as preprocessing_ui_router

HUB_PATH = Path(__file__).resolve().parent / "static" / "hub.html"

app = FastAPI(
    title="OCR Platform",
    description="Preprocessing, OCR, dan AI parsing (opsional) dalam satu proses.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocessing_ui_router)
app.include_router(preprocessing_api_router)
app.include_router(ocr_router)
app.include_router(ai_parsing_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "systems": {
            "preprocessing": "/systems/preprocessing/health",
            "ocr": "/systems/ocr/health",
            "ai_parsing": "/systems/ai-parsing/health",
        },
    }


@app.get("/")
def hub() -> FileResponse:
    if not HUB_PATH.is_file():
        raise HTTPException(404, "hub.html tidak ditemukan.")
    return FileResponse(HUB_PATH)


@app.get("/preprocess")
def legacy_preprocess() -> RedirectResponse:
    return RedirectResponse(url="/systems/preprocessing/", status_code=307)
