"""HTTP routes untuk sistem OCR (layanan terpisah: gambar → teks / markdown)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse

from systems.ocr.runtime_hint import (
    OCR_HEALTH_VS_INFERENCE,
    PADDLE_INSTALL_URL,
    PADDLEOCR_VL_DOC_URL,
    ocr_inference_unavailable_detail,
    paddlepaddle_importable,
)
from systems.ocr.vl_runner import run_paddleocr_vl

STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter(prefix="/systems/ocr", tags=["ocr"])


@router.get("/health")
def ocr_health() -> dict:
    paddle_ok = paddlepaddle_importable()
    return {
        "status": "ok",
        "system": "ocr",
        "model": "PaddleOCR-VL-1.5",
        "api": "/systems/ocr/api/v1/ocr",
        "paddlepaddle_importable": paddle_ok,
        "inference_ready": paddle_ok,
        "health_vs_inference": OCR_HEALTH_VS_INFERENCE,
        "inference_setup": {
            "paddle_install": PADDLE_INSTALL_URL,
            "paddleocr_vl_doc": PADDLEOCR_VL_DOC_URL,
            "vl_remote_env": ["OCR_VL_BACKEND", "OCR_VL_SERVER_URL"],
        },
    }


@router.get("/")
def ocr_page() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "index.html tidak ditemukan.")
    return FileResponse(page)


@router.post("/api/v1/ocr")
async def api_ocr_vl(
    file: UploadFile = File(..., description="Gambar masukan (JPEG/PNG/WebP, dll.)"),
    full_json: bool = Query(
        False,
        description="Jika true, lampirkan objek result_json lengkap (dapat besar).",
    ),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        payload = run_paddleocr_vl(raw, include_full_json=full_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e).lower()
        if any(
            x in msg
            for x in (
                "paddlepaddle",
                "dependency",
                "engine",
                "unavailable",
            )
        ):
            raise HTTPException(
                status_code=503,
                detail=ocr_inference_unavailable_detail(),
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(content=jsonable_encoder(payload))
