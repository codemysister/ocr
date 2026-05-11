"""HTTP routes untuk sistem preprocessing."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from systems.preprocessing.pipeline import preprocess_image_bytes
from systems.preprocessing.realesrgan_infer import realesrgan_status_for_health
from systems.observability.last_tuning_log import log_safe_failure, write_last_tuning_log

STATIC_DIR = Path(__file__).resolve().parent / "static"

ui_router = APIRouter(prefix="/systems/preprocessing", tags=["preprocessing"])
api_router = APIRouter(tags=["preprocessing"])


@ui_router.get("/health")
def preprocessing_health() -> dict:
    return {
        "status": "ok",
        "system": "preprocessing",
        "realesrgan": realesrgan_status_for_health(),
        "last_tuning_log": {
            "file": "logs/last_tuning.json",
            "env_override": "LAST_TUNING_LOG_PATH",
        },
    }


@ui_router.get("/")
def preprocess_page() -> FileResponse:
    page = STATIC_DIR / "preprocess.html"
    if not page.is_file():
        raise HTTPException(404, "preprocess.html tidak ditemukan.")
    return FileResponse(page)


@api_router.post("/api/v1/preprocess", response_model=None)
async def api_preprocess(
    file: UploadFile = File(..., description="File gambar"),
    fmt: str = Query("image", alias="format", description="'image' (PNG grayscale) atau 'json'"),
) -> Response | JSONResponse:
    raw = await file.read()
    path = "/api/v1/preprocess"
    subsystem = "preprocessing"
    if not raw:
        log_safe_failure(
            subsystem=subsystem,
            method="POST",
            path=path,
            http_status=400,
            detail="File kosong.",
        )
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        png_bytes, meta = preprocess_image_bytes(raw)
    except ValueError as e:
        log_safe_failure(
            subsystem=subsystem,
            method="POST",
            path=path,
            http_status=400,
            detail=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    base_log = {
        "success": True,
        "subsystem": subsystem,
        "method": "POST",
        "path": path,
        "input_bytes": len(raw),
        "output_png_bytes": len(png_bytes),
        "meta": meta,
    }
    if fmt.lower() == "json":
        payload = {
            "success": True,
            "mime": "image/png",
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            **meta,
        }
        write_last_tuning_log({**base_log, "response_format": "json"})
        return JSONResponse(payload)

    write_last_tuning_log({**base_log, "response_format": "image/png"})
    hdr = {
        "X-Preprocess-Width": str(meta["width"]),
        "X-Preprocess-Height": str(meta["height"]),
        "X-Preprocess-Card-Warped": "1" if meta.get("card_warped") else "0",
    }
    return Response(content=png_bytes, media_type="image/png", headers=hdr)
