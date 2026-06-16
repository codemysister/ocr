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
        "orientation": {
            "exif": "Dekode via Pillow + ImageOps.exif_transpose (lalu ke BGR) ketika berhasil.",
            "full_bleed_skip_warp": (
                "Jika kontur terluar menutupi ≥ PREPROCESS_SKIP_WARP_WHEN_COVER_RATIO (default 0.88), "
                "perspective warp dilewati agar scan yang sudah lurus tidak distorsi."
            ),
            "resize": (
                "Default: tanpa upscale (PREPROCESS_MIN_SIDE_TARGET=0). "
                "Downscale hanya jika sisi terpanjang > PREPROCESS_MAX_SIDE (default 2400). "
                "Upscale lama: set PREPROCESS_MIN_SIDE_TARGET=900."
            ),
            "card_warp": (
                "Default: mati (PREPROCESS_CARD_WARP=0) — screenshot/email tidak di-crop. "
                "Aktifkan PREPROCESS_CARD_WARP=1 untuk foto kartu fisik. "
                "PREPROCESS_CARD_WARP_STYLE=auto|axis_box|perspective (default auto): "
                "auto memakai crop poros bila kotak sudah hampir sejajar sumbu, supaya dokumen lurus "
                "tidak distorsi perspective; perspective = selalu seperti sebelumnya."
            ),
            "axis_crop_thresholds": (
                "Mode auto: jika skew minAreaRect ≤ PREPROCESS_AUTO_AXIS_ONLY_MAX_SKEW_DEG (default 24°), "
                "hanya crop poros — perspective tidak dipakai untuk kontur itu. "
                "Naikkan nilai (mis. 28) bila foto kartu agak miring tapi masih ingin tanpa perspective."
            ),
            "auto_rotate_quarters": (
                "PREPROCESS_AUTO_ROTATE_QUARTERS: off (default) | auto | on. "
                "off = jalur putar penuh lewat env dimatikan; opsional suplemen pra-warp "
                "(PREPROCESS_SUPPLEMENT_QUARTER_PRE_WARP=1 — default mati) hanya memutar ±90° jika s1 vs s3 "
                "menang jelas atas 0° dan **tidak** seri (menghindari tebakan arah salah). "
                "auto/on = putar penuh sebelum + sesudah warp; override PRE_* env seperti biasa. "
                "PREPROCESS_AUTO_ROTATE_ALLOW_180=1 mengizinkan pemilihan orientasi 180° otomatis (default mati)."
            ),
            "face_rotator": (
                "PREPROCESS_AUTO_IMAGE_ROTATOR=1 mengaktifkan putar berdasarkan wajah Haar/dlib sebelum langkah lain "
                "(default mati — mengurangi salah orientasi/terbalik pada scan dokumen)."
            ),
        },
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
        "X-Preprocess-Card-Warp-Mode": str(meta.get("card_warp_mode", "")),
        "X-Preprocess-Auto-Rotate-90ccw": str(meta.get("auto_rotate_90ccw_steps", 0)),
        "X-Preprocess-Auto-Rotate-Pre-90ccw": str(meta.get("auto_rotate_pre_90ccw_steps", 0)),
        "X-Preprocess-Auto-Rotate-Supplement-Pre-90ccw": str(
            meta.get("auto_rotate_supplement_pre_90ccw_steps", 0)
        ),
        "X-Preprocess-Exif-Decode": str(meta.get("exif_decode", "")),
    }
    return Response(content=png_bytes, media_type="image/png", headers=hdr)
