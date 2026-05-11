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
from systems.ocr.fast_runner import run_paddleocr_fast
from systems.ocr.vl_runner import run_paddleocr_vl
from systems.observability.last_tuning_log import log_safe_failure, summarize_text_fields, write_last_tuning_log

STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter(prefix="/systems/ocr", tags=["ocr"])


@router.get("/health")
def ocr_health() -> dict:
    paddle_ok = paddlepaddle_importable()
    return {
        "status": "ok",
        "system": "ocr",
        "model": "PaddleOCR-VL-1.5",
        "api_vl": "/systems/ocr/api/v1/ocr",
        "api_fast": "/systems/ocr/api/v1/ocr-fast",
        "paddlepaddle_importable": paddle_ok,
        "inference_ready": paddle_ok,
        "health_vs_inference": OCR_HEALTH_VS_INFERENCE,
        "ocr_fast": {
            "default_for_lang_en_latin": "PP-OCRv5_mobile_det + PP-OCRv5_server_rec (kualitas baca lebih baik, tetap jauh lebih ringan dari VL)",
            "env": {
                "OCR_FAST_LANG": "en atau latin (untuk dokumen Latin / KTP coba latin)",
                "OCR_FAST_DET_MODEL": "mengganti det, mis. PP-OCRv5_server_det untuk akurasi kotak maksimal",
                "OCR_FAST_REC_MODEL": "mengganti rec, mis. en_PP-OCRv5_mobile_rec untuk paling ringan",
                "OCR_FAST_DET_LIMIT_SIDE_LEN": "opsional, panjang sisi deteksi",
            },
        },
        "vl_memory_tuning": {
            "OCR_VL_MAX_LONG_SIDE": (
                "Opsional. Batasi sisi terpanjang gambar (px) sebelum inferensi VL "
                "untuk mengurangi puncak RAM dan waktu; mis. 2048 atau 2560. "
                "Kosongkan = tanpa resize (kualitas maksimal, memori maksimal)."
            ),
            "OCR_VL_BACKEND_and_SERVER": (
                "Alihkan inferensi VL ke server (mis. vLLM); lihat OCR_VL_BACKEND dan OCR_VL_SERVER_URL "
                "di runtime_hint / dokumen PaddleOCR-VL (layout lokal tetap butuh Paddle)."
            ),
        },
        "inference_setup": {
            "paddle_install": PADDLE_INSTALL_URL,
            "paddleocr_vl_doc": PADDLEOCR_VL_DOC_URL,
            "vl_remote_env": ["OCR_VL_BACKEND", "OCR_VL_SERVER_URL"],
        },
        "last_tuning_log": {
            "file": "logs/last_tuning.json",
            "env_override": "LAST_TUNING_LOG_PATH",
            "note": "Ringkasan request terakhir (semua subsistem); ditimpa setiap panggilan API terkait.",
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
    """Respons berisi `timing` (detik): decode, get_pipeline, predict, restructure_pages, total."""
    path = "/systems/ocr/api/v1/ocr"
    sub = "ocr_vl"
    raw = await file.read()
    if not raw:
        log_safe_failure(
            subsystem=sub, method="POST", path=path, http_status=400, detail="File kosong."
        )
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        payload = run_paddleocr_vl(raw, include_full_json=full_json)
    except ValueError as e:
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=400, detail=str(e))
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
            det = ocr_inference_unavailable_detail()
            log_safe_failure(subsystem=sub, method="POST", path=path, http_status=503, detail=det)
            raise HTTPException(status_code=503, detail=det) from e
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    write_last_tuning_log(
        {
            "success": True,
            "subsystem": sub,
            "method": "POST",
            "path": path,
            "input_bytes": len(raw),
            "query": {"full_json": full_json},
            "result": summarize_text_fields(payload),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))


@router.post("/api/v1/ocr-fast")
async def api_ocr_fast(
    file: UploadFile = File(..., description="Gambar masukan (JPEG/PNG/WebP, dll.)"),
) -> JSONResponse:
    """
    OCR ringan: PP-OCRv5 (default: mobile_det + server_rec untuk lang en/latin; tanpa VL).
    Respons: `text`, `markdown` (bullet per baris), `lines`, `timing`.
    Lingkungan: OCR_FAST_LANG, OCR_FAST_DET_MODEL, OCR_FAST_REC_MODEL, OCR_FAST_DET_LIMIT_SIDE_LEN.
    """
    path = "/systems/ocr/api/v1/ocr-fast"
    sub = "ocr_fast"
    raw = await file.read()
    if not raw:
        log_safe_failure(
            subsystem=sub, method="POST", path=path, http_status=400, detail="File kosong."
        )
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        payload = run_paddleocr_fast(raw)
    except ValueError as e:
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=400, detail=str(e))
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
            det = ocr_inference_unavailable_detail()
            log_safe_failure(subsystem=sub, method="POST", path=path, http_status=503, detail=det)
            raise HTTPException(status_code=503, detail=det) from e
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    write_last_tuning_log(
        {
            "success": True,
            "subsystem": sub,
            "method": "POST",
            "path": path,
            "input_bytes": len(raw),
            "result": summarize_text_fields(payload),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))
