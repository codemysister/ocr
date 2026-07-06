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
from systems.ocr.fast_runner import list_pp_ocr_tiers, run_paddleocr_fast
from systems.ocr.mistral_runner import run_mistral_ocr
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
        "default_engine": "pp-ocrv6",
        "model": "PP-OCRv6 (default) / PaddleOCR-VL-1.6 (opsional)",
        "api_vl": "/systems/ocr/api/v1/ocr",
        "api_fast": "/systems/ocr/api/v1/ocr-fast",
        "api_mistral": "/systems/ocr/api/v1/ocr-mistral",
        "paddlepaddle_importable": paddle_ok,
        "inference_ready": paddle_ok,
        "health_vs_inference": OCR_HEALTH_VS_INFERENCE,
        "ocr_fast": {
            "default_tier": "medium",
            "pp_ocr_tiers": list_pp_ocr_tiers(),
            "default_models": "PP-OCRv6_medium_det + PP-OCRv6_medium_rec",
            "paddle_modules_default": {
                "doc_orientation": True,
                "doc_unwarping": True,
                "textline_orientation": True,
            },
            "env": {
                "OCR_FAST_LANG": "en atau latin (untuk dokumen Latin / KTP; default en)",
                "OCR_FAST_TIER": "balanced | medium | small | tiny (default medium)",
                "OCR_FAST_DOC_ORIENTATION": "default 1 — koreksi orientasi halaman (0 matikan)",
                "OCR_FAST_TEXTLINE_ORIENTATION": "default 1 — koreksi orientasi baris teks",
                "OCR_FAST_DOC_UNWARPING": "default 1 — koreksi distorsi dokumen (0 matikan)",
                "OCR_FAST_DET_MODEL": "override det, mis. PP-OCRv6_medium_det",
                "OCR_FAST_REC_MODEL": "override rec, mis. PP-OCRv6_small_rec",
                "OCR_FAST_DET_LIMIT_SIDE_LEN": "opsional, panjang sisi deteksi",
                "OCR_ENABLE_MKLDNN": "default 0 — MKLDNN/oneDNN mati (bug PIR PaddlePaddle 3.x); set 1 untuk aktifkan",
            },
        },
        "vl_memory_tuning": {
            "OCR_VL_PIPELINE_VERSION": "default v1.6 — v1 | v1.5 | v1.6",
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
        "mistral_ocr": {
            "pricing": "~$2 / 1000 halaman standar, ~$3 / 1000 dengan document_annotation",
            "env": {
                "MISTRAL_API_KEY": "wajib — https://console.mistral.ai/",
                "MISTRAL_OCR_MODEL": "default mistral-ocr-2512",
                "MISTRAL_OCR_ANNOTATION": "default 1 — structured holder_name via document_annotation_format",
                "MISTRAL_OCR_ANNOTATION_PROMPT": "opsional — override prompt ekstraksi annotation",
                "MISTRAL_OCR_TABLE_FORMAT": "opsional: markdown | html",
                "MISTRAL_OCR_INCLUDE_IMAGE_BASE64": "1 untuk sertakan potongan gambar di respons",
            },
            "install": "pip install -r requirements-mistral.txt",
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
            det = ocr_inference_unavailable_detail(mode="vl", exc=e)
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
    pp_ocr_tier: str = Query(
        "medium",
        description="Tier PP-OCRv6 (default medium = akurasi maksimal)",
    ),
) -> JSONResponse:
    """
    OCR ringan: PP-OCRv6 (tier via query pp_ocr_tier).
    Respons: `text`, `markdown` (bullet per baris), `lines`, `timing`, `pp_ocr_tier`.
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
        payload = run_paddleocr_fast(raw, pp_ocr_tier=pp_ocr_tier.strip() or "medium")
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
                "paddleocr",
            )
        ):
            det = ocr_inference_unavailable_detail(mode="fast", exc=e)
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
            "query": {"pp_ocr_tier": pp_ocr_tier},
            "result": summarize_text_fields(payload),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))


@router.post("/api/v1/ocr-mistral")
async def api_ocr_mistral(
    file: UploadFile = File(..., description="Gambar atau PDF (JPEG/PNG/WebP/PDF, dll.)"),
    table_format: str | None = Query(
        None,
        description="Format tabel terpisah: markdown | html. Kosong = inline di markdown.",
    ),
) -> JSONResponse:
    """
    OCR cloud Mistral (berbayar). Butuh MISTRAL_API_KEY.
    Respons: text, markdown, lines, usage (biaya estimasi), timing.
    """
    path = "/systems/ocr/api/v1/ocr-mistral"
    sub = "ocr_mistral"
    raw = await file.read()
    if not raw:
        log_safe_failure(
            subsystem=sub, method="POST", path=path, http_status=400, detail="File kosong."
        )
        raise HTTPException(status_code=400, detail="File kosong.")

    tf = (table_format or "").strip() or None
    if tf and tf not in ("markdown", "html"):
        raise HTTPException(
            status_code=400,
            detail="table_format harus markdown atau html.",
        )

    try:
        payload = run_mistral_ocr(raw, table_format=tf)
    except ValueError as e:
        log_safe_failure(subsystem=sub, method="POST", path=path, http_status=400, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e).lower()
        if "mistral_api_key" in msg or "pasang sdk mistral" in msg:
            log_safe_failure(subsystem=sub, method="POST", path=path, http_status=503, detail=str(e))
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MISTRAL_OCR_UNAVAILABLE",
                    "message": str(e),
                    "install": "pip install -r requirements-mistral.txt",
                    "env": ["MISTRAL_API_KEY"],
                },
            ) from e
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
            "query": {"table_format": tf},
            "result": summarize_text_fields(payload),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))
