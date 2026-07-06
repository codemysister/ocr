"""Teks bantuan infrastruktur OCR Paddle (satu sumber kebenaran untuk API & UI)."""

from __future__ import annotations

from typing import Any, Literal

PADDLE_INSTALL_URL = "https://www.paddlepaddle.org.cn/install/quick"
PADDLEOCR_VL_DOC_URL = (
    "https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html"
)

OCR_FAST_UNAVAILABLE = (
    "PP-OCRv6 (mode fast) tidak bisa dijalankan di server ini. "
    "Pastikan PaddlePaddle dan PaddleOCR terpasang untuk Python/OS Anda "
    "(biasanya Python 3.10–3.12; pip install -r requirements.txt). "
    "Rujukan instalasi: " + PADDLE_INSTALL_URL
)

# Respons HTTP 503 saat PaddleOCR-VL tidak tersedia.
OCR_VL_INFERENCE_UNAVAILABLE = (
    "PaddleOCR-VL-1.6 tidak bisa dijalankan di server ini: pipeline membutuhkan PaddlePaddle untuk "
    "deteksi layout (mis. PP-DocLayoutV3) dan, bila backend VL disetel ke native, juga untuk inferensi VL. "
    "Pasang wheel untuk OS dan Python Anda (biasanya 3.10–3.12; "
    + PADDLE_INSTALL_URL
    + "). OCR_VL_BACKEND + OCR_VL_SERVER_URL hanya mengalihkan inferensi VL ke server jarak jauh "
    "(mis. vLLM); deteksi layout pada stack ini tetap memerlukan PaddlePaddle. Rujukan: "
    + PADDLEOCR_VL_DOC_URL
)

# Alias lama — tetap merujuk pesan VL agar tidak membingungkan pemanggil VL.
OCR_INFERENCE_UNAVAILABLE = OCR_VL_INFERENCE_UNAVAILABLE

OCR_HEALTH_VS_INFERENCE = (
    "Field status=ok hanya berarti layanan HTTP hidup. Inferensi OCR default memakai "
    "POST /systems/ocr/api/v1/ocr-fast (PP-OCRv6) dan membutuhkan PaddlePaddle + paddleocr. "
    "Mode vl memakai PaddleOCR-VL; tanpa PaddlePaddle respons inferensi 503."
)


def paddlepaddle_importable() -> bool:
    """True jika `import paddle` berhasil (cek ringan, tanpa memuat model VL)."""
    try:
        import paddle  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_fast_unavailable_detail(*, exc: BaseException | None = None) -> dict[str, Any]:
    """Payload 503 untuk PP-OCRv6 (ocr-fast / pipeline ocr_mode=fast)."""
    detail: dict[str, Any] = {
        "code": "OCR_FAST_UNAVAILABLE",
        "message": OCR_FAST_UNAVAILABLE,
        "engine": "pp-ocrv6",
        "links": {"paddle_install": PADDLE_INSTALL_URL},
        "install": "pip install -r requirements.txt",
        "env": [
            "OCR_FAST_LANG",
            "OCR_FAST_TIER",
            "OCR_FAST_DOC_ORIENTATION",
            "OCR_FAST_DOC_UNWARPING",
            "OCR_FAST_TEXTLINE_ORIENTATION",
        ],
    }
    if exc is not None:
        detail["error"] = str(exc)
    return detail


def ocr_vl_unavailable_detail(*, exc: BaseException | None = None) -> dict[str, Any]:
    """Payload 503 untuk PaddleOCR-VL."""
    detail: dict[str, Any] = {
        "code": "OCR_VL_UNAVAILABLE",
        "message": OCR_VL_INFERENCE_UNAVAILABLE,
        "engine": "paddleocr-vl",
        "links": {
            "paddle_install": PADDLE_INSTALL_URL,
            "paddleocr_vl": PADDLEOCR_VL_DOC_URL,
        },
        "remote_vl_env": ["OCR_VL_BACKEND", "OCR_VL_SERVER_URL"],
    }
    if exc is not None:
        detail["error"] = str(exc)
    return detail


def ocr_inference_unavailable_detail(
    *,
    mode: Literal["fast", "vl"] = "vl",
    exc: BaseException | None = None,
) -> dict[str, Any]:
    """Payload `detail` untuk HTTP 503 (mudah diparse klien)."""
    if mode == "fast":
        return ocr_fast_unavailable_detail(exc=exc)
    return ocr_vl_unavailable_detail(exc=exc)
