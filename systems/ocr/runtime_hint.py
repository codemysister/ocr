"""Teks bantuan infrastruktur PaddleOCR-VL (satu sumber kebenaran untuk API & UI)."""

from __future__ import annotations

from typing import Any

PADDLE_INSTALL_URL = "https://www.paddlepaddle.org.cn/install/quick"
PADDLEOCR_VL_DOC_URL = (
    "https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.html"
)

# Respons HTTP 503 saat PaddlePaddle / engine tidak tersedia.
OCR_INFERENCE_UNAVAILABLE = (
    "PaddleOCR-VL-1.5 tidak bisa dijalankan di server ini: pipeline membutuhkan PaddlePaddle untuk "
    "deteksi layout (mis. PP-DocLayoutV3) dan, bila backend VL disetel ke native, juga untuk inferensi VL. "
    "Pasang wheel untuk OS dan Python Anda (biasanya 3.10–3.12; "
    + PADDLE_INSTALL_URL
    + "). OCR_VL_BACKEND + OCR_VL_SERVER_URL hanya mengalihkan inferensi VL ke server jarak jauh "
    "(mis. vLLM); deteksi layout pada stack ini tetap memerlukan PaddlePaddle. Rujukan: "
    + PADDLEOCR_VL_DOC_URL
)

# Penjelasan untuk GET /health (mencegah salah paham: status ok ≠ OCR bisa inferensi).
OCR_HEALTH_VS_INFERENCE = (
    "Field status=ok hanya berarti layanan HTTP hidup. Inferensi gambar dilakukan oleh "
    "POST /systems/ocr/api/v1/ocr, yang memuat pipeline PaddleOCR-VL dan membutuhkan PaddlePaddle "
    "untuk deteksi layout; tanpa itu responsnya 503 dengan pesan inference_unavailable."
)


def paddlepaddle_importable() -> bool:
    """True jika `import paddle` berhasil (cek ringan, tanpa memuat model VL)."""
    try:
        import paddle  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_inference_unavailable_detail() -> dict[str, Any]:
    """Payload `detail` untuk HTTP 503 (mudah diparse klien; `message` tetap human-readable)."""
    return {
        "code": "OCR_INFERENCE_UNAVAILABLE",
        "message": OCR_INFERENCE_UNAVAILABLE,
        "links": {
            "paddle_install": PADDLE_INSTALL_URL,
            "paddleocr_vl": PADDLEOCR_VL_DOC_URL,
        },
        "remote_vl_env": ["OCR_VL_BACKEND", "OCR_VL_SERVER_URL"],
    }
