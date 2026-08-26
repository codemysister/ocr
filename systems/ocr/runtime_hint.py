"""Teks bantuan infrastruktur OCR Paddle (satu sumber kebenaran untuk API & UI)."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

logger = logging.getLogger(__name__)

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


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def paddle_cuda_status() -> dict[str, Any]:
    """Status kompilasi CUDA Paddle (tanpa memuat model OCR)."""
    status: dict[str, Any] = {
        "importable": False,
        "compiled_with_cuda": False,
        "cuda_device_count": 0,
        "gpu_ready": False,
    }
    try:
        import paddle
    except ImportError:
        return status
    status["importable"] = True
    try:
        compiled = bool(paddle.device.is_compiled_with_cuda())
        status["compiled_with_cuda"] = compiled
        count = int(paddle.device.cuda.device_count()) if compiled else 0
        status["cuda_device_count"] = count
        status["gpu_ready"] = compiled and count > 0
    except Exception as e:
        status["error"] = str(e)
    return status


def _normalize_ocr_device(raw: str) -> str:
    device = (raw or "").strip() or "gpu"
    lowered = device.lower()
    if lowered in ("gpu", "cuda"):
        return "gpu:0"
    return device


def nvidia_gpu_status() -> dict[str, Any]:
    """Deteksi GPU NVIDIA di host (nvidia-smi), terpisah dari wheel Paddle."""
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"detected": False, "gpus": [], "error": "nvidia-smi tidak ada di PATH"}
    try:
        proc = subprocess.run(
            [smi, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as e:
        return {"detected": False, "gpus": [], "error": str(e)}
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if proc.returncode != 0 or not lines:
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        return {"detected": False, "gpus": [], "error": err[:400]}
    return {"detected": True, "gpus": lines}


def paddle_gpu_healthcheck() -> dict[str, Any]:
    """Ringkasan: GPU terdeteksi vs device yang dipakai PaddleOCR."""
    cuda = paddle_cuda_status()
    nvidia = nvidia_gpu_status()
    requested = _normalize_ocr_device(os.environ.get("OCR_DEVICE") or "gpu")
    paddle_ok = bool(cuda.get("importable"))
    wants_gpu = requested.lower().startswith("gpu")
    using_gpu = bool(wants_gpu and cuda.get("gpu_ready"))
    if paddle_ok:
        active = requested if using_gpu else resolve_paddle_device()
    else:
        active = None
    gpu_detected = bool(nvidia.get("detected") or cuda.get("gpu_ready"))
    fallback = bool(wants_gpu and active and not using_gpu)

    version = None
    current_place = None
    gpu_names: list[str] = []
    if paddle_ok:
        try:
            import paddle

            version = getattr(paddle, "__version__", None)
            try:
                current_place = str(paddle.device.get_device())
            except Exception:
                current_place = None
            n = int(cuda.get("cuda_device_count") or 0)
            if cuda.get("compiled_with_cuda") and n > 0:
                for i in range(n):
                    try:
                        gpu_names.append(str(paddle.device.cuda.get_device_name(i)))
                    except Exception:
                        gpu_names.append(f"gpu:{i}")
        except Exception:
            pass

    if using_gpu:
        summary = "OCR memakai GPU."
    elif gpu_detected and not using_gpu:
        summary = (
            "GPU terdeteksi, tetapi PaddleOCR berjalan di CPU "
            "(wheel CPU atau OCR_DEVICE=cpu)."
        )
    elif not gpu_detected:
        summary = "GPU tidak terdeteksi. OCR memakai CPU."
    else:
        summary = "Status GPU tidak lengkap."

    return {
        "gpu_detected": gpu_detected,
        "using_gpu": using_gpu,
        "fallback_to_cpu": fallback,
        "requested_device": requested,
        "active_device": active,
        "summary": summary,
        "nvidia_smi": nvidia,
        "paddle": {
            **cuda,
            "version": version,
            "current_place": current_place,
            "gpu_names": gpu_names,
        },
        "env": {
            "OCR_DEVICE": (os.environ.get("OCR_DEVICE") or "").strip() or "gpu (default)",
            "OCR_DEVICE_FALLBACK": (os.environ.get("OCR_DEVICE_FALLBACK") or "").strip()
            or "1 (default)",
        },
    }


_resolved_paddle_device: str | None = None


def resolve_paddle_device() -> str:
    """Device PaddleOCR: default GPU (`OCR_DEVICE=gpu`). Fallback CPU jika CUDA tidak siap.

    Set `OCR_DEVICE=cpu` untuk memaksa CPU. Set `OCR_DEVICE_FALLBACK=0` agar gagal
    jika GPU diminta tetapi tidak tersedia.
    """
    global _resolved_paddle_device
    requested = _normalize_ocr_device(os.environ.get("OCR_DEVICE") or "gpu")
    if _resolved_paddle_device is not None:
        cached = _resolved_paddle_device
        if str(cached).lower().startswith("gpu") or not requested.lower().startswith("gpu"):
            return cached
        if not paddle_cuda_status().get("gpu_ready"):
            return cached
        _resolved_paddle_device = None
    wants_gpu = requested.lower().startswith("gpu")
    if not wants_gpu:
        _resolved_paddle_device = requested
        logger.info("PaddleOCR device=%s (OCR_DEVICE)", requested)
        return requested

    cuda = paddle_cuda_status()
    if cuda["gpu_ready"]:
        _resolved_paddle_device = requested
        logger.info("PaddleOCR device=%s (GPU)", requested)
        return requested

    if not _env_bool("OCR_DEVICE_FALLBACK", default=True):
        raise RuntimeError(
            "OCR_DEVICE meminta GPU tetapi PaddlePaddle tidak dikompilasi CUDA "
            "atau tidak ada GPU. Pasang paddlepaddle-gpu (lihat "
            + PADDLE_INSTALL_URL
            + ") atau set OCR_DEVICE=cpu."
        )

    _resolved_paddle_device = "cpu"
    logger.warning(
        "OCR_DEVICE default GPU, tetapi CUDA tidak siap "
        "(compiled_with_cuda=%s, cuda_device_count=%s). Fallback ke CPU. "
        "Pasang paddlepaddle-gpu + NVIDIA driver, atau set OCR_DEVICE=cpu.",
        cuda.get("compiled_with_cuda"),
        cuda.get("cuda_device_count"),
    )
    return _resolved_paddle_device


def ocr_fast_unavailable_detail(*, exc: BaseException | None = None) -> dict[str, Any]:
    """Payload 503 untuk PP-OCRv6 (ocr-fast / pipeline ocr_mode=fast)."""
    detail: dict[str, Any] = {
        "code": "OCR_FAST_UNAVAILABLE",
        "message": OCR_FAST_UNAVAILABLE,
        "engine": "pp-ocrv6",
        "links": {"paddle_install": PADDLE_INSTALL_URL},
        "install": "pip install -r requirements.txt",
        "env": [
            "OCR_DEVICE",
            "OCR_DEVICE_FALLBACK",
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
