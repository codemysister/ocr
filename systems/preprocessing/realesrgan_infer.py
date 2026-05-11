"""
Super-resolusi opsional dengan Real-ESRGAN (PyTorch).

Aktifkan dengan: PREPROCESS_USE_REALESRGAN=1
Dependensi: pip install -r requirements-realesrgan.txt

Model & bobot diunduh otomatis ke models/realesrgan/ pada pemakaian pertama.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

_REPO_MODELS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "models" / "realesrgan"

_infer_lock = threading.Lock()
_upsampler_cache: Any = None
_cache_key: tuple[Any, ...] | None = None


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def realesrgan_import_ok() -> bool:
    try:
        import torch  # noqa: F401
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from basicsr.utils.download_util import load_file_from_url  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401

        return True
    except ImportError:
        return False


def _model_spec(name: str) -> tuple[Any, int, list[str]]:
    from basicsr.archs.rrdbnet_arch import RRDBNet

    n = name.strip()
    if n == "RealESRGAN_x2plus":
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        return net, 2, [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    # default: general x4
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return net, 4, [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ]


def _resolve_model_name() -> str:
    n = os.environ.get("PREPROCESS_REALESRGAN_MODEL", "RealESRGAN_x4plus").strip()
    if n in ("RealESRGAN_x4plus", "RealESRGAN_x2plus"):
        return n
    return "RealESRGAN_x4plus"


def _get_upsampler() -> Any:
    global _upsampler_cache, _cache_key
    import torch
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    model_name = _resolve_model_name()
    tile = int(os.environ.get("PREPROCESS_REALESRGAN_TILE", "400"))
    tile_pad = int(os.environ.get("PREPROCESS_REALESRGAN_TILE_PAD", "10"))
    pre_pad = int(os.environ.get("PREPROCESS_REALESRGAN_PRE_PAD", "0"))
    fp32 = _truthy_env("PREPROCESS_REALESRGAN_FP32", "0")
    gpu_raw = os.environ.get("PREPROCESS_REALESRGAN_GPU_ID", "").strip()
    gpu_id: int | None
    if gpu_raw == "":
        gpu_id = None if not torch.cuda.is_available() else 0
    else:
        gpu_id = int(gpu_raw)

    half = (not fp32) and torch.cuda.is_available()
    key = (model_name, tile, tile_pad, pre_pad, half, gpu_id)
    with _infer_lock:
        if _upsampler_cache is not None and _cache_key == key:
            return _upsampler_cache

        model, netscale, urls = _model_spec(model_name)
        _REPO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = str(_REPO_MODELS_DIR / (urls[0].split("/")[-1]))
        if not Path(model_path).is_file():
            for url in urls:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=str(_REPO_MODELS_DIR),
                    progress=True,
                    file_name=os.path.basename(url),
                )

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            gpu_id=gpu_id,
        )
        _upsampler_cache = upsampler
        _cache_key = key
        return upsampler


def _clamp_max_side(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side or max_side <= 0:
        return bgr, 1.0
    s = max_side / m
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), s


def maybe_apply_realesrgan_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Jalankan Real-ESRGAN pada BGR 8-bit jika diaktifkan dan dependensi ada.
    Mengembalikan (gambar, metadata); gambar tidak berubah jika langkah dilewati.
    """
    meta: dict[str, Any] = {
        "realesrgan_applied": False,
        "realesrgan_model": None,
        "realesrgan_outscale": None,
        "realesrgan_reason": None,
    }
    if not _truthy_env("PREPROCESS_USE_REALESRGAN", "0"):
        meta["realesrgan_reason"] = "disabled (set PREPROCESS_USE_REALESRGAN=1)"
        return bgr, meta

    if not realesrgan_import_ok():
        meta["realesrgan_reason"] = "missing deps (pip install -r requirements-realesrgan.txt)"
        return bgr, meta

    if bgr.ndim != 3 or bgr.shape[2] not in (3, 4):
        meta["realesrgan_reason"] = "expected BGR/BGRA"
        return bgr, meta

    work = bgr
    if work.shape[2] == 4:
        work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)

    max_in = int(os.environ.get("PREPROCESS_REALESRGAN_MAX_SIDE_IN", "640"))
    work, scale_in = _clamp_max_side(work, max_in)
    outscale = float(os.environ.get("PREPROCESS_REALESRGAN_OUTSCALE", "2"))

    try:
        upsampler = _get_upsampler()
        model_name = _resolve_model_name()
        meta["realesrgan_model"] = model_name
        meta["realesrgan_outscale"] = outscale
        meta["realesrgan_max_side_in"] = max_in
        meta["realesrgan_input_scale"] = round(scale_in, 4)

        with _infer_lock:
            out, _ = upsampler.enhance(work, outscale=outscale)

        if out is None or out.size == 0:
            meta["realesrgan_reason"] = "empty output"
            return bgr, meta

        meta["realesrgan_applied"] = True
        meta["realesrgan_reason"] = None
        return out, meta
    except Exception as e:  # noqa: BLE001 — tetap lanjut pipeline OpenCV
        meta["realesrgan_reason"] = f"error: {type(e).__name__}: {e}"
        return bgr, meta


def realesrgan_status_for_health() -> dict[str, Any]:
    return {
        "use_realesrgan_env": _truthy_env("PREPROCESS_USE_REALESRGAN", "0"),
        "deps_importable": realesrgan_import_ok(),
        "model_dir": str(_REPO_MODELS_DIR),
        "model": _resolve_model_name(),
        "outscale_default": float(os.environ.get("PREPROCESS_REALESRGAN_OUTSCALE", "2")),
        "max_side_in_default": int(os.environ.get("PREPROCESS_REALESRGAN_MAX_SIDE_IN", "640")),
    }
