#!/usr/bin/env python3
"""Cek dependensi untuk preprocessing + PaddleOCR. Keluar 0 jika stack siap.

OCR_VERIFY_ALLOW_NO_CUDA=1: image GPU boleh gagal import paddle saat build
(python:slim tidak punya libcuda; import berhasil di server dengan --gpus all).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    missing: list[str] = []

    for mod in ("cv2", "numpy", "fastapi", "paddleocr"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print("Tidak terpasang:", ", ".join(missing))
        print("Jalankan: pip install -r requirements.txt")
        return 1

    try:
        import paddle

        print("PaddlePaddle:", paddle.__version__)
        compiled = bool(paddle.device.is_compiled_with_cuda())
        count = int(paddle.device.cuda.device_count()) if compiled else 0
        print("CUDA compiled:", compiled)
        print("CUDA devices:", count)
        if compiled and count > 0:
            print("OCR device default: gpu:0")
        else:
            print("OCR device default: gpu (akan fallback ke CPU bila tidak ada CUDA)")
    except Exception as e:
        allow = _env_flag("OCR_VERIFY_ALLOW_NO_CUDA")
        so_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "paddle" / "base" / "libpaddle.so"
        print(f"Import paddle gagal: {e}")
        if allow and so_path.is_file():
            print(
                "OCR_VERIFY_ALLOW_NO_CUDA=1 — wheel Paddle ada, import dilewati "
                "(normal saat build image GPU tanpa libcuda)."
            )
            print("Semua cek dasar OK (paddleocr + stack app; paddle diuji di server).")
            return 0
        print("PaddlePaddle: TIDAK terpasang atau tidak bisa di-load.")
        return 1

    print("Semua cek dasar OK (Paddle + paddleocr + stack app).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
