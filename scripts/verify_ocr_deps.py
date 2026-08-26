#!/usr/bin/env python3
"""Cek dependensi untuk preprocessing + PaddleOCR-VL. Keluar 0 jika Paddle terdeteksi; 1 jika OCR VL tidak siap."""

from __future__ import annotations

import sys


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
            print("OCR device default: gpu (akan fallback ke CPU — pasang paddlepaddle-gpu)")
    except ImportError:
        print("PaddlePaddle: TIDAK terpasang (wajib untuk PaddleOCR-VL layout + VL native).")
        if sys.version_info >= (3, 14):
            print(
                "Python 3.14+ umumnya belum punya wheel resmi paddlepaddle. "
                "Buat venv dengan Python 3.11 atau 3.12, lalu: pip install -r requirements.txt"
            )
        else:
            print(
                "Pasang mengikuti: https://www.paddlepaddle.org.cn/install/quick "
                "(CPU/GPU sesuai OS). Lalu: pip install paddlepaddle"
            )
        return 1

    print("Semua cek dasar OK (Paddle + paddleocr + stack app).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
