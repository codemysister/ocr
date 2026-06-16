#!/usr/bin/env python3
"""Unduh / preload weight PaddleOCR-VL-1.6 (layout + VL native).

Jalankan dari root repo:
  .venv/bin/python scripts/download_vl_16.py

Opsional:
  OCR_VL_PIPELINE_VERSION=v1.6  (default)
  OCR_VL_BACKEND / OCR_VL_SERVER_URL  — inferensi VL jarak jauh (layout tetap lokal)
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main() -> int:
    version = (os.environ.get("OCR_VL_PIPELINE_VERSION") or "v1.6").strip()
    print(f"Memuat pipeline PaddleOCR-VL ({version}) — unduh weight bisa beberapa menit…")
    t0 = time.perf_counter()
    try:
        from systems.ocr.vl_runner import get_vl_pipeline, _model_label

        pipe = get_vl_pipeline()
        elapsed = time.perf_counter() - t0
        print(f"OK: {_model_label(version)} siap ({elapsed:.1f}s)")
        print(f"Pipeline: {type(pipe).__name__}")
        return 0
    except Exception as e:
        print(f"GAGAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
