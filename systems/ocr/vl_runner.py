"""Jalankan PaddleOCR-VL-1.6 (satu predict + satu restructure_pages) pada bytes gambar."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_pipeline: Any = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()


def _max_long_side_from_env() -> int | None:
    """OCR_VL_MAX_LONG_SIDE: jika diset (px), gambar diperkecil agar sisi terpanjang ≤ nilai ini."""
    raw = (os.environ.get("OCR_VL_MAX_LONG_SIDE") or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _maybe_downscale_bgr(
    bgr: np.ndarray, max_long_side: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Kurangi resolusi sebelum VL untuk menurunkan puncak RAM/waktu (trade-off kualitas)."""
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    meta: dict[str, Any] = {
        "resized": False,
        "input_hw": {"height": h, "width": w},
    }
    if not max_long_side or max_long_side <= 0:
        return bgr, meta
    long_side = max(h, w)
    if long_side <= max_long_side:
        return bgr, meta
    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    meta.update(
        resized=True,
        scale_rounded=round(scale, 6),
        max_long_side=max_long_side,
        final_hw={"height": new_h, "width": new_w},
    )
    return resized, meta


def _decode_bgr(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Gunakan JPEG, PNG, WebP, atau format umum lainnya.")
    return img


def _plain_text_from_json_res(res: dict[str, Any]) -> str:
    blocks = res.get("parsing_res_list") or []

    def sort_key(b: dict[str, Any]) -> tuple[int, int]:
        o = b.get("block_order")
        if o is None:
            return (1, b.get("block_id", 0))
        return (0, int(o))

    lines: list[str] = []
    for b in sorted(blocks, key=sort_key):
        c = b.get("block_content")
        if isinstance(c, str) and c.strip():
            lines.append(c.strip())
    return "\n\n".join(lines)


def _pipeline_version() -> str:
    raw = (os.environ.get("OCR_VL_PIPELINE_VERSION") or "v1.6").strip()
    return raw if raw in ("v1", "v1.5", "v1.6") else "v1.6"


def _model_label(version: str) -> str:
    if version == "v1":
        return "PaddleOCR-VL"
    if version == "v1.5":
        return "PaddleOCR-VL-1.5"
    return "PaddleOCR-VL-1.6"


def get_vl_pipeline():
    """Singleton PaddleOCR-VL (lazy, default v1.6)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _init_lock:
        if _pipeline is not None:
            return _pipeline
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as e:
            raise RuntimeError(
                'Pasang dependensi OCR: pip install "paddleocr[doc-parser]>=3.5.0"'
            ) from e

        backend = os.environ.get("OCR_VL_BACKEND")
        server_url = os.environ.get("OCR_VL_SERVER_URL")
        version = _pipeline_version()
        from systems.ocr.fast_runner import paddle_mkldnn_enabled

        kw: dict[str, Any] = {
            "pipeline_version": version,
            "use_queues": False,
            # oneDNN rusak di PaddlePaddle 3.x + PIR (lihat fast_runner). Diteruskan ke sub-predictor.
            "enable_mkldnn": paddle_mkldnn_enabled(),
        }
        if backend:
            kw["vl_rec_backend"] = backend
        if server_url:
            kw["vl_rec_server_url"] = server_url

        _pipeline = PaddleOCRVL(**kw)
        return _pipeline


def run_paddleocr_vl(image_bytes: bytes, *, include_full_json: bool = False) -> dict[str, Any]:
    """
    Decode gambar → VL predict → restructure_pages (parsing layout sekali per permintaan).

    Membutuhkan paddlepaddle untuk backend native (layout + VL), kecuali Anda
    mengonfigurasi backend server lewat OCR_VL_BACKEND / OCR_VL_SERVER_URL.
    """
    t_wall0 = time.perf_counter()
    timing: dict[str, Any] = {}

    t0 = time.perf_counter()
    bgr = _decode_bgr(image_bytes)
    timing["decode_image_s"] = round(time.perf_counter() - t0, 3)
    max_side = _max_long_side_from_env()
    t0r = time.perf_counter()
    bgr, resize_meta = _maybe_downscale_bgr(bgr, max_side)
    timing["resize_s"] = round(time.perf_counter() - t0r, 3)
    timing["resize"] = resize_meta
    ih, iw = int(bgr.shape[0]), int(bgr.shape[1])
    timing["vl_input_hw"] = {"height": ih, "width": iw}
    orig = resize_meta.get("input_hw") or {"height": ih, "width": iw}
    timing["input_hw"] = orig

    t1 = time.perf_counter()
    pipe = get_vl_pipeline()
    timing["get_pipeline_s"] = round(time.perf_counter() - t1, 3)
    # Pemanggilan pertama: get_pipeline_s memuat layout + VL (bisa menit); berikutnya ~0 s.

    with _infer_lock:
        t2 = time.perf_counter()
        raw_pages = pipe.predict(bgr)
        timing["predict_s"] = round(time.perf_counter() - t2, 3)
        if not raw_pages:
            raise RuntimeError("PaddleOCR-VL tidak mengembalikan hasil.")

        t3 = time.perf_counter()
        parsed_pages = pipe.restructure_pages(
            raw_pages,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False,
        )
        timing["restructure_pages_s"] = round(time.perf_counter() - t3, 3)

    t4 = time.perf_counter()
    page = parsed_pages[0]
    md = page.markdown
    markdown_text = md.get("markdown_texts") or ""

    json_inner = page.json.get("res") or {}
    plain = _plain_text_from_json_res(json_inner if isinstance(json_inner, dict) else {})
    timing["build_text_markdown_s"] = round(time.perf_counter() - t4, 3)
    timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)

    logger.info(
        "ocr_timing total=%.3fs predict=%.3fs restructure=%.3fs get_pipeline=%.3fs decode=%.3fs hw=%dx%d",
        timing["total_wall_s"],
        timing["predict_s"],
        timing["restructure_pages_s"],
        timing["get_pipeline_s"],
        timing["decode_image_s"],
        iw,
        ih,
    )

    version = _pipeline_version()
    payload: dict[str, Any] = {
        "success": True,
        "model": _model_label(version),
        "pipeline_version": version,
        "markdown": markdown_text,
        "text": plain,
        "timing": timing,
    }
    if include_full_json:
        payload["result_json"] = page.json
    return payload
