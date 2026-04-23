"""Jalankan PaddleOCR-VL-1.5 (satu predict + satu restructure_pages) pada bytes gambar."""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

import cv2
import numpy as np
from PIL import Image

_pipeline: Any = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()


def _decode_bgr(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Gunakan JPEG, PNG, WebP, atau format umum lainnya.")
    return img


def _np_bgr_to_png_b64(arr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Gagal mengenkode gambar hasil OCR ke PNG.")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def _any_image_to_png_b64(img: Any) -> str:
    if isinstance(img, np.ndarray):
        return _np_bgr_to_png_b64(img)
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        rgb = img.convert("RGB")
        rgb.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    raise TypeError(f"Tipe gambar tidak didukung: {type(img)}")


def _collect_vis_images(result: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, img in result.img.items():
        try:
            out[key] = _any_image_to_png_b64(img)
        except (TypeError, ValueError):
            continue
    return out


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


def get_vl_pipeline():
    """Singleton PaddleOCR-VL-1.5 (lazy)."""
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
        kw: dict[str, Any] = {
            "pipeline_version": "v1.5",
            "use_queues": False,
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
    bgr = _decode_bgr(image_bytes)
    pipe = get_vl_pipeline()

    with _infer_lock:
        raw_pages = pipe.predict(bgr)
        if not raw_pages:
            raise RuntimeError("PaddleOCR-VL tidak mengembalikan hasil.")
        parsed_pages = pipe.restructure_pages(
            raw_pages,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False,
        )

    page = parsed_pages[0]
    md = page.markdown
    markdown_text = md.get("markdown_texts") or ""

    json_inner = page.json.get("res") or {}
    plain = _plain_text_from_json_res(json_inner if isinstance(json_inner, dict) else {})
    vis = _collect_vis_images(page)

    payload: dict[str, Any] = {
        "success": True,
        "model": "PaddleOCR-VL-1.5",
        "pipeline_version": "v1.5",
        "markdown": markdown_text,
        "text": plain,
        "ocr_images_png_base64": vis,
    }
    if include_full_json:
        payload["result_json"] = page.json
    return payload
