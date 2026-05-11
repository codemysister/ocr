"""OCR ringan: PaddleOCR det + rec (PP-OCRv5), tanpa PaddleOCR-VL / layout dokumen.

Default untuk lang ``en`` dan ``latin``: deteksi mobile + pengenalan server (lebih akurat dari full
mobile, lebih ringan dari full server atau VL). Ganti pasangan lewat OCR_FAST_DET_MODEL /
OCR_FAST_REC_MODEL.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ocr: Any = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()


def _decode_bgr(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Gunakan JPEG, PNG, WebP, atau format umum lainnya.")
    return img


def _get_paddle_ocr() -> Any:
    global _ocr
    if _ocr is not None:
        return _ocr
    with _init_lock:
        if _ocr is not None:
            return _ocr
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                'Pasang dependensi OCR: pip install "paddleocr[doc-parser]>=3.5.0"'
            ) from e

        lang = os.environ.get("OCR_FAST_LANG", "en").strip() or "en"
        det_env = os.environ.get("OCR_FAST_DET_MODEL", "").strip()
        rec_env = os.environ.get("OCR_FAST_REC_MODEL", "").strip()

        kw: dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

        # Jika hanya set det tanpa rec, Paddle mengabaikan `lang` untuk rec — set pasangan agar konsisten.
        if det_env or rec_env:
            if det_env:
                kw["text_detection_model_name"] = det_env
            if rec_env:
                kw["text_recognition_model_name"] = rec_env
        elif lang in ("en", "latin"):
            # Seimbang: deteksi mobile (ringan), pengenalan server (lebih akurat). Full server: set env di bawah.
            kw["text_detection_model_name"] = "PP-OCRv5_mobile_det"
            kw["text_recognition_model_name"] = "PP-OCRv5_server_rec"
        # else: biarkan Paddle memilih model default untuk `lang`

        lim = os.environ.get("OCR_FAST_DET_LIMIT_SIDE_LEN", "").strip()
        if lim.isdigit():
            kw["text_det_limit_side_len"] = int(lim)

        _ocr = PaddleOCR(**kw)
        return _ocr


def _result_to_lines(ocr_result: Any) -> tuple[list[dict[str, Any]], str]:
    """Ambil rec_texts + skor + poligon dari satu halaman OCRResult."""
    lines: list[dict[str, Any]] = []
    if ocr_result is None:
        return lines, ""

    texts = list(ocr_result["rec_texts"] or [])
    try:
        scores = list(ocr_result["rec_scores"] or [])
    except KeyError:
        scores = []
    polys: list[Any] = []
    try:
        rp = ocr_result["rec_polys"]
        if rp is not None:
            polys = list(rp)
    except KeyError:
        try:
            dp = ocr_result["dt_polys"]
            if dp is not None:
                polys = list(dp)
        except KeyError:
            polys = []

    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            continue
        sc = float(scores[i]) if i < len(scores) else None
        poly: Any = None
        if i < len(polys):
            p = polys[i]
            poly = p.tolist() if hasattr(p, "tolist") else p
        lines.append({"text": t.strip(), "score": sc, "polygon": poly})

    full = "\n".join(x["text"] for x in lines)
    return lines, full


def run_paddleocr_fast(image_bytes: bytes) -> dict[str, Any]:
    """
    Deteksi + recognition klasik (bukan VL). Lebih ringan dari PaddleOCR-VL untuk banyak kasus.
    """
    t0 = time.perf_counter()
    timing: dict[str, Any] = {}

    t_dec = time.perf_counter()
    bgr = _decode_bgr(image_bytes)
    timing["decode_image_s"] = round(time.perf_counter() - t_dec, 3)
    ih, iw = int(bgr.shape[0]), int(bgr.shape[1])
    timing["input_hw"] = {"height": ih, "width": iw}

    t_get = time.perf_counter()
    ocr = _get_paddle_ocr()
    timing["get_engine_s"] = round(time.perf_counter() - t_get, 3)

    with _infer_lock:
        t_pred = time.perf_counter()
        raw = ocr.predict(bgr)
        timing["predict_s"] = round(time.perf_counter() - t_pred, 3)

    t_build = time.perf_counter()
    if not raw:
        raise RuntimeError("PaddleOCR (fast) tidak mengembalikan hasil.")

    page0 = raw[0]
    line_items, plain = _result_to_lines(page0)
    markdown_lines = [f"- {x['text']}" for x in line_items]
    markdown = "\n".join(markdown_lines) if markdown_lines else ""

    timing["build_output_s"] = round(time.perf_counter() - t_build, 3)
    timing["total_wall_s"] = round(time.perf_counter() - t0, 3)

    logger.info(
        "ocr_fast total=%.3fs predict=%.3fs get_engine=%.3fs lines=%d hw=%dx%d",
        timing["total_wall_s"],
        timing["predict_s"],
        timing["get_engine_s"],
        len(line_items),
        iw,
        ih,
    )

    lang = os.environ.get("OCR_FAST_LANG", "en").strip() or "en"
    custom = bool(
        (os.environ.get("OCR_FAST_DET_MODEL") or "").strip()
        or (os.environ.get("OCR_FAST_REC_MODEL") or "").strip()
    )
    if custom:
        model_str = f"PP-OCRv5 det+rec (lang={lang}, OCR_FAST_* custom)"
    elif lang in ("en", "latin"):
        model_str = f"PP-OCRv5 mobile_det+server_rec (lang={lang})"
    else:
        model_str = f"PP-OCRv5 det+rec (lang={lang}, Paddle default)"
    return {
        "success": True,
        "mode": "paddleocr_fast",
        "model": model_str,
        "markdown": markdown,
        "text": plain,
        "lines": line_items,
        "timing": timing,
    }
