"""OCR ringan: PaddleOCR det + rec (PP-OCRv6), tanpa PaddleOCR-VL / layout dokumen.

Default tier ``medium`` (PP-OCRv6_medium det+rec, akurasi maksimal). Semua modul pipeline Paddle
(orientasi halaman, unwarp dokumen, orientasi baris teks) aktif default. Matikan lewat
OCR_FAST_DOC_ORIENTATION=0, OCR_FAST_DOC_UNWARPING=0, OCR_FAST_TEXTLINE_ORIENTATION=0.
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

PP_OCR_V6_SMALL_DET = "PP-OCRv6_small_det"
PP_OCR_V6_MEDIUM_DET = "PP-OCRv6_medium_det"
PP_OCR_V6_MEDIUM_REC = "PP-OCRv6_medium_rec"
PP_OCR_V6_SMALL_REC = "PP-OCRv6_small_rec"
PP_OCR_V6_TINY_DET = "PP-OCRv6_tiny_det"
PP_OCR_V6_TINY_REC = "PP-OCRv6_tiny_rec"

PP_OCR_V6_TIERS: dict[str, tuple[str, str, str]] = {
    "balanced": (PP_OCR_V6_SMALL_DET, PP_OCR_V6_MEDIUM_REC, "small_det+medium_rec"),
    "medium": (PP_OCR_V6_MEDIUM_DET, PP_OCR_V6_MEDIUM_REC, "medium_det+medium_rec"),
    "small": (PP_OCR_V6_SMALL_DET, PP_OCR_V6_SMALL_REC, "small_det+small_rec"),
    "tiny": (PP_OCR_V6_TINY_DET, PP_OCR_V6_TINY_REC, "tiny_det+tiny_rec"),
}

_ocr_cache: dict[str, Any] = {}
_init_lock = threading.Lock()
_infer_lock = threading.Lock()


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def paddle_mkldnn_enabled() -> bool:
    """MKLDNN (oneDNN) default MATI: pada PaddlePaddle 3.x + PIR, PP-OCRv6 memicu
    'ConvertPirAttribute2RuntimeAttribute not support' di jalur onednn. Set
    OCR_ENABLE_MKLDNN=1 untuk mengaktifkan kembali (mis. bila wheel sudah diperbaiki)."""
    return _env_bool("OCR_ENABLE_MKLDNN", default=False)


def paddle_fast_module_flags() -> dict[str, bool]:
    """Flag modul Paddle (orientation / unwarp / textline) untuk PP-OCRv6 fast."""
    return {
        "use_doc_orientation_classify": _env_bool(
            "OCR_FAST_DOC_ORIENTATION", default=True
        ),
        "use_doc_unwarping": _env_bool("OCR_FAST_DOC_UNWARPING", default=True),
        "use_textline_orientation": _env_bool(
            "OCR_FAST_TEXTLINE_ORIENTATION", default=True
        ),
    }


def _decode_bgr(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Gunakan JPEG, PNG, WebP, atau format umum lainnya.")
    return img


def list_pp_ocr_tiers() -> list[str]:
    return list(PP_OCR_V6_TIERS.keys())


def _resolve_fast_models(pp_ocr_tier: str | None) -> tuple[str, str, str]:
    """
    Kembalikan (det_model, rec_model, tier_id).
    Env OCR_FAST_DET_MODEL / OCR_FAST_REC_MODEL menang atas tier.
    """
    det_env = os.environ.get("OCR_FAST_DET_MODEL", "").strip()
    rec_env = os.environ.get("OCR_FAST_REC_MODEL", "").strip()
    if det_env or rec_env:
        return (
            det_env or PP_OCR_V6_MEDIUM_DET,
            rec_env or PP_OCR_V6_MEDIUM_REC,
            "custom",
        )
    tier = (pp_ocr_tier or os.environ.get("OCR_FAST_TIER") or "medium").strip().lower()
    if tier not in PP_OCR_V6_TIERS:
        tier = "medium"
    det, rec, _ = PP_OCR_V6_TIERS[tier]
    return det, rec, tier


def _get_paddle_ocr(*, pp_ocr_tier: str | None = None) -> Any:
    lang = os.environ.get("OCR_FAST_LANG", "en").strip() or "en"
    det, rec, tier_id = _resolve_fast_models(pp_ocr_tier)
    mod_flags = paddle_fast_module_flags()
    lim = (os.environ.get("OCR_FAST_DET_LIMIT_SIDE_LEN") or "").strip()
    mkldnn = paddle_mkldnn_enabled()
    cache_key = (
        f"{lang}|{det}|{rec}|ori={int(mod_flags['use_doc_orientation_classify'])}"
        f"|unwarp={int(mod_flags['use_doc_unwarping'])}"
        f"|tline={int(mod_flags['use_textline_orientation'])}|lim={lim or '-'}"
        f"|mkldnn={int(mkldnn)}"
    )
    cached = _ocr_cache.get(cache_key)
    if cached is not None:
        return cached
    with _init_lock:
        cached = _ocr_cache.get(cache_key)
        if cached is not None:
            return cached
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            import paddle  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "PaddlePaddle belum terpasang. Pasang wheel untuk OS dan Python Anda "
                "(lihat https://www.paddlepaddle.org.cn/install/quick)."
            ) from e
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                'Pasang dependensi OCR: pip install "paddleocr[doc-parser]>=3.6.0"'
            ) from e

        kw: dict[str, Any] = {
            "lang": lang,
            **mod_flags,
            "text_detection_model_name": det,
            "text_recognition_model_name": rec,
            "enable_mkldnn": mkldnn,
        }

        if lim.isdigit():
            kw["text_det_limit_side_len"] = int(lim)

        ocr = PaddleOCR(**kw)
        _ocr_cache[cache_key] = ocr
        return ocr


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


def run_paddleocr_fast(
    image_bytes: bytes,
    *,
    pp_ocr_tier: str | None = None,
) -> dict[str, Any]:
    """
    Deteksi + recognition klasik (bukan VL). Lebih ringan dari PaddleOCR-VL untuk banyak kasus.
    pp_ocr_tier: balanced | medium | small | tiny (lihat PP_OCR_V6_TIERS).
    """
    t0 = time.perf_counter()
    timing: dict[str, Any] = {}

    t_dec = time.perf_counter()
    bgr = _decode_bgr(image_bytes)
    timing["decode_image_s"] = round(time.perf_counter() - t_dec, 3)
    ih, iw = int(bgr.shape[0]), int(bgr.shape[1])
    timing["input_hw"] = {"height": ih, "width": iw}

    t_get = time.perf_counter()
    det, rec, tier_id = _resolve_fast_models(pp_ocr_tier)
    ocr = _get_paddle_ocr(pp_ocr_tier=pp_ocr_tier)
    timing["get_engine_s"] = round(time.perf_counter() - t_get, 3)
    timing["pp_ocr_tier"] = tier_id
    timing["paddle_modules"] = paddle_fast_module_flags()

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
    if tier_id == "custom":
        model_str = f"PP-OCRv6 custom ({det}+{rec}, lang={lang})"
    else:
        model_str = f"PP-OCRv6 tier={tier_id} ({det}+{rec}, lang={lang})"
    return {
        "success": True,
        "mode": "paddleocr_fast",
        "model": model_str,
        "pp_ocr_tier": tier_id,
        "paddle_modules": paddle_fast_module_flags(),
        "markdown": markdown,
        "text": plain,
        "lines": line_items,
        "timing": timing,
    }
