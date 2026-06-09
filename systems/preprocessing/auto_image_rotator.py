"""
Logika setara `auto-image-rotator/rotate.py` (dsposito): untuk cycle 0..3 uji rotasi
kelipatan 90° CCW pada gambar; **orientasi pertama** yang punya wajah frontal dipilih.

Implementasi asli memakai PIL + dlib dan salah memakai COLOR_BGR2GRAY pada array RGB.
Di sini: OpenCV BGR + detektor Haar (default) atau dlib opsional.

Referensi: repo `auto-image-rotator/` di root proyek (rotate.py).
"""

from __future__ import annotations

import os
from typing import Callable, Final

import cv2
import numpy as np

_HAAR: Final[str] = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar_cascade: cv2.CascadeClassifier | None = None
_dlib_detector: object | None = None


def _auto_image_rotator_enabled() -> bool:
    # Default mati: deteksi wajah sering salah pada dokumen/KK (false positive / tidak terdeteksi
    # di k=0) → orientasi "terbalik" atau mutar padahal gambar sudah normal.
    v = (os.environ.get("PREPROCESS_AUTO_IMAGE_ROTATOR") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _backend() -> str:
    v = (os.environ.get("PREPROCESS_AUTO_IMAGE_ROTATOR_BACKEND") or "haar").strip().lower()
    return v if v in ("haar", "dlib") else "haar"


def _haar() -> cv2.CascadeClassifier:
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(_HAAR)
    return _haar_cascade


def _dlib_det():
    global _dlib_detector
    if _dlib_detector is None:
        import dlib  # type: ignore[import-untyped]

        _dlib_detector = dlib.get_frontal_face_detector()
    return _dlib_detector


def _face_count_haar(gray: np.ndarray) -> int:
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0
    g = gray
    if min(h, w) < 360:
        s = 360.0 / float(min(h, w))
        g = cv2.resize(
            gray,
            (max(1, int(w * s)), max(1, int(h * s))),
            interpolation=cv2.INTER_LINEAR,
        )
    gh, gw = g.shape[:2]
    min_px = max(16, int(min(gh, gw) * 0.035))
    casc = _haar()
    if casc.empty():
        return 0
    faces = casc.detectMultiScale(
        g,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_px, min_px),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return int(len(faces))


def _face_count_dlib(gray: np.ndarray) -> int:
    if gray.size == 0:
        return 0
    return int(len(_dlib_det()(gray, 0)))


def _resolve_counter() -> tuple[Callable[[np.ndarray], int], str]:
    if _backend() == "dlib":
        try:
            _dlib_det()
            return _face_count_dlib, "dlib"
        except Exception:
            return _face_count_haar, "haar_fallback_from_dlib"
    return _face_count_haar, "haar"


def maybe_auto_image_rotator_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """
    Setara `Rotator.analyze_image`: uji k=0,1,2,3 putaran 90° CCW dari **asal**;
    kembalikan orientasi pertama dengan ≥1 wajah, atau asli bila tidak ada wajah di semua orientasi.
    """
    meta: dict[str, object] = {
        "auto_image_rotator_applied": False,
        "auto_image_rotator_90ccw_steps": 0,
        "auto_image_rotator_backend": "haar",
        "auto_image_rotator_skipped_reason": "",
    }
    if not _auto_image_rotator_enabled():
        meta["auto_image_rotator_skipped_reason"] = "disabled"
        return bgr, meta

    h0, w0 = bgr.shape[:2]
    if h0 < 24 or w0 < 24:
        meta["auto_image_rotator_skipped_reason"] = "too_small"
        return bgr, meta

    counter, backend_label = _resolve_counter()
    meta["auto_image_rotator_backend"] = backend_label

    for k in range(4):
        test = bgr
        for _ in range(k):
            test = cv2.rotate(test, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        try:
            n = int(counter(gray))
        except Exception:
            meta["auto_image_rotator_skipped_reason"] = "detector_error"
            return bgr, meta
        if n > 0:
            if k == 0:
                meta["auto_image_rotator_skipped_reason"] = "faces_at_upright"
                return bgr, meta
            meta["auto_image_rotator_applied"] = True
            meta["auto_image_rotator_90ccw_steps"] = k
            return test, meta

    meta["auto_image_rotator_skipped_reason"] = "no_face_any_orientation"
    return bgr, meta
