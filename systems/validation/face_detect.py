"""Deteksi wajah frontal — kompatibel OpenCV 4 (Haar) dan 5 (YuNet)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np

_HAAR: Final[str] = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_YUNET_MODEL: Final[Path] = (
    Path(__file__).resolve().parent / "data" / "face_detection_yunet_2023mar.onnx"
)

_haar_cascade: object | None = None
_yunet_detector: object | None = None
_face_backend: str | None = None


class FaceDetectionUnavailable(RuntimeError):
    """Tidak ada backend deteksi wajah yang tersedia di instalasi OpenCV ini."""


def face_detection_backend() -> str:
    """`haar`, `yunet`, atau `none`."""
    global _face_backend
    if _face_backend is None:
        if hasattr(cv2, "CascadeClassifier"):
            _face_backend = "haar"
        elif _YUNET_MODEL.is_file() and hasattr(cv2, "FaceDetectorYN_create"):
            _face_backend = "yunet"
        else:
            _face_backend = "none"
    return _face_backend


def _ensure_backend() -> str:
    backend = face_detection_backend()
    if backend == "none":
        raise FaceDetectionUnavailable(
            "Deteksi wajah tidak tersedia: OpenCV ini tidak punya CascadeClassifier "
            f"dan model YuNet tidak ditemukan di {_YUNET_MODEL}."
        )
    return backend


def _scale_for_detection(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return bgr, 1.0
    scale = 1.0
    if min(h, w) < 360:
        scale = 360.0 / float(min(h, w))
        bgr = cv2.resize(
            bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    return bgr, scale


def _map_faces_to_original(
    faces: list[tuple[int, int, int, int]],
    *,
    scale: float,
) -> list[tuple[int, int, int, int]]:
    if scale == 1.0:
        return list(faces)
    inv = 1.0 / scale
    out: list[tuple[int, int, int, int]] = []
    for fx, fy, fw, fh in faces:
        x = max(0, int(fx * inv))
        y = max(0, int(fy * inv))
        fw_i = max(1, int(fw * inv))
        fh_i = max(1, int(fh * inv))
        out.append((x, y, fw_i, fh_i))
    return out


def _haar() -> object:
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(_HAAR)
    return _haar_cascade


def _detect_haar(work_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    wh, ww = gray.shape[:2]
    min_px = max(16, int(min(wh, ww) * 0.03))
    casc = _haar()
    if casc.empty():
        return []
    found = casc.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_px, min_px),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(found) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]


def _yunet() -> object:
    global _yunet_detector
    if _yunet_detector is None:
        _yunet_detector = cv2.FaceDetectorYN.create(str(_YUNET_MODEL), "", (320, 320))
        _yunet_detector.setScoreThreshold(0.55)
        _yunet_detector.setNMSThreshold(0.3)
        _yunet_detector.setTopK(5000)
    return _yunet_detector


def _detect_yunet(work_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    wh, ww = work_bgr.shape[:2]
    det = _yunet()
    det.setInputSize((ww, wh))
    _ok, faces = det.detect(work_bgr)
    if faces is None or len(faces) == 0:
        return []
    out: list[tuple[int, int, int, int]] = []
    for row in faces:
        fx, fy, fw, fh = row[:4]
        out.append((int(fx), int(fy), int(fw), int(fh)))
    return out


def detect_frontal_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Kembalikan bbox wajah `(x, y, w, h)` dalam koordinat gambar asli."""
    if bgr is None or bgr.size == 0:
        return []
    _ensure_backend()
    work_bgr, scale = _scale_for_detection(bgr)
    backend = face_detection_backend()
    if backend == "haar":
        faces = _detect_haar(work_bgr)
    else:
        faces = _detect_yunet(work_bgr)
    return _map_faces_to_original(faces, scale=scale)


def count_frontal_faces(bgr: np.ndarray) -> int:
    return len(detect_frontal_faces(bgr))
