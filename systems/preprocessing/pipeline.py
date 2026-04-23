"""Pipeline preprocessing gambar untuk input OCR (satu pemanggilan = seluruh langkah)."""

from __future__ import annotations

import io
from typing import Final

import cv2
import numpy as np

MAX_SIDE: Final[int] = 2400
MIN_SIDE_TARGET: Final[int] = 900


def _decode_image(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan format didukung (JPEG, PNG, WebP, dll.).")
    return img


def _maybe_resize(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    max_side = max(h, w)
    min_side = min(h, w)
    if max_side > MAX_SIDE:
        scale = MAX_SIDE / max_side
        return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min_side < MIN_SIDE_TARGET:
        scale = MIN_SIDE_TARGET / min_side
        scale = min(scale, 3.0)
        return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return bgr


def _odd_kernel_from_fraction(min_side: int, frac: float, cap: int) -> int:
    k = int(round(min_side * frac)) | 1
    k = min(cap, max(3, k))
    if k % 2 == 0:
        k = min(cap, k + 1)
    k = min(k, min_side - 2) if min_side > 4 else 3
    if k < 3:
        k = 3
    if k % 2 == 0:
        k -= 1
    return max(3, k)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_warp(bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_quad_points(pts)
    (tl, tr, br, bl) = rect
    wa = float(np.hypot(br[0] - bl[0], br[1] - bl[1]))
    wb = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
    max_w = max(int(wa), int(wb))
    ha = float(np.hypot(tr[0] - br[0], tr[1] - br[1]))
    hb = float(np.hypot(tl[0] - bl[0], tl[1] - bl[1]))
    max_h = max(int(ha), int(hb))
    max_w = max(max_w, 120)
    max_h = max(max_h, 80)
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, m, (max_w, max_h), flags=cv2.INTER_LINEAR)


def _aspect_ok(w: int, h: int) -> bool:
    if h <= 0 or w <= 0:
        return False
    r = max(w / h, h / w)
    return 1.15 <= r <= 2.35


def _try_extract_card_warp(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Coba isolasi bidang kartu (KTP/dokumen persegi panjang) dari latar foto.
    Menggunakan kecerahan L (LAB) + Otsu + morfologi + minAreaRect.
    """
    h0, w0 = bgr.shape[:2]
    max_det = 900
    sc = min(1.0, max_det / max(h0, w0))
    sw, sh = int(w0 * sc), int(h0 * sc)
    if sw < 40 or sh < 40:
        return bgr, False

    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch = cv2.GaussianBlur(lab[:, :, 0], (13, 13), 0)

    best_warp: np.ndarray | None = None
    for invert in (False, True):
        _, th = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            th = cv2.bitwise_not(th)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        th2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=2)
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        th2 = cv2.dilate(th2, k_d, iterations=1)

        cnts, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(c)
            if area < sh * sw * 0.06:
                break
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(np.round(box))
            box_full = (box.astype(np.float32) / sc).astype(np.float32)
            warped = _four_point_warp(bgr, box_full)
            ww, hh = warped.shape[1], warped.shape[0]
            if _aspect_ok(ww, hh) and ww * hh >= w0 * h0 * 0.04:
                best_warp = warped
                break
        if best_warp is not None:
            break

    if best_warp is None:
        return bgr, False
    return best_warp, True


def _enhance_grayscale_for_ocr(gray: np.ndarray) -> np.ndarray:
    """
    Perjelas teks tanpa binarisasi adaptif global (menghindari tebal blobby & watermark keras).
    Output 8-bit grayscale siap OCR.
    """
    h, w = gray.shape[:2]
    min_side = min(h, w)

    blur_k = _odd_kernel_from_fraction(min_side, 0.11, 121)
    illum = cv2.GaussianBlur(gray, (blur_k, blur_k), 0).astype(np.float32)
    illum = np.maximum(illum, 20.0)
    flat = np.clip(gray.astype(np.float32) / illum * 252.0, 0, 255).astype(np.uint8)

    smooth = cv2.bilateralFilter(flat, d=5, sigmaColor=35, sigmaSpace=35)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    eq = clahe.apply(smooth)

    blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=0.9)
    sharp = cv2.addWeighted(eq, 1.22, blur, -0.22, 0)
    return sharp


def _preprocess_work_bgr(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    work = _maybe_resize(bgr)
    cropped, card_warped = _try_extract_card_warp(work)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return _enhance_grayscale_for_ocr(gray), card_warped


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    1) Resize batas.
    2) Deteksi bidang kartu + warp (menghindari tekstur kain/latar foto).
    3) Perkuat grayscale 8-bit untuk OCR (tanpa binarisasi keras).
    """
    out, _ = _preprocess_work_bgr(bgr)
    return out


def preprocess_image_bytes(data: bytes) -> tuple[bytes, dict]:
    """
    Decode bytes gambar → pipeline OCR → PNG bytes (grayscale 8-bit).

    Returns:
        (png_bytes, metadata) metadata berisi dimensi output.
    """
    bgr = _decode_image(data)
    out, card_warped = _preprocess_work_bgr(bgr)
    ok, encoded = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("Gagal mengenkode hasil ke PNG.")
    meta = {
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
        "channels": 1,
        "encoding": "grayscale_8bit",
        "card_warped": card_warped,
    }
    return encoded.tobytes(), meta


def preprocess_to_png_buffer(data: bytes) -> io.BytesIO:
    png_bytes, _ = preprocess_image_bytes(data)
    buf = io.BytesIO(png_bytes)
    buf.seek(0)
    return buf
