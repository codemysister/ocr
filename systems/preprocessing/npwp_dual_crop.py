"""Crop panel depan NPWP saat foto berisi depan+belakang (horizontal atau vertikal)."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

_NPWP_PROFILES = frozenset({"npwp", "pemadanan_npwp"})


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def npwp_dual_crop_enabled() -> bool:
    return _env_bool("PREPROCESS_NPWP_DUAL_CROP", default=True)


def _npwp_portrait_tall(h: int, w: int) -> bool:
    """Foto portrait tinggi: dua kartu distack atas-bawah."""
    return h / max(w, 1) >= 1.25


def _npwp_portrait_frame(h: int, w: int) -> bool:
    """Frame portrait (bukan landscape lebar)."""
    return w / max(h, 1) < 1.15


def _npwp_landscape_dual_aspect(w: int, h: int) -> bool:
    """Thumbnail / foto lebar: dua kartu NPWP berdampingan (depan kiri, belakang kanan)."""
    return w / max(h, 1) >= 1.85


def npwp_landscape_dual_aspect(w: int, h: int) -> bool:
    return _npwp_landscape_dual_aspect(w, h)


def _column_split_valley(
    smooth: np.ndarray,
    w: int,
    h: int,
    *,
    lo_frac: float,
    hi_frac: float,
    min_peak_h_frac: float,
    max_valley_peak_frac: float,
    min_side_ink_peak_frac: float,
    min_valley_depth: float,
) -> int | None:
    lo, hi = int(w * lo_frac), int(w * hi_frac)
    if hi - lo < 12:
        return None
    region = smooth[lo:hi]
    min_idx = int(region.argmin()) + lo
    min_val = float(smooth[min_idx])
    left_peak = float(smooth[lo:min_idx].max()) if min_idx > lo else 0.0
    right_peak = float(smooth[min_idx:hi].max()) if min_idx < hi else 0.0
    peak = max(left_peak, right_peak, 1.0)

    if peak < h * min_peak_h_frac:
        return None
    if min_val > peak * max_valley_peak_frac:
        return None

    left_ink = float(smooth[:min_idx].mean()) if min_idx > 0 else 0.0
    right_ink = float(smooth[min_idx:].mean()) if min_idx < w else 0.0
    if min(left_ink, right_ink) < peak * min_side_ink_peak_frac:
        return None
    if (peak - min_val) / peak < min_valley_depth:
        return None

    return min_idx


def detect_npwp_dual_split_x(bgr: np.ndarray) -> int | None:
    """
    Deteksi garis pemisah vertikal antara kartu NPWP depan (kiri) dan belakang (kanan).
    Mendukung foto portrait (dua kartu dalam frame tinggi) dan landscape lebar (thumbnail kecil).
    Mengembalikan indeks kolom split (crop [:, :split]) atau None.
    """
    h, w = bgr.shape[:2]
    aspect = w / max(h, 1)
    landscape_dual = _npwp_landscape_dual_aspect(w, h)

    if landscape_dual:
        if w < 400 or h < 100:
            return None
    else:
        if w < 700 or h < 350:
            return None
        if aspect > 1.15:
            return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 200).astype(np.float32)
    col_sum = ink.sum(axis=0)
    k = max(5, w // 80) | 1
    smooth = np.convolve(col_sum, np.ones(k, dtype=np.float32) / k, mode="same")

    if landscape_dual:
        return _column_split_valley(
            smooth,
            w,
            h,
            lo_frac=0.38,
            hi_frac=0.62,
            min_peak_h_frac=0.05,
            max_valley_peak_frac=0.50,
            min_side_ink_peak_frac=0.10,
            min_valley_depth=0.30,
        )

    return _column_split_valley(
        smooth,
        w,
        h,
        lo_frac=0.32,
        hi_frac=0.68,
        min_peak_h_frac=0.06,
        max_valley_peak_frac=0.42,
        min_side_ink_peak_frac=0.12,
        min_valley_depth=0.35,
    )


def detect_npwp_dual_split_y(bgr: np.ndarray) -> int | None:
    """
    Deteksi garis pemisah horizontal antara kartu NPWP depan (atas) dan belakang (bawah).
    Layout umum: foto portrait tinggi (status bar + 2 kartu distack).
    """
    h, w = bgr.shape[:2]
    if h < 700 or w < 350:
        return None
    if h / max(w, 1) < 1.25:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 210).astype(np.float32)
    row_density = ink.sum(axis=1) / max(w, 1)
    k = max(5, h // 100) | 1
    smooth = np.convolve(row_density, np.ones(k, dtype=np.float32) / k, mode="same")

    lo, hi = int(h * 0.38), int(h * 0.62)
    if hi - lo < 12:
        return None
    region = smooth[lo:hi]
    min_idx = int(region.argmin()) + lo
    min_val = float(smooth[min_idx])
    top_peak = float(smooth[int(h * 0.05) : min_idx].max()) if min_idx > h * 0.05 else 0.0
    bot_peak = float(smooth[min_idx : int(h * 0.95)].max()) if min_idx < h * 0.95 else 0.0
    peak = max(top_peak, bot_peak, 0.01)

    top_ink = float(smooth[int(h * 0.08) : min_idx].mean()) if min_idx > h * 0.08 else 0.0
    bot_ink = float(smooth[min_idx : int(h * 0.92)].mean()) if min_idx < h * 0.92 else 0.0
    if top_ink < 0.04 or bot_ink < 0.03:
        return None
    if min_val > peak * 0.35:
        return None
    if (peak - min_val) / peak < 0.55:
        return None

    return min_idx


def _vertical_crop_height(h: int, split_y: int | None, *, stacked_fallback: bool) -> int:
    """Tinggi crop panel atas — cukup untuk kartu depan, hindari belakang."""
    if split_y is not None:
        crop_h = max(int(h * 0.50), min(int(h * 0.60), split_y + int(h * 0.08)))
    elif stacked_fallback:
        crop_h = int(h * 0.56)
    else:
        crop_h = int(h * 0.55)
    crop_h = max(int(h * 0.30), min(crop_h, int(h * 0.62)))
    return crop_h


def _horizontal_crop_valid(w: int, h: int, crop_w: int) -> bool:
    """Tolak strip vertikal sempit (sering salah split / rusak OCR)."""
    if crop_w < int(w * 0.40):
        return False
    return crop_w / max(h, 1) >= 0.55


def _vertical_crop_valid(h: int, crop_h: int, split_y: int | None) -> bool:
    if crop_h < int(h * 0.30):
        return False
    if split_y is not None and split_y < int(h * 0.40):
        return False
    return True


def _apply_vertical_front_crop(
    bgr: np.ndarray,
    *,
    split_y: int | None,
    stacked_fallback: bool,
    from_hw: dict[str, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    h = from_hw["height"]
    w = from_hw["width"]
    crop_h = _vertical_crop_height(h, split_y, stacked_fallback=stacked_fallback)
    if not _vertical_crop_valid(h, crop_h, split_y):
        return bgr, {}
    cropped = bgr[:crop_h, :].copy()
    layout = "portrait_stacked" if split_y is not None else "portrait_stacked_fallback"
    return cropped, {
        "npwp_dual_crop_applied": True,
        "npwp_dual_crop_axis": "vertical",
        "npwp_dual_crop_layout": layout,
        "npwp_dual_split_y": split_y,
        "npwp_dual_crop_height": crop_h,
        "npwp_dual_crop_stacked_fallback": stacked_fallback,
        "npwp_dual_crop_from_hw": from_hw,
    }


def maybe_npwp_dual_front_crop_bgr(
    bgr: np.ndarray,
    *,
    document_profile_id: str = "",
    source_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop panel depan (kiri atau atas) bila terdeteksi layout dual NPWP."""
    profile = (document_profile_id or "").strip().casefold()
    meta: dict[str, Any] = {
        "npwp_dual_crop_applied": False,
        "npwp_dual_crop_profile": profile or None,
    }
    if profile not in _NPWP_PROFILES or not npwp_dual_crop_enabled():
        meta["npwp_dual_crop_skipped_reason"] = (
            "disabled" if profile in _NPWP_PROFILES else "not_npwp_profile"
        )
        return bgr, meta

    h, w = bgr.shape[:2]
    from_hw = {"width": w, "height": h}
    src_w, src_h = source_hw if source_hw else (w, h)
    src_landscape_single = src_w / max(src_h, 1) >= 1.08 and src_w / max(src_h, 1) < 1.85
    portrait_tall = _npwp_portrait_tall(h, w) and not src_landscape_single
    landscape_dual = _npwp_landscape_dual_aspect(w, h)

    # Portrait distack: panel depan di atas — jangan pakai crop kiri yang sering kena belakang.
    if portrait_tall and not landscape_dual:
        split_y = detect_npwp_dual_split_y(bgr)
        stacked_fallback = split_y is None and h / max(w, 1) >= 1.55
        if split_y is not None or stacked_fallback:
            cropped, patch = _apply_vertical_front_crop(
                bgr,
                split_y=split_y,
                stacked_fallback=stacked_fallback,
                from_hw=from_hw,
            )
            if patch:
                meta.update(patch)
                return cropped, meta

    split_x = detect_npwp_dual_split_x(bgr)
    if split_x is not None and not landscape_dual and _npwp_portrait_frame(h, w):
        # Side-by-side dalam frame portrait: split harus di tengah, bukan strip sempit kiri.
        if split_x < int(w * 0.36) or split_x > int(w * 0.58):
            split_x = None

    if split_x is not None:
        pad = max(4, w // 200)
        if landscape_dual:
            crop_w = max(int(w * 0.40), split_x - pad)
            crop_w = min(crop_w, int(w * 0.55))
            min_frac = 0.38
        else:
            crop_w = max(int(w * 0.42), split_x - pad)
            crop_w = min(crop_w, int(w * 0.55))
            min_frac = 0.40
        if crop_w >= int(w * min_frac) and _horizontal_crop_valid(w, h, crop_w):
            cropped = bgr[:, :crop_w].copy()
            meta.update(
                {
                    "npwp_dual_crop_applied": True,
                    "npwp_dual_crop_axis": "horizontal",
                    "npwp_dual_crop_layout": "landscape_dual" if landscape_dual else "portrait_side_by_side",
                    "npwp_dual_split_x": split_x,
                    "npwp_dual_crop_width": crop_w,
                    "npwp_dual_crop_from_hw": from_hw,
                }
            )
            return cropped, meta

    if landscape_dual and w >= 400:
        crop_w = int(w * 0.48)
        if crop_w >= int(w * 0.38):
            cropped = bgr[:, :crop_w].copy()
            meta.update(
                {
                    "npwp_dual_crop_applied": True,
                    "npwp_dual_crop_axis": "horizontal",
                    "npwp_dual_crop_layout": "landscape_dual_fallback",
                    "npwp_dual_crop_width": crop_w,
                    "npwp_dual_crop_from_hw": {"width": w, "height": h},
                }
            )
            return cropped, meta

    aspect = h / max(w, 1)
    if src_landscape_single:
        meta["npwp_dual_crop_skipped_reason"] = "landscape_single_card"
        return bgr, meta
    split_y = detect_npwp_dual_split_y(bgr) if aspect >= 1.25 else None
    stacked_fallback = aspect >= 1.75 and split_y is None
    if split_y is not None or stacked_fallback:
        cropped, patch = _apply_vertical_front_crop(
            bgr,
            split_y=split_y,
            stacked_fallback=stacked_fallback,
            from_hw=from_hw,
        )
        if patch:
            meta.update(patch)
            return cropped, meta

    meta["npwp_dual_crop_skipped_reason"] = "no_dual_layout"
    return bgr, meta
