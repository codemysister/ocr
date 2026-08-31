"""Validasi foto profil: wajah terdeteksi + latar dominan biru."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from systems.validation.document_profiles import profile_label
from systems.validation.face_detect import FaceDetectionUnavailable, detect_frontal_faces, face_detection_backend
from systems.validation.fuzzy_compare import build_document_verdict

# HSV biru pas foto — rentang lebar (cyan–navy, foto HP / flash).
_BLUE_HSV_LOWER = np.array([82, 22, 28], dtype=np.uint8)
_BLUE_HSV_UPPER = np.array([140, 255, 255], dtype=np.uint8)
# Merah dominan (bukan kulit tipis di sampel biru).
_RED_HSV_LOWER_A = np.array([0, 70, 50], dtype=np.uint8)
_RED_HSV_UPPER_A = np.array([12, 255, 255], dtype=np.uint8)
_RED_HSV_LOWER_B = np.array([168, 70, 50], dtype=np.uint8)
_RED_HSV_UPPER_B = np.array([180, 255, 255], dtype=np.uint8)


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _min_blue_ratio() -> float:
    return max(0.0, min(1.0, _env_float("FOTO_PROFILE_MIN_BLUE_RATIO", 0.32)))


def _max_red_ratio() -> float:
    return max(0.0, min(1.0, _env_float("FOTO_PROFILE_MAX_RED_RATIO", 0.22)))


def _min_blue_card_ratio() -> float:
    return max(0.0, min(1.0, _env_float("FOTO_PROFILE_MIN_BLUE_CARD_RATIO", 0.38)))


def _min_blue_portrait_crop_ratio() -> float:
    return max(0.05, min(0.5, _env_float("FOTO_PROFILE_MIN_BLUE_CROP_RATIO", 0.10)))


def _face_min_area_ratio() -> float:
    return max(0.005, min(0.5, _env_float("FOTO_PROFILE_FACE_MIN_AREA_RATIO", 0.012)))


def _face_max_area_ratio() -> float:
    return max(0.2, min(0.95, _env_float("FOTO_PROFILE_FACE_MAX_AREA_RATIO", 0.75)))


def _rotate_90ccw(bgr: np.ndarray, steps: int) -> np.ndarray:
    out = bgr
    for _ in range(steps % 4):
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


def _filter_dominant_faces(
    faces: list[tuple[int, int, int, int]],
    *,
    min_area_ratio: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Buang deteksi wajah kecil (false positive) jauh di bawah wajah utama."""
    if len(faces) <= 1:
        return list(faces)
    areas = [float(fw * fh) for _x, _y, fw, fh in faces]
    max_a = max(areas)
    if max_a <= 0:
        return list(faces)
    return [f for f, a in zip(faces, areas) if a >= max_a * min_area_ratio]


def _blue_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, _BLUE_HSV_LOWER, _BLUE_HSV_UPPER)


def _blue_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    """Biru dominan di BGR — menangkap cyan/navy yang HSV kadang lewatkan."""
    b, g, r = cv2.split(bgr)
    b_i = b.astype(np.int16)
    g_i = g.astype(np.int16)
    r_i = r.astype(np.int16)
    dom = (b_i >= r_i + 6) & (b_i >= g_i + 4) & (b_i >= 55)
    return dom.astype(np.uint8) * 255


def _combined_blue_mask(hsv: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(_blue_mask(hsv), _blue_mask_bgr(bgr))


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    a = cv2.inRange(hsv, _RED_HSV_LOWER_A, _RED_HSV_UPPER_A)
    b = cv2.inRange(hsv, _RED_HSV_LOWER_B, _RED_HSV_UPPER_B)
    return cv2.bitwise_or(a, b)


def _color_ratio(bgr: np.ndarray, sample_mask: np.ndarray, color_mask: np.ndarray) -> float:
    total = int(np.count_nonzero(sample_mask))
    if total == 0:
        return 0.0
    hits = int(np.count_nonzero(cv2.bitwise_and(color_mask, sample_mask)))
    return hits / float(total)


def _background_sample_excluding_faces(
    height: int,
    width: int,
    faces: list[tuple[int, int, int, int]],
    *,
    roi_mask: np.ndarray | None = None,
    border_frac: float = 0.14,
    face_expand: float = 0.18,
) -> np.ndarray:
    if roi_mask is not None and np.count_nonzero(roi_mask) >= 64:
        sample = roi_mask.copy()
    else:
        sample = _background_sample_mask(
            height, width, faces, border_frac=border_frac, face_expand=face_expand
        )
    if faces:
        for x, y, fw, fh in faces:
            pad_x = int(fw * face_expand)
            pad_y = int(fh * face_expand)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + fw + pad_x)
            y2 = min(height, y + fh + pad_y)
            sample[y1:y2, x1:x2] = 0
    return sample


def _extract_blue_portrait_crop(
    bgr: np.ndarray,
    *,
    min_contour_area_ratio: float | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Isolasi kartu pas foto biru dari foto meja/scan (region biru terbesar).

    Hanya dipanggil dari validasi foto_profile — jangan dipakai di preprocessing OCR umum.
    """
    min_contour_area_ratio = (
        _min_blue_portrait_crop_ratio() if min_contour_area_ratio is None else min_contour_area_ratio
    )
    height, width = bgr.shape[:2]
    img_area = float(max(1, height * width))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = _combined_blue_mask(hsv, bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meta: dict[str, Any] = {
        "portrait_crop_applied": False,
        "portrait_crop_bbox": None,
        "blue_region_area_ratio": 0.0,
    }
    if not contours:
        return None, meta

    largest = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(largest)) / img_area
    meta["blue_region_area_ratio"] = round(area_ratio, 6)
    if area_ratio < min_contour_area_ratio:
        return None, meta

    x, y, fw, fh = cv2.boundingRect(largest)
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[y : y + fh, x : x + fw] = 255
    blue_density = _color_ratio(bgr, roi, blue)
    meta["blue_region_density"] = round(blue_density, 6)
    if blue_density < 0.30:
        return None, meta
    pad = max(4, int(0.02 * max(fw, fh)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + fw + pad)
    y2 = min(height, y + fh + pad)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, meta

    meta["portrait_crop_applied"] = True
    meta["portrait_crop_bbox"] = {
        "x": x1,
        "y": y1,
        "width": int(x2 - x1),
        "height": int(y2 - y1),
    }
    return crop, meta


def _card_background_color_ratios(
    bgr: np.ndarray,
    faces: list[tuple[int, int, int, int]],
) -> tuple[float, float, str]:
    """
    Persentase biru & merah pada area latar (di luar wajah).
    Biru diukur dari pixel background aktual — bukan dari mask biru itu sendiri.
    """
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = _combined_blue_mask(hsv, bgr)
    red = _red_mask(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    card_mask = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    card_fill = float(np.count_nonzero(card_mask)) / float(max(1, card_mask.size))

    if card_fill >= 0.28 and faces:
        sample = _background_sample_excluding_faces(
            height, width, faces, roi_mask=None, border_frac=0.08, face_expand=0.14
        )
        blue_ratio = _color_ratio(bgr, sample, blue)
        red_ratio = _color_ratio(bgr, sample, red)
        border_sample = _background_sample_excluding_faces(
            height, width, faces, roi_mask=None, border_frac=0.12, face_expand=0.22
        )
        blue_border = _color_ratio(bgr, border_sample, blue)
        red_border = _color_ratio(bgr, border_sample, red)
        if blue_border >= 0.45:
            blue_ratio = max(blue_ratio, blue_border)
            red_ratio = min(red_ratio, red_border)
        return blue_ratio, red_ratio, "full_frame_excluding_face"

    roi_mask: np.ndarray | None = None
    method = "border_band"
    if card_fill >= 0.10 and faces:
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, fw, fh = cv2.boundingRect(largest)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            roi_mask[y : y + fh, x : x + fw] = 255
            method = "card_roi_excluding_face"

    sample = _background_sample_excluding_faces(height, width, faces, roi_mask=roi_mask)
    blue_ratio = _color_ratio(bgr, sample, blue)
    red_ratio = _color_ratio(bgr, sample, red)

    border_sample = _background_sample_excluding_faces(
        height,
        width,
        faces,
        roi_mask=None,
        border_frac=0.12,
        face_expand=0.22,
    )
    blue_border = _color_ratio(bgr, border_sample, blue)
    red_border = _color_ratio(bgr, border_sample, red)
    if blue_border >= 0.45:
        blue_ratio = max(blue_ratio, blue_border)
        red_ratio = min(red_ratio, red_border)

    return blue_ratio, red_ratio, method


def _background_sample_mask(
    height: int,
    width: int,
    faces: list[tuple[int, int, int, int]],
    *,
    border_frac: float = 0.10,
    face_expand: float = 0.20,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    bt = max(1, int(height * border_frac))
    bl = max(1, int(width * border_frac))
    mask[:bt, :] = 255
    mask[-bt:, :] = 255
    mask[:, :bl] = 255
    mask[:, -bl:] = 255

    if faces:
        inner = np.full((height, width), 255, dtype=np.uint8)
        for x, y, fw, fh in faces:
            pad_x = int(fw * face_expand)
            pad_y = int(fh * face_expand)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + fw + pad_x)
            y2 = min(height, y + fh + pad_y)
            inner[y1:y2, x1:x2] = 0
        mask = cv2.bitwise_or(mask, inner)

    return mask


def _blue_ratio(bgr: np.ndarray, sample_mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _color_ratio(bgr, sample_mask, _combined_blue_mask(hsv, bgr))


def _background_passes(
    *,
    blue_ratio: float,
    red_ratio: float,
    method: str,
    min_blue: float,
    max_red: float,
) -> bool:
    if method == "border_band":
        blue_min = min_blue
    elif method == "full_frame_excluding_face":
        blue_min = max(min_blue - 0.04, 0.28)
    else:
        blue_min = max(min_blue, _min_blue_card_ratio())

    if blue_ratio < blue_min:
        return False

    # Biru kuat → lolos meski ada sedikit noise merah/kulit di sampel.
    if blue_ratio >= 0.52 and red_ratio <= max(0.38, max_red + 0.12):
        return True

    # Gagal merah hanya bila dominan nyata atas biru (bukan sekadar kulit di pinggir wajah).
    if red_ratio > max(blue_ratio * 0.62, max_red + 0.08):
        return False
    if red_ratio > max_red and blue_ratio < 0.40:
        return False
    return True


def _face_metrics(
    faces: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> dict[str, Any]:
    if not faces:
        return {
            "face_count": 0,
            "face_area_ratio": 0.0,
            "face_center_x_ratio": None,
            "face_center_y_ratio": None,
            "face_centered": False,
        }

    img_area = float(max(1, width * height))
    areas = [float(fw * fh) / img_area for _x, _y, fw, fh in faces]
    idx = int(max(range(len(faces)), key=lambda i: areas[i]))
    x, y, fw, fh = faces[idx]
    cx = (x + fw / 2.0) / float(max(1, width))
    cy = (y + fh / 2.0) / float(max(1, height))
    centered = 0.18 <= cx <= 0.82 and 0.15 <= cy <= 0.85
    return {
        "face_count": len(faces),
        "face_area_ratio": round(areas[idx], 6),
        "face_center_x_ratio": round(cx, 6),
        "face_center_y_ratio": round(cy, 6),
        "face_centered": centered,
        "primary_face_bbox": {"x": x, "y": y, "width": fw, "height": fh},
    }


@dataclass
class _FrameAnalysis:
    bgr: np.ndarray
    orientation_90ccw_steps: int
    crop_meta: dict[str, Any]
    face_meta: dict[str, Any]
    blue_ratio: float
    red_ratio: float
    blue_ratio_method: str
    face_pass: bool
    blue_pass: bool
    score: float


def _orientation_rank_key(c: _FrameAnalysis) -> tuple[float, float, int]:
    """
    Skor akhir pemilihan orientasi: hindari terbalik (180°) dan utamakan wajah di bagian atas.
    """
    adjusted = c.score
    cy = c.face_meta.get("face_center_y_ratio")
    if cy is not None:
        if 0.18 <= float(cy) <= 0.52:
            adjusted += 1.5
        elif float(cy) > 0.56:
            adjusted -= 4.0

    k = c.orientation_90ccw_steps
    if k == 0:
        adjusted += 0.6
    elif k == 2:
        adjusted -= 6.0
    elif k == 3:
        adjusted -= 0.2
    elif k == 1:
        adjusted -= 0.1

    # Tie-break: skor lebih tinggi, lalu orientasi asli (0) lebih diutamakan.
    k_order = {0: 0, 1: 1, 3: 2, 2: 9}
    return (adjusted, -float(c.face_meta.get("face_area_ratio") or 0.0), k_order.get(k, 9))


def _analyze_frame(
    bgr: np.ndarray,
    *,
    orientation_90ccw_steps: int,
    crop_meta: dict[str, Any],
    min_blue: float,
    max_red: float,
    min_face_area: float,
    max_face_area: float,
) -> _FrameAnalysis:
    height, width = bgr.shape[:2]
    faces = _filter_dominant_faces(detect_frontal_faces(bgr))
    face_meta = _face_metrics(faces, width, height)
    face_count = int(face_meta["face_count"])
    face_area_ratio = float(face_meta["face_area_ratio"])

    face_count_pass = face_count == 1
    face_size_pass = face_count_pass and min_face_area <= face_area_ratio <= max_face_area
    face_pass = face_count_pass and face_size_pass

    blue_ratio, red_ratio, blue_method = _card_background_color_ratios(bgr, faces)
    blue_pass = _background_passes(
        blue_ratio=blue_ratio,
        red_ratio=red_ratio,
        method=blue_method,
        min_blue=min_blue,
        max_red=max_red,
    )

    score = 0.0
    if face_pass:
        score += 3.0 + face_area_ratio
    elif face_count == 1:
        score += 1.0
    elif face_count > 1:
        score += 0.2
    score += min(2.0, blue_ratio / max(min_blue, 0.01))
    if crop_meta.get("portrait_crop_applied"):
        score += 0.5
    if face_pass and blue_pass:
        score += 5.0

    return _FrameAnalysis(
        bgr=bgr,
        orientation_90ccw_steps=orientation_90ccw_steps,
        crop_meta=crop_meta,
        face_meta=face_meta,
        blue_ratio=blue_ratio,
        red_ratio=red_ratio,
        blue_ratio_method=blue_method,
        face_pass=face_pass,
        blue_pass=blue_pass,
        score=score,
    )


def _best_frame_analysis(
    bgr: np.ndarray,
    *,
    min_blue: float,
    max_red: float,
    min_face_area: float,
    max_face_area: float,
) -> _FrameAnalysis:
    """Coba 4 orientasi + crop region biru; pilih kandidat terbaik."""
    candidates: list[_FrameAnalysis] = []

    for k in range(4):
        rotated = _rotate_90ccw(bgr, k)
        crop, crop_meta = _extract_blue_portrait_crop(rotated)
        if crop is not None:
            candidates.append(
                _analyze_frame(
                    crop,
                    orientation_90ccw_steps=k,
                    crop_meta=crop_meta,
                    min_blue=min_blue,
                    max_red=max_red,
                    min_face_area=min_face_area,
                    max_face_area=max_face_area,
                )
            )
        else:
            empty_crop_meta = {
                "portrait_crop_applied": False,
                "portrait_crop_bbox": None,
                "blue_region_area_ratio": 0.0,
            }
            candidates.append(
                _analyze_frame(
                    rotated,
                    orientation_90ccw_steps=k,
                    crop_meta=empty_crop_meta,
                    min_blue=min_blue,
                    max_red=max_red,
                    min_face_area=min_face_area,
                    max_face_area=max_face_area,
                )
            )

    if not candidates:
        raise ValueError("Tidak dapat menganalisis gambar.")

    # Utamakan orientasi asli (EXIF) bila sudah ada tepat satu wajah dominan.
    upright_native = [
        c
        for c in candidates
        if c.orientation_90ccw_steps == 0 and int(c.face_meta.get("face_count") or 0) == 1
    ]
    if upright_native:
        return max(upright_native, key=_orientation_rank_key)

    # Koreksi menyamping (±90°) sebelum pertimbangkan terbalik (180°).
    sideways = [c for c in candidates if c.orientation_90ccw_steps in (1, 3)]
    if sideways:
        return max(sideways, key=_orientation_rank_key)

    return max(candidates, key=_orientation_rank_key)


def _foto_profile_unavailable_result(
    *,
    document_type: str,
    expected_name: str,
    error: str,
) -> dict[str, Any]:
    profile = "foto_profile"
    blockers = ["FACE_DETECTOR"]
    summary = "Foto profil tidak dapat divalidasi: deteksi wajah tidak tersedia di server."
    explanation = {
        "summary": summary,
        "detail_lines": [error],
        "primary_blockers": blockers,
        "hints": [
            "Pastikan OpenCV terpasang dengan benar (opencv-python-headless).",
            f"Backend deteksi: {face_detection_backend()}.",
        ],
        "gates": {
            "face": {"pass": False},
            "blue_background": {"pass": False},
            "document_type": {"pass": False},
        },
        "document_matched": False,
        "validation_mode": "image",
    }
    detection: dict[str, Any] = {
        "detected_profile_id": None,
        "detected_document_type": None,
        "confidence_score": 0.0,
        "confidence_ratio": 0.0,
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_image_only_profile",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=expected_name,
        identity_pass=None,
        identity=None,
        identity_min_score=65.0,
        document_type_pass=False,
        document_matched=False,
        name_extraction=name_extraction,
    )
    return {
        "valid": False,
        "document_type": (document_type or "").strip(),
        "document_profile_id": profile,
        "validation_mode": "image",
        "document_type_pass": False,
        "document_matched": False,
        "identity_pass": None,
        "identity": None,
        "name_extraction": name_extraction,
        "image_validation": {
            "face_pass": False,
            "face_count": 0,
            "face_count_pass": False,
            "face_size_pass": False,
            "face_centered_pass": False,
            "blue_background_pass": False,
            "face_detector_backend": face_detection_backend(),
            "face_detector_error": error,
        },
        "keywords": [],
        "excluded_keywords": [],
        "exclusion_violated": False,
        "explanation": explanation,
        "document_type_detection": detection,
        "verdict": verdict,
        "is_own_document": verdict["is_own_document"],
        "document_type_current": verdict["document_type_current"],
        "document_type_current_label": verdict["document_type_current_label"],
    }


def validate_foto_profile(
    bgr: np.ndarray,
    *,
    document_type: str = "foto_profile",
    expected_name: str = "",
) -> dict[str, Any]:
    """
    Validasi foto profil berlatar biru.

    Gate:
    - Tepat satu wajah frontal terdeteksi (setelah koreksi orientasi + crop biru)
    - Ukuran wajah wajar di frame analisis
    - Latar di luar wajah dominan biru (HSV)
    """
    profile = "foto_profile"
    min_blue = _min_blue_ratio()
    max_red = _max_red_ratio()
    min_face_area = _face_min_area_ratio()
    max_face_area = _face_max_area_ratio()

    if bgr is None or bgr.size == 0:
        raise ValueError("Gambar kosong atau tidak valid.")

    try:
        analysis = _best_frame_analysis(
            bgr,
            min_blue=min_blue,
            max_red=max_red,
            min_face_area=min_face_area,
            max_face_area=max_face_area,
        )
    except FaceDetectionUnavailable as e:
        return _foto_profile_unavailable_result(
            document_type=document_type,
            expected_name=expected_name,
            error=str(e),
        )

    face_meta = analysis.face_meta
    face_count = int(face_meta["face_count"])
    face_area_ratio = float(face_meta["face_area_ratio"])
    blue_ratio = analysis.blue_ratio
    red_ratio = analysis.red_ratio

    face_count_pass = face_count == 1
    face_size_pass = face_count_pass and min_face_area <= face_area_ratio <= max_face_area
    face_centered_pass = bool(face_meta.get("face_centered")) if face_count_pass else False
    face_pass = analysis.face_pass
    blue_background_pass = analysis.blue_pass

    document_type_pass = bool(face_pass and blue_background_pass)
    document_matched = document_type_pass

    detail_lines: list[str] = []
    blockers: list[str] = []
    hints: list[str] = []

    orient_note = ""
    if analysis.orientation_90ccw_steps:
        orient_note = f" Orientasi dikoreksi {analysis.orientation_90ccw_steps * 90}°."
    if analysis.crop_meta.get("portrait_crop_applied"):
        orient_note += " Kotak pas foto diambil dari gambar."

    if face_count_pass and face_size_pass:
        detail_lines.append(
            f"Wajah terdeteksi: 1 wajah, area ~{face_area_ratio * 100:.1f}% dari area analisis."
            + orient_note
        )
    elif face_count == 0:
        blockers.append("FACE")
        detail_lines.append("Tidak ada wajah terdeteksi pada foto." + orient_note)
        hints.append("Pastikan wajah frontal terlihat jelas, tidak tertutup, dan cukup besar.")
        hints.append("Foto menyamping atau pas foto kecil di atas meja putih sering gagal — coba crop lebih dekat.")
    elif face_count > 1:
        blockers.append("FACE")
        detail_lines.append(f"Terdeteksi {face_count} wajah — foto profil harus satu orang." + orient_note)
        hints.append("Crop foto sehingga hanya satu wajah yang terlihat.")
    else:
        blockers.append("FACE")
        detail_lines.append(
            f"Ukuran wajah tidak wajar (~{face_area_ratio * 100:.1f}% area analisis; "
            f"syarat {min_face_area * 100:.0f}–{max_face_area * 100:.0f}%)."
            + orient_note
        )
        hints.append("Gunakan foto close-up standar paspor (wajah cukup besar, tidak terlalu jauh).")

    if blue_background_pass:
        method_label = (
            "area kartu (luar wajah)"
            if analysis.blue_ratio_method == "card_roi_excluding_face"
            else "pinggir frame"
        )
        detail_lines.append(
            f"Latar biru lolos: ~{blue_ratio * 100:.1f}% biru pada {method_label} "
            f"(syarat ≥ {min_blue * 100:.0f}%, merah ≤ {max_red * 100:.0f}%)."
        )
    else:
        blockers.append("BLUE_BACKGROUND")
        if red_ratio > max_red or red_ratio >= blue_ratio * 0.35:
            detail_lines.append(
                f"Latar tidak lolos: dominan merah/non-biru (~{red_ratio * 100:.1f}% merah, "
                f"~{blue_ratio * 100:.1f}% biru)."
            )
        else:
            detail_lines.append(
                f"Latar biru tidak lolos: ~{blue_ratio * 100:.1f}% area sampel biru "
                f"(syarat ≥ {min_blue * 100:.0f}%)."
            )
        hints.append(
            "Gunakan background biru solid seperti pas foto resmi; hindari putih, merah, atau pola."
        )

    if face_count_pass and not face_centered_pass:
        hints.append("Posisikan wajah lebih ke tengah frame (opsional, tidak memblokir).")

    if document_matched:
        summary = "Foto profil valid: wajah terdeteksi dengan latar biru yang memadai."
    else:
        summary = "Foto profil tidak valid: " + "; ".join(
            b.replace("_", " ").lower() for b in blockers
        ) + "."

    explanation = {
        "summary": summary,
        "detail_lines": detail_lines,
        "primary_blockers": blockers,
        "hints": hints,
        "gates": {
            "face": {
                "pass": face_pass,
                "face_count_pass": face_count_pass,
                "face_size_pass": face_size_pass,
                "face_centered_pass": face_centered_pass,
            },
            "blue_background": {
                "pass": blue_background_pass,
                "blue_ratio": round(blue_ratio, 6),
                "red_ratio": round(red_ratio, 6),
                "min_blue_ratio": min_blue,
                "max_red_ratio": max_red,
            },
            "document_type": {"pass": document_type_pass},
        },
        "document_matched": document_matched,
        "validation_mode": "image",
    }

    detection: dict[str, Any] = {
        "detected_profile_id": profile if document_type_pass else None,
        "detected_document_type": profile_label(profile) if document_type_pass else None,
        "confidence_score": round(blue_ratio * 100.0, 4) if document_type_pass else 0.0,
        "confidence_ratio": round(blue_ratio, 6) if document_type_pass else 0.0,
        "min_aggregate_ratio": min_blue,
        "candidates": [],
    }

    name_extraction = {
        "candidate_raw": None,
        "candidate_normalized": "",
        "method": "skipped_image_only_profile",
    }

    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=expected_name,
        identity_pass=None,
        identity=None,
        identity_min_score=65.0,
        document_type_pass=document_type_pass,
        document_matched=document_matched,
        name_extraction=name_extraction,
    )

    ah, aw = analysis.bgr.shape[:2]
    image_validation = {
        **face_meta,
        "face_pass": face_pass,
        "face_count_pass": face_count_pass,
        "face_size_pass": face_size_pass,
        "face_centered_pass": face_centered_pass,
        "blue_background_ratio": round(blue_ratio, 6),
        "red_background_ratio": round(red_ratio, 6),
        "blue_background_ratio_method": analysis.blue_ratio_method,
        "blue_background_pass": blue_background_pass,
        "min_blue_ratio": min_blue,
        "max_red_ratio": max_red,
        "face_min_area_ratio": min_face_area,
        "face_max_area_ratio": max_face_area,
        "image_width": aw,
        "image_height": ah,
        "orientation_correction_90ccw_steps": analysis.orientation_90ccw_steps,
        **analysis.crop_meta,
    }

    return {
        "valid": document_matched,
        "_analysis_bgr": analysis.bgr,
        "document_type": (document_type or "").strip(),
        "document_profile_id": profile,
        "validation_mode": "image",
        "document_type_pass": document_type_pass,
        "document_matched": document_matched,
        "identity_pass": None,
        "identity": None,
        "name_extraction": name_extraction,
        "image_validation": image_validation,
        "keywords": [],
        "excluded_keywords": [],
        "exclusion_violated": False,
        "explanation": explanation,
        "document_type_detection": detection,
        "verdict": verdict,
        "is_own_document": verdict["is_own_document"],
        "document_type_current": verdict["document_type_current"],
        "document_type_current_label": verdict["document_type_current_label"],
    }
