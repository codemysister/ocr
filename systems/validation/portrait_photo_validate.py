"""Validasi foto profil: wajah terdeteksi + latar dominan biru."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

import cv2
import numpy as np

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict

_HAAR: Final[str] = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar_cascade: cv2.CascadeClassifier | None = None

# HSV biru pas foto (sedikit longgar untuk foto HP / pencahayaan ruangan).
_BLUE_HSV_LOWER = np.array([80, 25, 25], dtype=np.uint8)
_BLUE_HSV_UPPER = np.array([140, 255, 255], dtype=np.uint8)


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _min_blue_ratio() -> float:
    return max(0.0, min(1.0, _env_float("FOTO_PROFILE_MIN_BLUE_RATIO", 0.40)))


def _face_min_area_ratio() -> float:
    return max(0.01, min(0.5, _env_float("FOTO_PROFILE_FACE_MIN_AREA_RATIO", 0.03)))


def _face_max_area_ratio() -> float:
    return max(0.2, min(0.95, _env_float("FOTO_PROFILE_FACE_MAX_AREA_RATIO", 0.75)))


def _haar() -> cv2.CascadeClassifier:
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(_HAAR)
    return _haar_cascade


def _rotate_90ccw(bgr: np.ndarray, steps: int) -> np.ndarray:
    out = bgr
    for _ in range(steps % 4):
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return []

    scale = 1.0
    work = gray
    if min(h, w) < 360:
        scale = 360.0 / float(min(h, w))
        work = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    wh, ww = work.shape[:2]
    min_px = max(16, int(min(wh, ww) * 0.03))
    casc = _haar()
    if casc.empty():
        return []

    found = casc.detectMultiScale(
        work,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_px, min_px),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(found) == 0:
        return []

    inv = 1.0 / scale
    faces: list[tuple[int, int, int, int]] = []
    for fx, fy, fw, fh in found:
        x = max(0, int(fx * inv))
        y = max(0, int(fy * inv))
        fw_i = max(1, int(fw * inv))
        fh_i = max(1, int(fh * inv))
        faces.append((x, y, fw_i, fh_i))
    return faces


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


def _extract_blue_portrait_crop(
    bgr: np.ndarray,
    *,
    min_contour_area_ratio: float = 0.02,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Isolasi kartu pas foto biru dari foto meja/scan (region biru terbesar).

    Hanya dipanggil dari validasi foto_profile — jangan dipakai di preprocessing OCR umum.
  """
    height, width = bgr.shape[:2]
    img_area = float(max(1, height * width))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = _blue_mask(hsv)
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


def _card_background_blue_ratio(
    bgr: np.ndarray,
    faces: list[tuple[int, int, int, int]],
) -> tuple[float, str]:
    """
    Persentase biru pada area kartu pas foto (mask HSV), di luar wajah.
    Metode ini lebih akurat daripada pinggir frame penuh saat kartu sudah di-crop.
    """
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = _blue_mask(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    card_mask = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    card_fill = float(np.count_nonzero(card_mask)) / float(max(1, card_mask.size))

    if card_fill >= 0.12 and faces:
        sample = card_mask.copy()
        x, y, fw, fh = max(
            faces,
            key=lambda f: f[2] * f[3],
        )
        pad_x = int(fw * 0.22)
        pad_y = int(fh * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + fw + pad_x)
        y2 = min(height, y + fh + pad_y)
        sample[y1:y2, x1:x2] = 0
        total = int(np.count_nonzero(sample))
        if total >= 64:
            hits = int(np.count_nonzero(cv2.bitwise_and(blue, sample)))
            return hits / float(total), "card_mask_excluding_face"

    sample = _background_sample_mask(
        height, width, faces, border_frac=0.14, face_expand=0.18
    )
    return _blue_ratio(bgr, sample), "border_band"


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
    total = int(np.count_nonzero(sample_mask))
    if total == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = _blue_mask(hsv)
    blue_masked = cv2.bitwise_and(blue, blue, mask=sample_mask)
    return float(np.count_nonzero(blue_masked)) / float(total)


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
    min_face_area: float,
    max_face_area: float,
) -> _FrameAnalysis:
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _filter_dominant_faces(_detect_faces(gray))
    face_meta = _face_metrics(faces, width, height)
    face_count = int(face_meta["face_count"])
    face_area_ratio = float(face_meta["face_area_ratio"])

    face_count_pass = face_count == 1
    face_size_pass = face_count_pass and min_face_area <= face_area_ratio <= max_face_area
    face_pass = face_count_pass and face_size_pass

    blue_ratio, blue_method = _card_background_blue_ratio(bgr, faces)
    blue_min = min_blue if blue_method == "border_band" else max(min_blue, 0.65)
    blue_pass = blue_ratio >= blue_min

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
        blue_ratio_method=blue_method,
        face_pass=face_pass,
        blue_pass=blue_pass,
        score=score,
    )


def _best_frame_analysis(
    bgr: np.ndarray,
    *,
    min_blue: float,
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
    min_face_area = _face_min_area_ratio()
    max_face_area = _face_max_area_ratio()

    if bgr is None or bgr.size == 0:
        raise ValueError("Gambar kosong atau tidak valid.")

    analysis = _best_frame_analysis(
        bgr,
        min_blue=min_blue,
        min_face_area=min_face_area,
        max_face_area=max_face_area,
    )

    face_meta = analysis.face_meta
    face_count = int(face_meta["face_count"])
    face_area_ratio = float(face_meta["face_area_ratio"])
    blue_ratio = analysis.blue_ratio

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
            if analysis.blue_ratio_method == "card_mask_excluding_face"
            else "pinggir frame"
        )
        detail_lines.append(
            f"Latar biru lolos: ~{blue_ratio * 100:.1f}% pada {method_label} "
            f"(syarat ≥ {min_blue * 100:.0f}%)."
        )
    else:
        blockers.append("BLUE_BACKGROUND")
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
                "min_blue_ratio": min_blue,
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
        "blue_background_ratio_method": analysis.blue_ratio_method,
        "blue_background_pass": blue_background_pass,
        "min_blue_ratio": min_blue,
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
