"""Gate Surat Kesanggupan BPJS: wajib materai fisik (stempel merah/keemasan)."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

_STAMP_PHRASES = (
    "meterai tempel",
    "materai tempel",
    "bea meterai",
    "bea materai",
    "e-meterai",
    "e-materai",
)

_STAMP_PHRASE_RE = re.compile(
    r"\b(?:e[\s\-]?m[ae]t[e]?r[ae][il1]|m[ae]t[e]?r[ae][il1]\s+tempel|bea\s+m[ae]t[e]?r[ae][il1])\b",
    re.I,
)


def _normalize_ocr(ocr_text: str) -> str:
    return re.sub(r"\s+", " ", (ocr_text or "").casefold()).strip()


def ocr_has_meterai(ocr_text: str, *, min_partial: float = 88.0) -> bool:
    """Teks pada stempel (METERAI TEMPEL), bukan label kotak 'Materai Rp10000'."""
    raw = ocr_text or ""
    if not raw.strip():
        return False
    if _STAMP_PHRASE_RE.search(raw):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", raw.casefold())
    if any(p in compact for p in ("meteraitempel", "materaitempel", "beameterai", "beamaterai", "emeterai", "ematerai")):
        return True
    ocr_n = _normalize_ocr(raw)
    for phrase in ("meterai tempel", "materai tempel", "bea meterai", "bea materai"):
        if fuzz.partial_ratio(phrase, ocr_n) >= min_partial:
            return True
    return False


def _bgr_from_image(image: Any):
    import numpy as np

    if image is None:
        return None
    if hasattr(image, "shape") and getattr(image, "ndim", 0) == 3:
        return image
    if isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
        if not raw:
            return None
        try:
            import cv2
        except ImportError:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


def image_has_red_meterai(image: Any) -> bool:
    """True bila ada blob berwarna (merah/pink/keemasan) berbentuk materai di area tanda tangan."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    bgr = _bgr_from_image(image)
    if bgr is None or bgr.size == 0:
        return False

    h, w = bgr.shape[:2]
    if h < 80 or w < 80:
        return False

    y0, y1 = int(h * 0.48), int(h * 0.93)
    x0, x1 = int(w * 0.32), int(w * 0.98)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    a = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    chroma = np.sqrt((a - 128.0) ** 2 + (b_ch - 128.0) ** 2)
    base = float(np.median(chroma))
    thr = max(11.0, base + 6.5)
    mask = ((chroma > thr) | ((a >= 136) & (v_ch >= 40) & (v_ch <= 250))).astype(np.uint8) * 255
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_page = float(h * w)
    min_side = float(min(h, w))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        frac = area / area_page
        if not (0.0009 <= frac <= 0.035):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        ar = bw / max(bh, 1)
        if not (0.55 <= ar <= 1.8):
            continue
        if min(bw, bh) < 0.028 * min_side:
            continue
        fill = area / max(float(bw * bh), 1.0)
        if fill < 0.38:
            continue
        patch_a = a[y : y + bh, x : x + bw]
        patch_c = chroma[y : y + bh, x : x + bw]
        a95 = float(np.percentile(patch_a, 95))
        c95 = float(np.percentile(patch_c, 95))
        # Stempel asli kemerahan (Lab a tinggi). Bayangan scan/kertas kekuningan
        # bisa punya chroma tinggi tanpa a merah — jangan lolos.
        if a95 < 136:
            continue
        if c95 < 14:
            continue
        return True
    return False


def document_has_meterai(ocr_text: str, image: Any = None) -> bool:
    """Jika ada gambar: wajib stempel fisik. Tanpa gambar: frasa OCR 'meterai tempel'."""
    if image is not None and _bgr_from_image(image) is not None:
        return image_has_red_meterai(image)
    return ocr_has_meterai(ocr_text)


def evaluate_bpjs_kesanggupan_stamp_gate(
    ocr_text: str,
    *,
    profile_id: str | None = None,
    image: Any = None,
) -> dict[str, Any] | None:
    pid = (profile_id or "").strip().casefold()
    if pid != "bpjs_kesanggupan":
        return None
    found = document_has_meterai(ocr_text, image=image)
    if found:
        reason = "Materai merah terdeteksi pada surat kesanggupan BPJS."
    else:
        reason = "Surat kesanggupan BPJS wajib memakai materai merah."
    return {
        "stamp_pass": found,
        "reason": reason,
        "visual_checked": image is not None and _bgr_from_image(image) is not None,
    }


def apply_bpjs_kesanggupan_stamp_gate(
    validation: dict[str, Any],
    *,
    ocr_text: str,
    profile_id: str | None = None,
    image: Any = None,
) -> dict[str, Any]:
    out = dict(validation)
    profile = (profile_id or out.get("document_profile_id") or "").strip().casefold()
    gate = evaluate_bpjs_kesanggupan_stamp_gate(ocr_text, profile_id=profile, image=image)
    if gate is None:
        out.setdefault("stamp_pass", None)
        return out

    out["stamp"] = {"reason": gate["reason"], "visual_checked": gate["visual_checked"]}
    out["stamp_pass"] = gate["stamp_pass"]

    explanation = dict(out.get("explanation") or {})
    details = [gate["reason"], *list(explanation.get("detail_lines") or [])]
    explanation["detail_lines"] = details
    gates = dict(explanation.get("gates") or {})
    gates["bpjs_kesanggupan_stamp"] = {
        "pass": gate["stamp_pass"],
        "visual_checked": gate["visual_checked"],
    }
    explanation["gates"] = gates
    blockers = [b for b in list(explanation.get("primary_blockers") or []) if b != "BPJS_KESANGGUPAN_STAMP"]

    if not gate["stamp_pass"]:
        out["document_matched"] = False
        explanation["summary"] = gate["reason"]
        blockers.insert(0, "BPJS_KESANGGUPAN_STAMP")
        explanation["primary_blockers"] = blockers
        verdict = dict(out.get("verdict") or {})
        verdict["document_matched"] = False
        verdict["summary"] = gate["reason"]
        out["verdict"] = verdict
    else:
        explanation["primary_blockers"] = blockers
        identity_ok = out.get("identity_pass") in (True, None)
        if out.get("document_type_pass") is True and identity_ok and not blockers:
            out["document_matched"] = True
            verdict = dict(out.get("verdict") or {})
            verdict["document_matched"] = True
            out["verdict"] = verdict

    out["explanation"] = explanation
    return out
