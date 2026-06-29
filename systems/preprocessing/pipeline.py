"""Pipeline preprocessing gambar untuk input OCR (satu pemanggilan = seluruh langkah)."""

from __future__ import annotations

import io
import os
from typing import Final

import cv2
import numpy as np

from systems.preprocessing.auto_image_rotator import maybe_auto_image_rotator_bgr
from systems.preprocessing.realesrgan_infer import maybe_apply_realesrgan_bgr
from systems.validation.document_profiles import skip_physical_preprocess_isolation

MAX_SIDE: Final[int] = 2400
# Upscale sisi pendek dinonaktifkan secara default (0) — hindari frame membesar & crop kartu salah
# pada screenshot / dokumen digital penuh. Aktifkan: PREPROCESS_MIN_SIDE_TARGET=900
MIN_SIDE_TARGET_DEFAULT: Final[int] = 0

# Varians Laplacian: di bawah ambang = tambah sedikit kontras (tetap pelan untuk VL).
# Penguatan berlebihan membuat watermark KTP + noise jadi "tekstur rumus" → halusinasi markdown.
_BLUR_SCORE_HEAVY: Final[float] = 160.0
# detailEnhance hanya jika sangat blur (jarang); di atas ambang ini dilewati.
_DETAIL_ENHANCE_MAX_BLUR_SCORE: Final[float] = 55.0


def _decode_image(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan format didukung (JPEG, PNG, WebP, dll.).")
    return img


def _decode_image_with_exif(data: bytes) -> tuple[np.ndarray, dict]:
    """
    Dekode + terapkan EXIF Orientation (foto ponsel sering perlu ini).
    Fallback ke OpenCV jika Pillow gagal.
    """
    meta: dict = {"exif_decode": "opencv", "exif_transpose_applied": False}
    try:
        from PIL import Image, ImageOps

        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or ("transparency" in getattr(im, "info", {})):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        meta["exif_decode"] = "pillow"
        meta["exif_transpose_applied"] = True
        return bgr, meta
    except Exception:
        return _decode_image(data), meta


def _auto_rotate_quarters_mode() -> str:
    """
    off = default — tidak ada putar 90° (aman untuk gambar yang sudah lurus).
    auto / on = putar sebelum + sesudah warp; pra-warp pakai ambang ketat supaya jarang salah +90°.
    """
    v = (os.environ.get("PREPROCESS_AUTO_ROTATE_QUARTERS") or "off").strip().lower()
    if v in ("0", "false", "no", "off"):
        return "off"
    if v in ("1", "true", "yes", "on"):
        return "on"
    return "auto"


def _empty_auto_rotate_meta(*, reason: str) -> dict[str, object]:
    return {
        "auto_rotate_quarters_applied": False,
        "auto_rotate_90ccw_steps": 0,
        "auto_rotate_pre_90ccw_steps": 0,
        "auto_rotate_post_90ccw_steps": 0,
        "auto_rotate_pre_applied": False,
        "auto_rotate_post_applied": False,
        "auto_rotate_skipped_reason": reason,
    }


def _card_warp_enabled() -> bool:
    # Default mati: warp/crop kartu sering memotong screenshot email/app (NPWP digital, dll.).
    v = (os.environ.get("PREPROCESS_CARD_WARP") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _min_side_target() -> int:
    raw = (os.environ.get("PREPROCESS_MIN_SIDE_TARGET") or str(MIN_SIDE_TARGET_DEFAULT)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return MIN_SIDE_TARGET_DEFAULT


def _max_side_limit() -> int:
    raw = (os.environ.get("PREPROCESS_MAX_SIDE") or str(MAX_SIDE)).strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return MAX_SIDE


def _skip_warp_cover_ratio() -> float:
    """Jika kontur terluar menutupi ≥ rasio ini, anggap scan/full-bleed — jangan perspective warp."""
    try:
        return float(os.environ.get("PREPROCESS_SKIP_WARP_WHEN_COVER_RATIO", "0.88"))
    except ValueError:
        return 0.88


def _card_warp_style() -> str:
    """auto | axis_box | perspective — default auto mengutamakan crop lurus bila dokumen sudah tegak."""
    v = (os.environ.get("PREPROCESS_CARD_WARP_STYLE") or "auto").strip().lower()
    if v in ("perspective", "axis_box", "auto"):
        return v
    return "auto"


def _axis_only_max_skew_deg() -> float:
    """Di mode auto: jika skew kotak ≤ ini, hanya crop poros — tidak ada perspective (sumber miring)."""
    try:
        return float(os.environ.get("PREPROCESS_AUTO_AXIS_ONLY_MAX_SKEW_DEG", "24"))
    except ValueError:
        return 24.0


def _quad_skew_deg_from_box_points(box: np.ndarray) -> float:
    """Deviasi minimum sisi kotak terhadap sumbu x/y gambar, derajat ∈ [0, 45]."""
    pts = box.astype(np.float64)
    devs: list[float] = []
    for i in range(4):
        e = pts[(i + 1) % 4] - pts[i]
        ln = float(np.hypot(e[0], e[1]))
        if ln < 1e-6:
            continue
        ang = abs(float(np.degrees(np.arctan2(e[1], e[0]))))
        ang = ang % 180.0
        if ang > 90.0:
            ang = 180.0 - ang
        devs.append(min(ang, 90.0 - ang))
    return float(min(devs)) if devs else 45.0


def _rot_vec_y_after_deg(vx: float, vy: float, phi_deg: float) -> float:
    """Komponen y vektor setelah rotasi OpenCV (+phi CCW) pada vektor dari asal."""
    phi = float(np.radians(phi_deg))
    ca = float(np.cos(phi))
    sa = float(np.sin(phi))
    return sa * vx + ca * vy


def _axis_warp_and_crop_rect(
    bgr: np.ndarray,
    *,
    cx_f: float,
    cy_f: float,
    phi_deg: float,
    w_i: int,
    h_i: int,
) -> np.ndarray | None:
    """Rotasi + crop poros dengan pusat dan ukuran crop tetap."""
    h0, w0 = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((cx_f, cy_f), phi_deg, 1.0)
    cos = abs(float(M[0, 0]))
    sin = abs(float(M[0, 1]))
    nW = max(1, int(h0 * sin + w0 * cos))
    nH = max(1, int(h0 * cos + w0 * sin))
    M[0, 2] += nW / 2.0 - cx_f
    M[1, 2] += nH / 2.0 - cy_f
    rot = cv2.warpAffine(
        bgr,
        M,
        (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    x0 = int(round(nW / 2.0 - w_i / 2.0))
    y0 = int(round(nH / 2.0 - h_i / 2.0))
    x1 = min(nW, x0 + w_i)
    y1 = min(nH, y0 + h_i)
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return None
    return rot[y0:y1, x0:x1].copy()


def _narrow_band_ink_top_bias_gray(gray: np.ndarray, *, frac: float = 0.05) -> float:
    """
    Otsu: lebih banyak tinta di strip atas vs bawah (frac tinggi gambar).
    Positif ≈ judul/kepala di atas; membantu memecah seri terbalik 180° bila skor teks mendatar sama.
    """
    h0, w0 = gray.shape[:2]
    if h0 < 16 or w0 < 8:
        return 0.0
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = max(2, int(h0 * frac))
    ink_top = float(np.mean(th[:b, :] < 128))
    ink_bot = float(np.mean(th[h0 - b :, :] < 128))
    return ink_top - ink_bot


def _ink_center_upper_bias_gray(gray: np.ndarray) -> float:
    """1 − (pusat massa tinta baris / h); lebih besar ≈ konten condong ke bagian atas."""
    h0, w0 = gray.shape[:2]
    if h0 < 16 or w0 < 8:
        return 0.0
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (th < 128).astype(np.float64)
    mass = float(ink.sum())
    if mass < 8.0:
        return 0.0
    ys = np.arange(h0, dtype=np.float64)[:, np.newaxis]
    com_y = float((ink * ys).sum() / mass)
    return float(1.0 - com_y / float(h0))


def _axis_crop_from_box(bgr: np.ndarray, box_full: np.ndarray) -> np.ndarray | None:
    """
    Luruskan bidang dengan rotasi affine lalu crop persegi poros.
    Dua kandidat (±beta + penyesuaian 180°); urut: skor teks mendatar lalu bias strip 5% atas/bawah.
    Seri penuh → ambil kandidat pertama (stabil; hindari memilih 180° salah lewat heuristik lemah).
    """
    rect = cv2.minAreaRect(box_full.astype(np.float32))
    bp = cv2.boxPoints(rect).astype(np.float32)
    e0 = bp[1] - bp[0]
    e1 = bp[2] - bp[1]
    n0 = float(np.hypot(float(e0[0]), float(e0[1])))
    n1 = float(np.hypot(float(e1[0]), float(e1[1])))
    if max(n0, n1) < 1e-3:
        return None
    if n0 >= n1:
        v = e0
        mt = (bp[0] + bp[1]) * 0.5
        mb = (bp[2] + bp[3]) * 0.5
    else:
        v = e1
        mt = (bp[1] + bp[2]) * 0.5
        mb = (bp[3] + bp[0]) * 0.5

    c = np.mean(bp, axis=0)
    cx_f, cy_f = float(c[0]), float(c[1])

    beta = float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))
    mtvx, mtvy = float(mt[0] - cx_f), float(mt[1] - cy_f)
    mbvx, mbvy = float(mb[0] - cx_f), float(mb[1] - cy_f)

    (_cx_r, _cy_r), (rw, rh), _ang_ignored = rect
    ww, hh = float(rw), float(rh)
    if ww < hh:
        ww, hh = hh, ww
    w_i = max(8, int(round(ww)))
    h_i = max(8, int(round(hh)))

    cands: list[tuple[float, float, np.ndarray]] = []
    for b_try in (-beta, beta):
        phi = float(b_try)
        if _rot_vec_y_after_deg(mtvx, mtvy, phi) >= _rot_vec_y_after_deg(mbvx, mbvy, phi):
            phi += 180.0
        cropped = _axis_warp_and_crop_rect(
            bgr, cx_f=cx_f, cy_f=cy_f, phi_deg=phi, w_i=w_i, h_i=h_i
        )
        if cropped is None or cropped.size == 0:
            continue
        g = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        sc = float(_document_horizontal_text_score(g))
        nb = float(_narrow_band_ink_top_bias_gray(g))
        cands.append((sc, nb, cropped))
    if not cands:
        return None
    return max(cands, key=lambda t: (t[0], t[1]))[2]


def _auto_rotate_margin_ratio() -> float:
    try:
        return float(os.environ.get("PREPROCESS_AUTO_ROTATE_MARGIN", "1.12"))
    except ValueError:
        return 1.12


def _auto_rotate_min_delta() -> float:
    """Minimal selisih skor mutlak (best − baseline) agar benar-benar diputar."""
    try:
        return float(os.environ.get("PREPROCESS_AUTO_ROTATE_MIN_DELTA", "0.04"))
    except ValueError:
        return 0.04


def _auto_rotate_margin_no_warp() -> float:
    """Lebih ketat bila tidak ada warp kartu (gambar sudah sering lurus)."""
    try:
        return float(os.environ.get("PREPROCESS_AUTO_ROTATE_MARGIN_NO_WARP", "1.22"))
    except ValueError:
        return 1.22


def _auto_rotate_min_delta_no_warp() -> float:
    try:
        return float(os.environ.get("PREPROCESS_AUTO_ROTATE_MIN_DELTA_NO_WARP", "0.08"))
    except ValueError:
        return 0.08


def _downscale_gray_for_heuristic(gray: np.ndarray, *, target: int = 420) -> np.ndarray:
    if gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= target:
        return gray
    s = target / float(m)
    return cv2.resize(
        gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
    )


def _document_horizontal_text_score(gray: np.ndarray) -> float:
    """
    Skor heuristik: teks baris mendatar menghasilkan pola proyeksi baris yang lebih kuat.
    Dipakai memilih 0° / 90° / 180° / 270° tanpa model OCR.
    """
    if gray.size == 0:
        return 0.0
    gray = _downscale_gray_for_heuristic(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th

    def _score_one(bin_img: np.ndarray) -> float:
        proj = np.sum(bin_img < 128, axis=1).astype(np.float64)
        if proj.size < 4:
            return 0.0
        pv = float(np.var(proj) / (np.mean(proj) + 3.0))
        edge = float(
            np.mean(np.abs(bin_img[1:, :].astype(np.int16) - bin_img[:-1, :].astype(np.int16)))
        )
        return pv + 0.008 * edge

    return max(_score_one(th), _score_one(inv))


def _edge_horizontal_text_score(gray: np.ndarray) -> float:
    """
    Skor proyeksi baris pada tepi horizontal — lebih tahan pola diagonal watermark KTP
    yang sering menipu skor Otsu pada projection deskew.
    """
    if gray.size == 0:
        return 0.0
    gray = _downscale_gray_for_heuristic(gray, target=520)
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 42, 118)
    kw = max(9, min(41, w // 28))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    proj = np.sum(horiz > 0, axis=1).astype(np.float64)
    if proj.size < 4:
        return 0.0
    return float(np.var(proj) / (np.mean(proj) + 3.0))


def _blended_horizontal_text_score(gray: np.ndarray) -> float:
    otsu = float(_document_horizontal_text_score(gray))
    edge = float(_edge_horizontal_text_score(gray))
    return 0.32 * otsu + 0.68 * edge


def _estimate_text_line_skew_deg(gray: np.ndarray) -> float | None:
    """
    Estimasi kemiringan teks terhadap horizontal (derajat, positif = miring searah jarum jam).
    """
    if gray.size == 0:
        return None
    gray = _downscale_gray_for_heuristic(gray, target=640)
    h, w = gray.shape[:2]
    if h < 24 or w < 24:
        return None

    estimates: list[float] = []

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    ink = 255 - bw
    kw = max(15, min(48, w // 18))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    lines_mask = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(lines_mask, 40, 120)
    min_len = max(24, w // 9)
    hough = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(28, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(8, w // 40),
    )
    if hough is not None:
        angs: list[float] = []
        for seg in hough[:120]:
            x1, y1, x2, y2 = (int(v) for v in seg[0])
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 6.0:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if abs(ang) <= 32.0:
                angs.append(ang)
        if len(angs) >= 3:
            estimates.append(float(np.median(np.asarray(angs, dtype=np.float64))))

    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)
    kw2 = max(21, min(55, w // 12))
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw2, 1))
    blackhat = cv2.morphologyEx(blur2, cv2.MORPH_BLACKHAT, bh_kernel)
    _, bh_bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bh_edges = cv2.Canny(bh_bw, 35, 110)
    hough2 = cv2.HoughLinesP(
        bh_edges,
        1,
        np.pi / 180.0,
        threshold=40,
        minLineLength=max(30, w // 8),
        maxLineGap=18,
    )
    if hough2 is not None:
        angs2: list[float] = []
        for seg in hough2[:120]:
            x1, y1, x2, y2 = (int(v) for v in seg[0])
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 8.0:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if abs(ang) <= 28.0:
                angs2.append(ang)
        if len(angs2) >= 4:
            estimates.append(float(np.median(np.asarray(angs2, dtype=np.float64))))

    edges3 = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    hough3 = cv2.HoughLinesP(
        edges3,
        1,
        np.pi / 180.0,
        threshold=50,
        minLineLength=max(40, w // 7),
        maxLineGap=20,
    )
    if hough3 is not None:
        angs3: list[float] = []
        for seg in hough3[:100]:
            x1, y1, x2, y2 = (int(v) for v in seg[0])
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 8.0:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if abs(ang) <= 25.0:
                angs3.append(ang)
        if len(angs3) >= 5:
            estimates.append(float(np.median(np.asarray(angs3, dtype=np.float64))))

    if not estimates:
        return None
    if len(estimates) == 1:
        return estimates[0]
    med = float(np.median(np.asarray(estimates, dtype=np.float64)))
    cluster = [e for e in estimates if abs(e - med) <= 4.0]
    return float(np.median(np.asarray(cluster, dtype=np.float64)))


def _residual_text_skew_deg(gray: np.ndarray, rotate_deg: float) -> float | None:
    if abs(rotate_deg) > 1e-6:
        gray = _rotate_gray_expand(gray, rotate_deg)
    est = _estimate_text_line_skew_deg(gray)
    if est is None:
        return None
    return abs(est)


def _pick_deskew_angle_deg(gray: np.ndarray, peak_deg: float) -> tuple[float, dict[str, object]]:
    """
    Pilih sudut koreksi: gabungkan puncak proyeksi + estimasi garis teks agar tidak salah arah.
    """
    meta: dict[str, object] = {"projection_deskew_peak_deg": round(float(peak_deg), 3)}
    skew0 = _estimate_text_line_skew_deg(gray)
    if skew0 is not None:
        meta["projection_deskew_detected_skew"] = round(skew0, 3)

    candidates: list[float] = [0.0]
    if abs(peak_deg) >= 0.35:
        candidates.extend([float(peak_deg), -float(peak_deg)])
    if skew0 is not None and abs(skew0) >= 1.0:
        candidates.append(float(-skew0))

    uniq: list[float] = []
    for a in candidates:
        if not any(abs(a - u) < 0.25 for u in uniq):
            uniq.append(a)

    def rank(a: float) -> tuple[float, float, float]:
        res = _residual_text_skew_deg(gray, a)
        res_v = res if res is not None else 999.0
        g2 = _rotate_gray_expand(gray, a) if abs(a) > 1e-6 else gray
        blend = float(_blended_horizontal_text_score(g2))
        return (res_v, -blend, abs(a))

    best = min(uniq, key=rank)
    meta["projection_deskew_candidates"] = [round(a, 3) for a in uniq]
    meta["projection_deskew_chosen_deg"] = round(best, 3)
    return best, meta


def _disambiguate_180_after_warp_enabled() -> bool:
    v = (os.environ.get("PREPROCESS_DISAMBIGUATE_180") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _maybe_disambiguate_upside_down_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """
    Putar 180° hanya bila skor teks mendatar **identik** untuk 0° vs 180° **dan**
    **dua** heuristik geometri (pita atas/bawah + pusat massa teks) sama-sama mendukung balik.

    **Tidak** memutar berdasarkan “skor 180° sedikit lebih tinggi” — itu sering salah pada
    dokumen yang sudah tegak (noise / tabel KK).
    """
    meta: dict[str, object] = {"preprocess_flip_180_applied": False}
    if not _disambiguate_180_after_warp_enabled():
        meta["preprocess_flip_180_skipped_reason"] = "disabled"
        return bgr, meta
    h0, w0 = bgr.shape[:2]
    if h0 < 32 or w0 < 32:
        return bgr, meta
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b180 = cv2.rotate(bgr, cv2.ROTATE_180)
    g180 = cv2.cvtColor(b180, cv2.COLOR_BGR2GRAY)
    s0 = float(_document_horizontal_text_score(gray))
    s2 = float(_document_horizontal_text_score(g180))
    mx = max(s0, s2, 1e-6)
    rel_tie = abs(s0 - s2) / mx <= 0.025
    if not rel_tie:
        return bgr, meta
    b0 = float(_narrow_band_ink_top_bias_gray(gray))
    b2 = float(_narrow_band_ink_top_bias_gray(g180))
    c0 = float(_ink_center_upper_bias_gray(gray))
    c2 = float(_ink_center_upper_bias_gray(g180))
    band_gap = 0.02
    com_gap = 0.045
    if (b2 > b0 + band_gap) and (c2 > c0 + com_gap):
        meta["preprocess_flip_180_applied"] = True
        meta["preprocess_flip_180_reason"] = "dual_spatial_tie_break"
        return b180, meta
    return bgr, meta


def _projection_deskew_enabled() -> bool:
    v = (os.environ.get("PREPROCESS_PROJECTION_DESKEW") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _projection_deskew_probe_max_side() -> int:
    try:
        return int(os.environ.get("PREPROCESS_PROJECTION_DESKEW_PROBE_MAX", "520"))
    except ValueError:
        return 520


def _projection_deskew_max_deg() -> float:
    try:
        return float(os.environ.get("PREPROCESS_PROJECTION_DESKEW_MAX_DEG", "22"))
    except ValueError:
        return 22.0


def _rotate_bgr_expand(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotasi CCW (OpenCV), kanvas diperbesar; isi kosong diisi replikasi tepi."""
    if abs(angle_deg) < 1e-9:
        return bgr
    h0, w0 = bgr.shape[:2]
    cx_f = float(w0 - 1) * 0.5
    cy_f = float(h0 - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx_f, cy_f), float(angle_deg), 1.0)
    cos = abs(float(M[0, 0]))
    sin = abs(float(M[0, 1]))
    nW = max(1, int(h0 * sin + w0 * cos))
    nH = max(1, int(h0 * cos + w0 * sin))
    M[0, 2] += nW / 2.0 - cx_f
    M[1, 2] += nH / 2.0 - cy_f
    return cv2.warpAffine(
        bgr,
        M,
        (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate_gray_expand(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return gray
    h0, w0 = gray.shape[:2]
    cx_f = float(w0 - 1) * 0.5
    cy_f = float(h0 - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx_f, cy_f), float(angle_deg), 1.0)
    cos = abs(float(M[0, 0]))
    sin = abs(float(M[0, 1]))
    nW = max(1, int(h0 * sin + w0 * cos))
    nH = max(1, int(h0 * cos + w0 * sin))
    M[0, 2] += nW / 2.0 - cx_f
    M[1, 2] += nH / 2.0 - cy_f
    return cv2.warpAffine(
        gray,
        M,
        (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _projection_peak_skew_deg(
    gray: np.ndarray,
    *,
    max_deg: float,
    coarse_step: float = 1.15,
    refine_step: float = 0.35,
) -> tuple[float, float, float]:
    """
    Cari sudut rotasi (OpenCV, CCW positif) yang memaksimalkan skor teks mendatar.
    Menghindari ambigu ± seperti minAreaRect: tanda sudut hasil pencarian = arah koreksi yang dipakai.
    Returns: (best_angle_deg, base_score, best_score).
    """
    h0, w0 = gray.shape[:2]
    if h0 < 8 or w0 < 8:
        return 0.0, 0.0, 0.0
    base_sc = float(_blended_horizontal_text_score(gray))
    best_a = 0.0
    best_sc = base_sc
    tie_eps = 1e-4

    def consider(aa: float, sc: float) -> None:
        nonlocal best_a, best_sc
        if sc > best_sc + tie_eps or (
            abs(sc - best_sc) <= tie_eps and abs(aa) + 1e-6 < abs(best_a)
        ):
            best_sc = sc
            best_a = aa

    a = -float(max_deg)
    while a <= float(max_deg) + 1e-6:
        aa = float(a)
        g2 = _rotate_gray_expand(gray, aa) if abs(aa) > 1e-9 else gray
        consider(aa, float(_blended_horizontal_text_score(g2)))
        a += coarse_step

    lo = max(-float(max_deg), best_a - coarse_step)
    hi = min(float(max_deg), best_a + coarse_step)
    rr = float(refine_step)
    x = lo
    while x <= hi + 1e-6:
        aa = float(round(x, 4))
        g2 = _rotate_gray_expand(gray, aa)
        consider(aa, float(_blended_horizontal_text_score(g2)))
        x += rr

    return best_a, base_sc, best_sc


def _maybe_projection_deskew_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """
    Luruskan kemiringan kecil dengan memindai sudut: pilih rotasi yang skor proyeksi baris terbaik.
    Sudut hasil = rotasi CCW OpenCV yang diterapkan (positif = putar gambar berlawanan jarum jam).
    """
    meta: dict[str, object] = {
        "projection_deskew_applied": False,
        "projection_deskew_deg": 0.0,
        "projection_deskew_base_score": 0.0,
        "projection_deskew_best_score": 0.0,
    }
    if not _projection_deskew_enabled():
        meta["projection_deskew_skipped_reason"] = "disabled"
        return bgr, meta
    h0, w0 = bgr.shape[:2]
    if h0 < 16 or w0 < 16:
        meta["projection_deskew_skipped_reason"] = "too_small"
        return bgr, meta
    max_side = _projection_deskew_probe_max_side()
    m = max(h0, w0)
    scale = min(1.0, max_side / float(m))
    if scale < 1.0:
        small_bgr = cv2.resize(
            bgr,
            (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_bgr = bgr
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    max_deg = _projection_deskew_max_deg()
    peak_a, base_sc, peak_sc = _projection_peak_skew_deg(gray, max_deg=max_deg)
    meta["projection_deskew_base_score"] = round(base_sc, 4)
    meta["projection_deskew_best_score"] = round(peak_sc, 4)

    chosen_a, pick_meta = _pick_deskew_angle_deg(gray, peak_a)
    meta.update(pick_meta)

    skew0 = meta.get("projection_deskew_detected_skew")
    if isinstance(skew0, (int, float)) and abs(float(skew0)) < 2.0 and abs(chosen_a) < 0.35:
        meta["projection_deskew_skipped_reason"] = "already_straight"
        return bgr, meta

    res0 = _residual_text_skew_deg(gray, 0.0)
    res1 = _residual_text_skew_deg(gray, chosen_a)
    if res0 is not None:
        meta["projection_deskew_residual_skew_before"] = round(res0, 3)
    if res1 is not None:
        meta["projection_deskew_residual_skew_after"] = round(res1, 3)

    if abs(chosen_a) < 0.35:
        meta["projection_deskew_skipped_reason"] = "angle_too_small"
        return bgr, meta

    rel_imp = (peak_sc - base_sc) / max(base_sc, 1e-6)
    abs_imp = peak_sc - base_sc
    improves_residual = (
        res0 is not None and res1 is not None and res1 + 0.4 < res0
    )
    if not improves_residual and abs_imp < 0.012 and rel_imp < 0.018:
        meta["projection_deskew_skipped_reason"] = "low_gain"
        return bgr, meta
    if res0 is not None and res1 is not None and res1 > res0 + 0.8:
        meta["projection_deskew_skipped_reason"] = "worsens_residual_skew"
        return bgr, meta

    if scale < 1.0:
        chosen_a = float(chosen_a)

    out = _rotate_bgr_expand(bgr, chosen_a)
    meta["projection_deskew_applied"] = True
    meta["projection_deskew_deg"] = round(chosen_a, 3)
    return out, meta


def _pre_rotate_margin_delta(mode: str) -> tuple[float, float]:
    """Ambang pra-warp; longgarkan hanya lewat env bila perlu foto menyamping ekstrem."""
    try:
        em = (os.environ.get("PREPROCESS_AUTO_ROTATE_PRE_MARGIN") or "").strip()
        ed = (os.environ.get("PREPROCESS_AUTO_ROTATE_PRE_MIN_DELTA") or "").strip()
        if em and ed:
            return float(em), float(ed)
    except ValueError:
        pass
    if mode == "on":
        return 1.24, 0.10
    return 1.34, 0.14


def _auto_rotate_allow_180_quarter() -> bool:
    """Tanpa ini, pemutaran kuartal tidak mempertimbangkan 180° (k=2) — kurangi dokumen tegak jadi terbalik."""
    v = (os.environ.get("PREPROCESS_AUTO_ROTATE_ALLOW_180") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _maybe_auto_rotate_quarters(
    bgr: np.ndarray,
    *,
    card_warped: bool,
    margin: float | None = None,
    min_delta: float | None = None,
    meta_prefix: str = "auto_rotate",
) -> tuple[np.ndarray, dict]:
    """Putar kelipatan 90° agar teks cenderung mendatar (heuristik proyeksi baris)."""
    pfx_applied = f"{meta_prefix}_applied"
    pfx_steps = f"{meta_prefix}_90ccw_steps"
    meta: dict[str, object] = {
        pfx_applied: False,
        pfx_steps: 0,
        f"{meta_prefix}_best_score": 0.0,
        f"{meta_prefix}_baseline_score": 0.0,
        f"{meta_prefix}_scores_all_quarters": [],
        f"{meta_prefix}_effective_margin": 0.0,
        f"{meta_prefix}_effective_min_delta": 0.0,
        f"{meta_prefix}_card_warped_context": card_warped,
        f"{meta_prefix}_180_quarter_forbidden": not _auto_rotate_allow_180_quarter(),
    }

    if margin is None:
        m = _auto_rotate_margin_ratio()
        d = _auto_rotate_min_delta()
        if not card_warped:
            m = max(m, _auto_rotate_margin_no_warp())
            d = max(d, _auto_rotate_min_delta_no_warp())
    else:
        m = float(margin)
        d = float(min_delta) if min_delta is not None else _auto_rotate_min_delta()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    scores: list[tuple[int, float]] = []
    g = gray
    for k in range(4):
        scores.append((k, _document_horizontal_text_score(g)))
        if k < 3:
            g = cv2.rotate(g, cv2.ROTATE_90_COUNTERCLOCKWISE)

    pool = list(scores) if _auto_rotate_allow_180_quarter() else [t for t in scores if t[0] != 2]
    mx = max(s for _, s in pool)
    # Jika beberapa orientasi skornya hampir sama dengan yang terbaik, utamakan k terkecil.
    atol = max(0.02, 1e-5 * max(mx, 1.0))
    tied_top = [(k, s) for k, s in pool if s >= mx - atol]
    best_k, best_s = min(tied_top, key=lambda t: t[0])
    s0 = scores[0][1]
    meta[f"{meta_prefix}_baseline_score"] = round(s0, 4)
    meta[f"{meta_prefix}_best_score"] = round(best_s, 4)
    meta[f"{meta_prefix}_scores_all_quarters"] = [round(s, 4) for _, s in scores]
    meta[f"{meta_prefix}_effective_margin"] = round(m, 4)
    meta[f"{meta_prefix}_effective_min_delta"] = round(d, 4)

    if best_k != 0 and (best_s < s0 * m or (best_s - s0) < d):
        best_k = 0

    if best_k == 0:
        meta[pfx_steps] = 0
        return bgr, meta

    out = bgr
    for _ in range(best_k):
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    meta[pfx_applied] = True
    meta[pfx_steps] = best_k
    return out, meta


def _supplement_quarter_pre_warp_enabled() -> bool:
    """
    Default mati: putaran ±90° pra-warp mudah mengganggu scan/dokumen yang sudah tegak.
    Aktifkan dengan PREPROCESS_SUPPLEMENT_QUARTER_PRE_WARP=1 bila banyak foto KK/KTP menyamping.
    """
    v = (os.environ.get("PREPROCESS_SUPPLEMENT_QUARTER_PRE_WARP") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _right_tilt_90_ccw_enabled() -> bool:
    """
    Opsional: putar 90° CCW hanya bila kuartal k=1 **jelas** lebih baik dari tegak (default: mati).

    Default mati karena heuristik longgar memicu putar pada gambar yang sudah normal.
    """
    v = (os.environ.get("PREPROCESS_RIGHT_TILT_90_CCW") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _right_tilt_90_ccw_min_ratio_over_s0() -> float:
    try:
        return float(os.environ.get("PREPROCESS_RIGHT_TILT_90_CCW_MIN_RATIO", "1.18"))
    except ValueError:
        return 1.18


def _right_tilt_90_ccw_min_abs_lead() -> float:
    try:
        return float(os.environ.get("PREPROCESS_RIGHT_TILT_90_CCW_MIN_LEAD", "0.10"))
    except ValueError:
        return 0.10


def _supplement_tie_min_lead_over_s0() -> float:
    """
    Jika s1≈s3 (seri), putar hanya bila max(s1,s3) unggul s0 sekurang-kurangnya nilai ini.
    Membedakan KK benar-benar salah 90° vs gambar tegak yang skor proyeksi mirip pola "salah 90°".
    """
    try:
        return float(os.environ.get("PREPROCESS_SUPPLEMENT_TIE_MIN_LEAD", "78"))
    except ValueError:
        return 78.0


def _maybe_supplement_quarter_pre_warp_if_off(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """
    Satu putaran ±90° sebelum warp jika env putar penuh = off **dan**
    PREPROCESS_SUPPLEMENT_QUARTER_PRE_WARP=1 (default mati — hindari putar dokumen yang sudah tegak).

    Bila s1≈s3 hanya putar jika unggulan 90° atas s0 ≥ PREPROCESS_SUPPLEMENT_TIE_MIN_LEAD (default 78).
    """
    meta: dict[str, object] = {
        "auto_rotate_supplement_pre_applied": False,
        "auto_rotate_supplement_pre_90ccw_steps": 0,
        "auto_rotate_supplement_pre_scores": [],
    }
    if _auto_rotate_quarters_mode() != "off" or not _supplement_quarter_pre_warp_enabled():
        return bgr, meta
    h0, w0 = bgr.shape[:2]
    portrait = h0 >= int(w0 * 1.03)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scores: list[tuple[int, float]] = []
    g = gray
    for k in range(4):
        scores.append((k, _document_horizontal_text_score(g)))
        if k < 3:
            g = cv2.rotate(g, cv2.ROTATE_90_COUNTERCLOCKWISE)

    s0, s1, s2, s3 = (scores[i][1] for i in range(4))
    meta["auto_rotate_supplement_pre_scores"] = [round(s0, 4), round(s1, 4), round(s2, 4), round(s3, 4)]

    ninety_best = max(s1, s3)
    if ninety_best < s2 - 1e-6:
        return bgr, meta
    if portrait:
        if ninety_best <= s0 * 1.06 or (ninety_best - s0) < 0.03:
            return bgr, meta
    else:
        if ninety_best <= s0 * 1.15 or (ninety_best - s0) < 0.06:
            return bgr, meta

    # Seri s1 vs s3: utamakan k=1 hanya jika lonjakan skor atas s0 cukup besar (hindari putar tegak).
    sep = max(4.0, 0.015 * max(ninety_best, 1.0))
    lead_need = _supplement_tie_min_lead_over_s0()
    if abs(s1 - s3) < sep:
        if (ninety_best - s0) < lead_need:
            return bgr, meta
        best_k = 1
    elif s1 > s3:
        best_k = 1
    else:
        best_k = 3

    out = bgr
    for _ in range(best_k):
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    meta["auto_rotate_supplement_pre_applied"] = True
    meta["auto_rotate_supplement_pre_90ccw_steps"] = best_k
    return out, meta


def _maybe_right_tilt_90_ccw_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """
    Satu putaran 90° CCW **hanya** jika skor teks mendatar pada k=1 jauh lebih baik dari k=0
    dan k=1 mengalahkan k=3 (bukan lebih baik mutar searah jarum jam).

    Tidak memakai sudut proyeksi kecil: itu memperbaiki kemiringan sedikit, bukan salah kuartal,
    dan sering salah pada dokumen yang sudah tegak.
    """
    meta: dict[str, object] = {
        "right_tilt_90_ccw_applied": False,
        "right_tilt_90_ccw_projection_deg": 0.0,
        "right_tilt_90_ccw_skipped_reason": "",
    }
    if not _right_tilt_90_ccw_enabled():
        meta["right_tilt_90_ccw_skipped_reason"] = "disabled"
        return bgr, meta
    h0, w0 = bgr.shape[:2]
    if h0 < 32 or w0 < 32:
        meta["right_tilt_90_ccw_skipped_reason"] = "too_small"
        return bgr, meta

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scores: list[tuple[int, float]] = []
    g = gray
    for k in range(4):
        scores.append((k, _document_horizontal_text_score(g)))
        if k < 3:
            g = cv2.rotate(g, cv2.ROTATE_90_COUNTERCLOCKWISE)
    s0, s1, s2, s3 = (scores[i][1] for i in range(4))
    meta["right_tilt_90_ccw_scores_s0_s1_s2_s3"] = [
        round(s0, 4),
        round(s1, 4),
        round(s2, 4),
        round(s3, 4),
    ]

    if max(s1, s3) < s2 - 1e-6:
        meta["right_tilt_90_ccw_skipped_reason"] = "upside_down_quarter_better"
        return bgr, meta

    sep = max(4.0, 0.015 * max(s1, s3, 1.0))
    if s1 + 1e-6 < s3 + sep:
        meta["right_tilt_90_ccw_skipped_reason"] = "cw_quarter_preferred_or_tied"
        return bgr, meta

    min_ratio = _right_tilt_90_ccw_min_ratio_over_s0()
    min_lead = _right_tilt_90_ccw_min_abs_lead()
    if s1 < s0 * min_ratio or (s1 - s0) < min_lead:
        meta["right_tilt_90_ccw_skipped_reason"] = "insufficient_lead_over_upright"
        return bgr, meta

    out = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    meta["right_tilt_90_ccw_applied"] = True
    meta["right_tilt_90_ccw_effective_min_ratio"] = round(min_ratio, 4)
    meta["right_tilt_90_ccw_effective_min_lead"] = round(min_lead, 4)
    return out, meta


def _maybe_resize(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Hanya downscale jika terlalu besar; upscale hanya bila PREPROCESS_MIN_SIDE_TARGET > 0."""
    h, w = bgr.shape[:2]
    max_side = max(h, w)
    min_side = min(h, w)
    max_lim = _max_side_limit()
    min_tgt = _min_side_target()
    meta: dict[str, object] = {
        "resize_applied": False,
        "resize_scale": 1.0,
        "resize_reason": None,
        "resize_input_hw": {"width": w, "height": h},
    }

    if max_side > max_lim:
        scale = max_lim / max_side
        out = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        meta.update(resize_applied=True, resize_scale=round(scale, 4), resize_reason="downscale_max_side")
        return out, meta

    if min_tgt > 0 and min_side < min_tgt:
        scale = min_tgt / min_side
        scale = min(scale, 3.0)
        if scale > 1.001:
            out = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            meta.update(resize_applied=True, resize_scale=round(scale, 4), resize_reason="upscale_min_side")
            return out, meta

    return bgr, meta


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


def _full_bleed_straighten_enabled() -> bool:
    v = (os.environ.get("PREPROCESS_FULL_BLEED_STRAIGHTEN") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _accept_axis_crop_candidate(
    bgr: np.ndarray,
    axis_img: np.ndarray,
    *,
    skew_deg: float,
    area_frac: float,
) -> bool:
    """Terima crop hanya bila tidak membalik orientasi dan benar-benar meluruskan dokumen."""
    h0, w0 = bgr.shape[:2]
    hh, ww = axis_img.shape[1], axis_img.shape[0]
    in_land = w0 >= h0
    out_land = ww >= hh
    if in_land != out_land:
        return False

    g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(axis_img, cv2.COLOR_BGR2GRAY)
    r0 = _residual_text_skew_deg(g0, 0.0)
    r1 = _residual_text_skew_deg(g1, 0.0)
    if r0 is not None and r1 is not None and r1 + 1.0 < r0:
        return True

    if abs(skew_deg) >= 1.4:
        return True
    if area_frac < 0.88 and abs(skew_deg) >= 0.8:
        return True
    if area_frac > 0.92 and abs(skew_deg) < 1.4:
        return False
    return abs(skew_deg) >= 1.0


def _find_best_axis_crop_from_lab_morphology(
    bgr: np.ndarray,
    *,
    min_area_frac: float = 0.10,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Cari kontur dokumen (LAB+morfologi) lalu axis-crop terbaik."""
    meta: dict[str, object] = {}
    h0, w0 = bgr.shape[:2]
    max_det = 900
    sc = min(1.0, max_det / max(h0, w0))
    sw, sh = int(w0 * sc), int(h0 * sc)
    if sw < 40 or sh < 40:
        return None, meta

    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    warp_style = _card_warp_style()
    skew_soft = _axis_only_max_skew_deg()
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch = cv2.GaussianBlur(lab[:, :, 0], (13, 13), 0)

    best_img: np.ndarray | None = None
    best_meta: dict[str, object] = {}
    best_key = (-1.0, -1.0)

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
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(c)
            area_frac = area / float(sh * sw)
            if area_frac < min_area_frac:
                break
            rect = cv2.minAreaRect(c)
            box_full = cv2.boxPoints(rect).astype(np.float32)
            box_full[:, 0] /= sc
            box_full[:, 1] /= sc
            skew_deg = _quad_skew_deg_from_box_points(box_full)
            if warp_style == "perspective":
                continue
            if warp_style == "auto" and skew_deg > skew_soft:
                continue
            axis_img = _axis_crop_from_box(bgr, box_full)
            if axis_img is None or axis_img.size == 0:
                continue
            ww, hh = axis_img.shape[1], axis_img.shape[0]
            if not _aspect_ok(ww, hh) or ww * hh < w0 * h0 * 0.04:
                continue
            if not _accept_axis_crop_candidate(
                bgr, axis_img, skew_deg=skew_deg, area_frac=area_frac
            ):
                continue
            key = (area_frac, skew_deg)
            if key > best_key:
                best_key = key
                best_img = axis_img
                best_meta = {
                    "axis_crop_source": "lab_morphology",
                    "axis_crop_area_frac": round(area_frac, 4),
                    "axis_crop_skew_deg": round(skew_deg, 3),
                    "axis_crop_invert": invert,
                }
        if best_img is not None:
            break
    return best_img, best_meta


def _maybe_full_bleed_axis_straighten(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """
    Luruskan dokumen ID dari foto: kontur LAB (KTP di atas kain/latar) atau kontur
    full-bleed bila bidang kartu memenuhi frame.
    """
    meta: dict[str, object] = {
        "full_bleed_straighten_applied": False,
        "full_bleed_straighten_skew_deg": 0.0,
        "full_bleed_straighten_skipped_reason": "",
    }
    if not _full_bleed_straighten_enabled():
        meta["full_bleed_straighten_skipped_reason"] = "disabled"
        return bgr, meta

    h0, w0 = bgr.shape[:2]
    if h0 < 48 or w0 < 48:
        meta["full_bleed_straighten_skipped_reason"] = "too_small"
        return bgr, meta

    axis_img, morph_meta = _find_best_axis_crop_from_lab_morphology(bgr, min_area_frac=0.08)
    if axis_img is not None:
        meta.update(morph_meta)
        meta["full_bleed_straighten_applied"] = True
        meta["full_bleed_straighten_mode"] = "lab_morphology"
        meta["full_bleed_straighten_skew_deg"] = morph_meta.get("axis_crop_skew_deg", 0.0)
        return axis_img, meta

    max_det = 900
    sc = min(1.0, max_det / max(h0, w0))
    sw, sh = int(w0 * sc), int(h0 * sc)
    if sw < 40 or sh < 40:
        meta["full_bleed_straighten_skipped_reason"] = "too_small"
        return bgr, meta

    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    gray_quick = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, th_cover = cv2.threshold(gray_quick, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts_cover, _ = cv2.findContours(th_cover, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cover_thr = _skip_warp_cover_ratio()
    if not cnts_cover:
        meta["full_bleed_straighten_skipped_reason"] = "no_contour"
        return bgr, meta

    c_cover = max(cnts_cover, key=cv2.contourArea)
    if cv2.contourArea(c_cover) < cover_thr * sh * sw:
        meta["full_bleed_straighten_skipped_reason"] = "no_document_contour"
        return bgr, meta

    rect = cv2.minAreaRect(c_cover)
    box_full = cv2.boxPoints(rect).astype(np.float32)
    box_full[:, 0] /= sc
    box_full[:, 1] /= sc
    skew_deg = _quad_skew_deg_from_box_points(box_full)
    meta["full_bleed_straighten_detected_skew"] = round(skew_deg, 3)
    if skew_deg < 1.4:
        meta["full_bleed_straighten_skipped_reason"] = "already_axis_aligned"
        return bgr, meta

    axis_img = _axis_crop_from_box(bgr, box_full)
    if axis_img is None or axis_img.size == 0:
        meta["full_bleed_straighten_skipped_reason"] = "axis_crop_failed"
        return bgr, meta
    ww, hh = axis_img.shape[1], axis_img.shape[0]
    if not _aspect_ok(ww, hh) or ww * hh < w0 * h0 * 0.04:
        meta["full_bleed_straighten_skipped_reason"] = "axis_crop_rejected"
        return bgr, meta
    area_frac = float(cv2.contourArea(c_cover)) / float(sh * sw)
    if not _accept_axis_crop_candidate(
        bgr, axis_img, skew_deg=skew_deg, area_frac=area_frac
    ):
        meta["full_bleed_straighten_skipped_reason"] = "axis_crop_low_benefit"
        return bgr, meta

    meta["full_bleed_straighten_applied"] = True
    meta["full_bleed_straighten_mode"] = "full_bleed_outer"
    meta["full_bleed_straighten_skew_deg"] = round(skew_deg, 3)
    return axis_img, meta


def _try_extract_card_warp(bgr: np.ndarray) -> tuple[np.ndarray, bool, dict]:
    """
    Coba isolasi bidang kartu (KTP/dokumen persegi panjang) dari latar foto.
    Mode `auto` (default): jika kotak hampir sejajar sumbu, pakai crop poros — hindari perspective
    yang sering membuat dokumen lurus tampak “miring ke kanan/kiri”.
    """
    info: dict[str, object] = {"card_warp_mode": "none"}

    if not _card_warp_enabled():
        info["card_warp_mode"] = "disabled"
        return bgr, False, info

    h0, w0 = bgr.shape[:2]
    max_det = 900
    sc = min(1.0, max_det / max(h0, w0))
    sw, sh = int(w0 * sc), int(h0 * sc)
    if sw < 40 or sh < 40:
        return bgr, False, info

    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    # Scan / dokumen full-bleed: hampir seluruh frame isi konten → jangan perspective warp
    gray_quick = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, th_cover = cv2.threshold(gray_quick, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts_cover, _ = cv2.findContours(th_cover, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cover_thr = _skip_warp_cover_ratio()
    if cnts_cover:
        c_cover = max(cnts_cover, key=cv2.contourArea)
        if cv2.contourArea(c_cover) >= cover_thr * sh * sw:
            rect = cv2.minAreaRect(c_cover)
            box_full = cv2.boxPoints(rect).astype(np.float32)
            box_full[:, 0] /= sc
            box_full[:, 1] /= sc
            skew_deg = _quad_skew_deg_from_box_points(box_full)
            if skew_deg >= 1.4:
                axis_img = _axis_crop_from_box(bgr, box_full)
                if axis_img is not None and axis_img.size > 0:
                    ww, hh = axis_img.shape[1], axis_img.shape[0]
                    if _aspect_ok(ww, hh) and ww * hh >= w0 * h0 * 0.04:
                        info["card_warp_mode"] = "full_bleed_axis"
                        info["card_warp_detected_skew_deg"] = round(skew_deg, 3)
                        return axis_img, True, info
            info["card_warp_mode"] = "skipped_full_bleed"
            return bgr, False, info

    warp_style = _card_warp_style()
    skew_soft = _axis_only_max_skew_deg()

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch = cv2.GaussianBlur(lab[:, :, 0], (13, 13), 0)

    best_warp: np.ndarray | None = None
    best_mode = "none"

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
            # Titik kotak float (hindari int32 di skala kecil) — mengurangi bias sudut.
            rect = cv2.minAreaRect(c)
            box_full = cv2.boxPoints(rect).astype(np.float32)
            box_full[:, 0] /= sc
            box_full[:, 1] /= sc
            skew_deg = _quad_skew_deg_from_box_points(box_full)

            # auto + kotak cukup lurus: HANYA crop poros — jangan perspective (sering bikin "miring").
            if warp_style == "auto" and skew_deg <= skew_soft:
                axis_img = _axis_crop_from_box(bgr, box_full)
                if axis_img is not None and axis_img.size > 0:
                    ww, hh = axis_img.shape[1], axis_img.shape[0]
                    if _aspect_ok(ww, hh) and ww * hh >= w0 * h0 * 0.04:
                        best_warp = axis_img
                        best_mode = "axis_box"
                        break
                continue

            if warp_style == "axis_box":
                axis_img = _axis_crop_from_box(bgr, box_full)
                if axis_img is not None and axis_img.size > 0:
                    ww, hh = axis_img.shape[1], axis_img.shape[0]
                    if _aspect_ok(ww, hh) and ww * hh >= w0 * h0 * 0.04:
                        best_warp = axis_img
                        best_mode = "axis_box"
                        break
                continue

            # auto (skew besar) atau perspective: sesuaikan dokumen miring di foto
            warped = _four_point_warp(bgr, box_full)
            ww, hh = warped.shape[1], warped.shape[0]
            if _aspect_ok(ww, hh) and ww * hh >= w0 * h0 * 0.04:
                best_warp = warped
                best_mode = "perspective"
                break
        if best_warp is not None:
            break

    if best_warp is None:
        info["card_warp_mode"] = "failed"
        return bgr, False, info

    info["card_warp_mode"] = best_mode
    info["card_warp_style_setting"] = warp_style
    return best_warp, True, info


def _laplacian_blur_score(gray: np.ndarray) -> float:
    """
    Metrik keburaman: varians respons Laplacian (downscale untuk stabil).
    Nilai lebih kecil ≈ lebih blur; foto tajam biasanya ratusan ke atas.
    """
    if gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    m = min(h, w)
    if m > 520:
        s = 520.0 / float(m)
        sw, sh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _unsharp(eq: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(eq, 1.0 + amount, blur, -amount, 0)


def _enhance_grayscale_for_ocr(
    gray: np.ndarray,
    *,
    blur_score_hint: float | None = None,
) -> np.ndarray:
    """
    Perjelas teks pelan untuk input VL/OCR: pertahankan midtone, hindari kontras ekstrem
    pada watermark (sering memicu keluaran markdown/LaTeX berulang).
    """
    h, w = gray.shape[:2]
    min_side = min(h, w)

    blur_score = float(blur_score_hint) if blur_score_hint is not None else _laplacian_blur_score(gray)
    heavy = blur_score < _BLUR_SCORE_HEAVY

    blur_k = _odd_kernel_from_fraction(min_side, 0.10, 101)
    illum = cv2.GaussianBlur(gray, (blur_k, blur_k), 0).astype(np.float32)
    illum = np.maximum(illum, 24.0)
    # Lebih lembut dari *252 agar tidak "membakar" pola latar KTP.
    flat = np.clip(gray.astype(np.float32) / illum * 238.0, 0, 255).astype(np.uint8)

    # Sedikit lebih banyak penghalusan pada blur agar watermark halus sebelum CLAHE.
    if heavy:
        smooth = cv2.bilateralFilter(flat, d=5, sigmaColor=42, sigmaSpace=42)
    else:
        smooth = cv2.bilateralFilter(flat, d=5, sigmaColor=32, sigmaSpace=32)

    if heavy:
        clahe = cv2.createCLAHE(clipLimit=1.75, tileGridSize=(8, 8))
    else:
        clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    eq = clahe.apply(smooth)
    # Campur kembali dengan smooth: kurangi artefak halftone / diagonal watermark KTP.
    eq = cv2.addWeighted(smooth, 0.42, eq, 0.58, 0)

    if heavy:
        sharp = _unsharp(eq, sigma=1.15, amount=0.22)
    else:
        sharp = _unsharp(eq, sigma=0.85, amount=0.16)

    return np.clip(sharp, 0, 255).astype(np.uint8)


def _hough_median_skew_deg(gray: np.ndarray) -> float | None:
    """Estimasi kemiringan dominan dari garis hampir-horizontal (post-enhance)."""
    if gray.size == 0:
        return None
    gray = _downscale_gray_for_heuristic(gray, target=720)
    h, w = gray.shape[:2]
    if h < 24 or w < 24:
        return None
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    min_len = max(36, w // 8)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=55,
        minLineLength=min_len,
        maxLineGap=max(12, w // 35),
    )
    if lines is None:
        return None
    angs: list[float] = []
    for seg in lines[:100]:
        x1, y1, x2, y2 = (int(v) for v in seg[0])
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 8.0:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if abs(ang) <= 22.0:
            angs.append(ang)
    if len(angs) < 5:
        return None
    return float(np.median(np.asarray(angs, dtype=np.float64)))


def _maybe_post_enhance_micro_deskew_gray(gray: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Rapikan kemiringan artefak setelah enhance (garis diagonal watermark KTP)."""
    meta: dict[str, object] = {
        "post_enhance_micro_deskew_applied": False,
        "post_enhance_micro_deskew_deg": 0.0,
    }
    skew = _hough_median_skew_deg(gray)
    if skew is None or abs(skew) < 2.2:
        meta["post_enhance_micro_deskew_skipped_reason"] = "angle_too_small"
        return gray, meta
    corr = float(-skew)
    if abs(corr) > 8.0:
        meta["post_enhance_micro_deskew_skipped_reason"] = "angle_too_large"
        return gray, meta
    corrected = _rotate_gray_expand(gray, corr)
    skew1 = _hough_median_skew_deg(corrected)
    if skew1 is not None and abs(skew1) + 0.8 >= abs(skew):
        meta["post_enhance_micro_deskew_skipped_reason"] = "low_improvement"
        return gray, meta
    meta["post_enhance_micro_deskew_applied"] = True
    meta["post_enhance_micro_deskew_deg"] = round(corr, 3)
    meta["post_enhance_micro_deskew_detected_skew"] = round(skew, 3)
    return corrected, meta


def _maybe_pre_enhance_micro_deskew_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Koreksi kemiringan kecil terakhir sebelum enhance grayscale (hindari artefak diagonal)."""
    meta: dict[str, object] = {
        "pre_enhance_micro_deskew_applied": False,
        "pre_enhance_micro_deskew_deg": 0.0,
    }
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    skew = _estimate_text_line_skew_deg(gray)
    if skew is None or abs(skew) < 1.8:
        meta["pre_enhance_micro_deskew_skipped_reason"] = "angle_too_small"
        return bgr, meta
    corr = float(-skew)
    if abs(corr) > 20.0:
        meta["pre_enhance_micro_deskew_skipped_reason"] = "angle_too_large"
        return bgr, meta
    res0 = _residual_text_skew_deg(gray, 0.0)
    res1 = _residual_text_skew_deg(gray, corr)
    if res0 is not None and res1 is not None and res1 > res0 + 0.6:
        meta["pre_enhance_micro_deskew_skipped_reason"] = "worsens_residual_skew"
        return bgr, meta
    out = _rotate_bgr_expand(bgr, corr)
    meta["pre_enhance_micro_deskew_applied"] = True
    meta["pre_enhance_micro_deskew_deg"] = round(corr, 3)
    meta["pre_enhance_micro_deskew_detected_skew"] = round(float(skew), 3)
    return out, meta


def _maybe_detail_enhance_bgr(bgr: np.ndarray, *, blur_score: float) -> np.ndarray:
    """
    Hampir selalu dilewati: detailEnhance + CLAHE kuat mempertegas watermark KTP
    dan memicu halusinasi (mis. blok LaTeX) pada PaddleOCR-VL.
    Hanya dipakai jika blur ekstrem (sangat jarang).
    """
    if blur_score >= _DETAIL_ENHANCE_MAX_BLUR_SCORE:
        return bgr
    try:
        return cv2.detailEnhance(bgr, None, sigma_s=8, sigma_r=0.08)
    except cv2.error:
        return bgr


def _skip_physical_isolation_meta(profile: str) -> tuple[dict[str, object], dict[str, object]]:
    """Metadata bila crop kartu / full-bleed dilewati untuk profil tertentu."""
    reason = f"profile_{profile or 'unknown'}"
    return (
        {
            "full_bleed_straighten_applied": False,
            "full_bleed_straighten_skipped_reason": reason,
        },
        {"card_warp_mode": "skipped_profile", "card_warp_profile": profile or None},
    )


def _preprocess_work_bgr(
    bgr: np.ndarray,
    *,
    document_profile_id: str = "",
) -> tuple[np.ndarray, bool, dict]:
    work, resize_meta = _maybe_resize(bgr)
    work, air_meta = maybe_auto_image_rotator_bgr(work)
    work, supplement_meta = _maybe_supplement_quarter_pre_warp_if_off(work)
    mode = _auto_rotate_quarters_mode()
    profile = (document_profile_id or "").strip().casefold()
    skip_isolation = bool(profile) and skip_physical_preprocess_isolation(profile)
    if bool(supplement_meta.get("auto_rotate_supplement_pre_applied")):
        rtl_meta = {
            "right_tilt_90_ccw_applied": False,
            "right_tilt_90_ccw_projection_deg": 0.0,
            "right_tilt_90_ccw_skipped_reason": "supplement_pre_already_rotated",
        }
    elif mode != "off":
        rtl_meta = {
            "right_tilt_90_ccw_applied": False,
            "right_tilt_90_ccw_projection_deg": 0.0,
            "right_tilt_90_ccw_skipped_reason": "auto_rotate_quarters_handles_orientation",
        }
    else:
        work, rtl_meta = _maybe_right_tilt_90_ccw_bgr(work)

    if mode == "off":
        if skip_isolation:
            fb_meta, warp_info = _skip_physical_isolation_meta(profile)
            cropped = work
            card_warped = False
        else:
            work, fb_meta = _maybe_full_bleed_axis_straighten(work)
            cropped, card_warped, warp_info = _try_extract_card_warp(work)
            if bool(fb_meta.get("full_bleed_straighten_applied")):
                card_warped = True
        doc_q_meta: dict[str, object] = {}
        wm = str(warp_info.get("card_warp_mode") or "")
        # Hanya bila tidak ada crop perspektif: scan penuh / gagal / warp mati — putar 4-arah.
        # Jangan putar lagi setelah axis_box/perspective (risiko putar ganda / rusak yang sudah tegak).
        # Hanya bila tidak ada crop perspektif — pakai ambang sama seperti putar kuartal utama
        # (margin/delta ketat, bukan 1.05/0.01 yang mudah memutar dokumen yang sudah normal).
        if wm in ("skipped_full_bleed", "failed", "disabled") and not bool(
            fb_meta.get("full_bleed_straighten_applied")
        ):
            cropped, doc_q_meta = _maybe_auto_rotate_quarters(
                cropped,
                card_warped=False,
                margin=None,
                min_delta=None,
                meta_prefix="document_flat_quarter",
            )
        merged = {
            **air_meta,
            **supplement_meta,
            **rtl_meta,
            **fb_meta,
            **warp_info,
            **doc_q_meta,
            **_empty_auto_rotate_meta(reason="PREPROCESS_AUTO_ROTATE_QUARTERS=off"),
        }
    else:
        if skip_isolation:
            fb_meta, warp_info = _skip_physical_isolation_meta(profile)
            cropped = work
            card_warped = False
            pre_meta = _empty_auto_rotate_meta(reason=f"profile_{profile}")
            post_meta = _empty_auto_rotate_meta(reason=f"profile_{profile}")
        else:
            work, fb_meta = _maybe_full_bleed_axis_straighten(work)
            pre_margin, pre_delta = _pre_rotate_margin_delta(mode)
            work, pre_meta = _maybe_auto_rotate_quarters(
                work,
                card_warped=False,
                margin=pre_margin,
                min_delta=pre_delta,
                meta_prefix="auto_rotate_pre",
            )
            cropped, card_warped, warp_info = _try_extract_card_warp(work)
            if bool(fb_meta.get("full_bleed_straighten_applied")):
                card_warped = True
            cropped, post_meta = _maybe_auto_rotate_quarters(
                cropped,
                card_warped=card_warped,
                margin=None,
                min_delta=None,
                meta_prefix="auto_rotate_post",
            )
        pre_applied = bool(pre_meta.get("auto_rotate_pre_applied"))
        post_applied = bool(post_meta.get("auto_rotate_post_applied"))
        merged = {
            **air_meta,
            **supplement_meta,
            **rtl_meta,
            **fb_meta,
            **warp_info,
            **pre_meta,
            **post_meta,
            "auto_rotate_quarters_applied": pre_applied or post_applied,
            "auto_rotate_pre_applied": pre_applied,
            "auto_rotate_post_applied": post_applied,
            "auto_rotate_pre_90ccw_steps": int(pre_meta.get("auto_rotate_pre_90ccw_steps") or 0),
            "auto_rotate_post_90ccw_steps": int(post_meta.get("auto_rotate_post_90ccw_steps") or 0),
            # Header / kompatibilitas: langkah CCW setelah warp (jalur lama).
            "auto_rotate_90ccw_steps": int(post_meta.get("auto_rotate_post_90ccw_steps") or 0),
        }

    cropped, flip180_meta = _maybe_disambiguate_upside_down_bgr(cropped)
    merged = {**merged, **flip180_meta}

    cropped, proj_meta = _maybe_projection_deskew_bgr(cropped)
    merged = {**merged, **proj_meta}

    cropped, micro_meta = _maybe_pre_enhance_micro_deskew_bgr(cropped)
    merged = {**merged, **micro_meta}

    gray_probe = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur_score = _laplacian_blur_score(gray_probe)
    cropped_sr, sr_meta = maybe_apply_realesrgan_bgr(cropped)
    cropped_enh = _maybe_detail_enhance_bgr(cropped_sr, blur_score=blur_score)
    gray = cv2.cvtColor(cropped_enh, cv2.COLOR_BGR2GRAY)
    out = _enhance_grayscale_for_ocr(gray, blur_score_hint=blur_score)
    out, post_meta = _maybe_post_enhance_micro_deskew_gray(out)
    meta = {
        **resize_meta,
        "document_profile_id": profile or None,
        "physical_isolation_skipped": skip_isolation,
        "blur_score_laplacian": round(blur_score, 2),
        "heavy_blur_enhance": blur_score < _BLUR_SCORE_HEAVY,
        "detail_enhance_bgr": blur_score < _DETAIL_ENHANCE_MAX_BLUR_SCORE,
        **merged,
        **sr_meta,
        **post_meta,
    }
    return out, card_warped, meta


def preprocess_for_ocr(bgr: np.ndarray, *, document_profile_id: str = "") -> np.ndarray:
    """
    1) Resize batas.
    2) Deteksi bidang kartu + warp (menghindari tekstur kain/latar foto).
    3) Perkuat grayscale 8-bit untuk OCR (tanpa binarisasi keras).
    """
    out, _, _ = _preprocess_work_bgr(bgr, document_profile_id=document_profile_id)
    return out


def decode_image_bytes_bgr(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode upload bytes ke BGR + metadata EXIF (tanpa preprocess OCR)."""
    return _decode_image_with_exif(data)


def preprocess_image_bytes(
    data: bytes,
    *,
    document_profile_id: str = "",
) -> tuple[bytes, dict]:
    """
    Decode bytes gambar → pipeline OCR → PNG bytes (grayscale 8-bit).

    Langkah opsional Real-ESRGAN (super-res): set PREPROCESS_USE_REALESRGAN=1
    dan pasang requirements-realesrgan.txt — lihat `realesrgan_infer.py`.

    Returns:
        (png_bytes, metadata) metadata berisi dimensi output + info blur / realesrgan.
    """
    bgr, decode_meta = _decode_image_with_exif(data)
    out, card_warped, enhance_meta = _preprocess_work_bgr(
        bgr,
        document_profile_id=document_profile_id,
    )
    ok, encoded = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("Gagal mengenkode hasil ke PNG.")
    meta = {
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
        "channels": 1,
        "encoding": "grayscale_8bit",
        "card_warped": card_warped,
        **decode_meta,
        **enhance_meta,
    }
    return encoded.tobytes(), meta


def preprocess_to_png_buffer(data: bytes) -> io.BytesIO:
    png_bytes, _ = preprocess_image_bytes(data)
    buf = io.BytesIO(png_bytes)
    buf.seek(0)
    return buf
