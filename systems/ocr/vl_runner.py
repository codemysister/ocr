"""Jalankan PaddleOCR-VL-1.6 (satu predict + satu restructure_pages) pada bytes gambar."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
import time
from typing import Any

import cv2
import numpy as np

from systems.ocr.runtime_hint import apply_paddle_device, paddlepaddle_importable, resolve_paddle_device

logger = logging.getLogger(__name__)

_pipeline: Any = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()


def _ktp_max_long_side() -> int | None:
    """KTP VL: jangan pakai OCR_VL_MAX_LONG_SIDE (itu merusak KTP kecil di frame besar)."""
    raw = (os.environ.get("OCR_VL_KTP_MAX_LONG_SIDE") or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _max_long_side_from_env() -> int | None:
    """OCR_VL_MAX_LONG_SIDE: jika diset (px), gambar diperkecil agar sisi terpanjang ≤ nilai ini."""
    raw = (os.environ.get("OCR_VL_MAX_LONG_SIDE") or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _maybe_downscale_bgr(
    bgr: np.ndarray, max_long_side: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Kurangi resolusi sebelum VL untuk menurunkan puncak RAM/waktu (trade-off kualitas)."""
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    meta: dict[str, Any] = {
        "resized": False,
        "input_hw": {"height": h, "width": w},
    }
    if not max_long_side or max_long_side <= 0:
        return bgr, meta
    long_side = max(h, w)
    if long_side <= max_long_side:
        return bgr, meta
    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    meta.update(
        resized=True,
        scale_rounded=round(scale, 6),
        max_long_side=max_long_side,
        final_hw={"height": new_h, "width": new_w},
    )
    return resized, meta


def _decode_bgr(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Gunakan JPEG, PNG, WebP, atau format umum lainnya.")
    return img


def _plain_text_from_json_res(res: dict[str, Any]) -> str:
    blocks = res.get("parsing_res_list") or []

    def sort_key(b: dict[str, Any]) -> tuple[int, int]:
        o = b.get("block_order")
        if o is None:
            return (1, b.get("block_id", 0))
        return (0, int(o))

    lines: list[str] = []
    for b in sorted(blocks, key=sort_key):
        c = b.get("block_content")
        if isinstance(c, str) and c.strip():
            lines.append(c.strip())
    return "\n\n".join(lines)


def _pipeline_version() -> str:
    raw = (os.environ.get("OCR_VL_PIPELINE_VERSION") or "v1.6").strip()
    return raw if raw in ("v1", "v1.5", "v1.6") else "v1.6"


def _model_label(version: str) -> str:
    if version == "v1":
        return "PaddleOCR-VL"
    if version == "v1.5":
        return "PaddleOCR-VL-1.5"
    return "PaddleOCR-VL-1.6"


def _paddleocr_vl_importable() -> bool:
    if not paddlepaddle_importable():
        return False
    try:
        from paddleocr import PaddleOCRVL  # noqa: F401
    except ImportError:
        return False
    return True


def get_vl_pipeline():
    """Singleton PaddleOCR-VL (lazy, default v1.6)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _init_lock:
        if _pipeline is not None:
            return _pipeline
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as e:
            raise RuntimeError(
                'Pasang dependensi OCR: pip install "paddleocr[doc-parser]>=3.5.0"'
            ) from e

        backend = os.environ.get("OCR_VL_BACKEND")
        server_url = os.environ.get("OCR_VL_SERVER_URL")
        version = _pipeline_version()
        from systems.ocr.fast_runner import paddle_mkldnn_enabled

        device = apply_paddle_device()
        kw: dict[str, Any] = {
            "pipeline_version": version,
            "use_queues": False,
            # oneDNN rusak di PaddlePaddle 3.x + PIR (lihat fast_runner). Diteruskan ke sub-predictor.
            "enable_mkldnn": paddle_mkldnn_enabled(),
            "device": device,
        }
        if backend:
            kw["vl_rec_backend"] = backend
        if server_url:
            kw["vl_rec_server_url"] = server_url

        try:
            _pipeline = PaddleOCRVL(**kw)
        except TypeError as e:
            err = str(e)
            if "is_bfloat16_supported" not in err and "undefined" not in err.casefold():
                raise
            logger.warning("PaddleOCR-VL gagal di %s (%s); coba CPU", device, e)
            apply_paddle_device("cpu")
            kw["device"] = "cpu"
            _pipeline = PaddleOCRVL(**kw)
        return _pipeline


def run_paddleocr_vl(image_bytes: bytes, *, include_full_json: bool = False) -> dict[str, Any]:
    """
    Decode gambar → VL predict → restructure_pages (parsing layout sekali per permintaan).

    Membutuhkan paddlepaddle untuk backend native (layout + VL), kecuali Anda
    mengonfigurasi backend server lewat OCR_VL_BACKEND / OCR_VL_SERVER_URL.
    """
    t_wall0 = time.perf_counter()
    timing: dict[str, Any] = {}

    t0 = time.perf_counter()
    bgr = _decode_bgr(image_bytes)
    timing["decode_image_s"] = round(time.perf_counter() - t0, 3)
    max_side = _max_long_side_from_env()
    t0r = time.perf_counter()
    bgr, resize_meta = _maybe_downscale_bgr(bgr, max_side)
    timing["resize_s"] = round(time.perf_counter() - t0r, 3)
    timing["resize"] = resize_meta
    ih, iw = int(bgr.shape[0]), int(bgr.shape[1])
    timing["vl_input_hw"] = {"height": ih, "width": iw}
    orig = resize_meta.get("input_hw") or {"height": ih, "width": iw}
    timing["input_hw"] = orig
    timing["paddle_device"] = resolve_paddle_device()

    t1 = time.perf_counter()
    pipe = get_vl_pipeline()
    timing["get_pipeline_s"] = round(time.perf_counter() - t1, 3)
    # Pemanggilan pertama: get_pipeline_s memuat layout + VL (bisa menit); berikutnya ~0 s.

    with _infer_lock:
        t2 = time.perf_counter()
        raw_pages = pipe.predict(bgr)
        timing["predict_s"] = round(time.perf_counter() - t2, 3)
        if not raw_pages:
            raise RuntimeError("PaddleOCR-VL tidak mengembalikan hasil.")

        t3 = time.perf_counter()
        parsed_pages = pipe.restructure_pages(
            raw_pages,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False,
        )
        timing["restructure_pages_s"] = round(time.perf_counter() - t3, 3)

    t4 = time.perf_counter()
    page = parsed_pages[0]
    md = page.markdown
    markdown_text = md.get("markdown_texts") or ""

    json_inner = page.json.get("res") or {}
    plain = _plain_text_from_json_res(json_inner if isinstance(json_inner, dict) else {})
    timing["build_text_markdown_s"] = round(time.perf_counter() - t4, 3)
    timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)

    logger.info(
        "ocr_timing total=%.3fs predict=%.3fs restructure=%.3fs get_pipeline=%.3fs decode=%.3fs hw=%dx%d",
        timing["total_wall_s"],
        timing["predict_s"],
        timing["restructure_pages_s"],
        timing["get_pipeline_s"],
        timing["decode_image_s"],
        iw,
        ih,
    )

    version = _pipeline_version()
    payload: dict[str, Any] = {
        "success": True,
        "model": _model_label(version),
        "pipeline_version": version,
        "markdown": markdown_text,
        "text": plain,
        "timing": timing,
        "paddle_device": resolve_paddle_device(),
    }
    if include_full_json:
        payload["result_json"] = page.json
    return payload


KTP_EXTRACT_PROMPT = """You are an Indonesian identity document extraction system.

Extract the following fields from the KTP image:

- nik
- nama
- status_perkawinan

Rules:

1. NIK must contain exactly 16 numeric digits.
2. Do not invent missing characters.
3. Preserve the name exactly as written.
4. status_perkawinan must be one of:
   BELUM KAWIN
   KAWIN
   CERAI HIDUP
   CERAI MATI
5. If a value cannot be read confidently, return null.
6. Return JSON only.

Output:

{
  "nik": null,
  "nama": null,
  "status_perkawinan": null,
  "confidence": {
    "nik": 0.0,
    "nama": 0.0,
    "status_perkawinan": 0.0
  }
}"""

_KTP_STATUS = ("BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI")
_NEXT_KTP_FIELD = re.compile(
    r"\b("
    r"Tempat(?:\s*/\s*Tgl)?(?:\s+Lahir)?"
    r"|Jenis\s*kelamin"
    r"|Gol(?:ongan)?\.?\s*Darah"
    r"|Alamat"
    r"|RT\s*/\s*RW"
    r"|Kel(?:urahan)?\s*/\s*Desa"
    r"|Kel/Dosa"
    r"|Kol/Dosa"
    r"|Kecamatan"
    r"|Agama"
    r"|Status\s*Perkawinan\w*"
    r"|Pekerjaan"
    r"|Pokerjaan"
    r"|Kewarganegaraan"
    r"|Kewarganegarian"
    r"|Berlaku"
    r"|NIK"
    r")\b",
    re.I,
)


def _vl_recognition_model(pipe: Any) -> Any:
    """Model VL yang menerima query bebas. Pipeline publik hanya mengizinkan prompt_label bawaan."""
    inner = getattr(pipe, "paddlex_pipeline", None)
    model = getattr(inner, "vl_rec_model", None) if inner is not None else None
    if model is None or not hasattr(model, "predict"):
        raise RuntimeError(
            "PaddleOCR-VL di versi ini tidak membuka model pengenalan untuk prompt kustom."
        )
    return model


def _predict_prompt(model: Any, image: np.ndarray, prompt: str) -> Any:
    sample = [{"image": image, "query": prompt}]
    attempts: list[dict[str, Any]] = [
        {"skip_special_tokens": True, "max_new_tokens": 512, "temperature": 0},
        {"skip_special_tokens": True, "max_new_tokens": 512},
        {},
    ]
    last: Exception | None = None
    for kw in attempts:
        try:
            rows = list(model.predict(sample, **kw))
        except TypeError as e:
            last = e
            continue
        if not rows:
            raise RuntimeError("PaddleOCR-VL tidak mengembalikan hasil.")
        return rows[0]
    raise RuntimeError(f"Prompt kustom tidak diterima model VL: {last}")


def _result_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("result", "text", "output"):
            val = item.get(key)
            if isinstance(val, str):
                return val
        return json.dumps(item, ensure_ascii=False)
    for attr in ("result", "text"):
        val = getattr(item, attr, None)
        if isinstance(val, str):
            return val
    return str(item)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S | re.I)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < 0.0 or number > 1.0:
        return 0.0
    return round(number, 4)


def normalize_ktp_extraction(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Paksa skema KTP: NIK 16 digit, status dari daftar, null jika tidak yakin."""
    data = parsed if isinstance(parsed, dict) else {}
    conf_in = data.get("confidence")
    conf_in = conf_in if isinstance(conf_in, dict) else {}

    nik: str | None = None
    nik_raw = data.get("nik")
    if isinstance(nik_raw, str):
        compact = re.sub(r"\s+", "", nik_raw.strip())
        if re.fullmatch(r"\d{16}", compact):
            nik = compact
    elif isinstance(nik_raw, int) and not isinstance(nik_raw, bool):
        digits = str(nik_raw)
        if re.fullmatch(r"\d{16}", digits):
            nik = digits

    nama: str | None = None
    nama_raw = data.get("nama")
    if isinstance(nama_raw, str) and nama_raw.strip():
        nama = nama_raw.strip()

    status: str | None = None
    status_raw = data.get("status_perkawinan")
    if isinstance(status_raw, str):
        candidate = re.sub(r"\s+", " ", status_raw.strip()).upper()
        if candidate in _KTP_STATUS:
            status = candidate

    return {
        "nik": nik,
        "nama": nama,
        "status_perkawinan": status,
        "confidence": {
            "nik": _confidence(conf_in.get("nik")) if nik else 0.0,
            "nama": _confidence(conf_in.get("nama")) if nama else 0.0,
            "status_perkawinan": _confidence(conf_in.get("status_perkawinan")) if status else 0.0,
        },
    }


def _flatten_ktp_ocr_text(text: str) -> str:
    blob = (text or "").replace("\r\n", "\n")
    blob = blob.replace("\\n", "\n").replace("\\t", " ")
    blob = html.unescape(blob)
    blob = re.sub(r"(?i)<br\s*/?>", "\n", blob)
    blob = re.sub(r"(?i)</\s*(tr|p|div|h[1-6]|li|table)\s*>", "\n", blob)
    blob = re.sub(r"(?i)</\s*td\s*>", "\n", blob)
    blob = re.sub(r"<[^>]+>", " ", blob)
    lines = []
    for line in blob.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip(" |")
        if line:
            lines.append(line)
    return "\n".join(lines)


def _looks_like_person_name(value: str) -> bool:
    candidate = re.sub(r"^#+\s*", "", value or "").strip()
    if not candidate:
        return False
    if re.search(
        r"PROVINSI|KABUPATEN|KOTA\s|KALIMANTAN|SUMATERA|JAWA\s|NUSANTARA|REPUBLIK",
        candidate,
        re.I,
    ):
        return False
    if re.search(r"\d{5,}", candidate):
        return False
    letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", candidate)
    if len(letters) < 4 or len(letters) > 60:
        return False
    if len(candidate.split()) > 8:
        return False
    return True


def _name_from_line(value: str) -> str | None:
    candidate = _NEXT_KTP_FIELD.split(value, maxsplit=1)[0]
    candidate = re.sub(r"^#+\s*", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .:-\\")
    if not _looks_like_person_name(candidate):
        return None
    return candidate


def _match_marital_status(raw: str) -> tuple[str | None, float]:
    blob = re.sub(r"[^A-Za-z]+", " ", raw or "").strip().upper()
    if not blob:
        return None, 0.0
    compact = re.sub(r"\s+", " ", blob)
    for item in sorted(_KTP_STATUS, key=len, reverse=True):
        if compact == item or compact.startswith(item + " "):
            return item, 0.9
    try:
        from rapidfuzz import fuzz
    except ImportError:
        if "BELUM" in compact and "KAWIN" in compact:
            return "BELUM KAWIN", 0.6
        if "CERAI" in compact and "HIDUP" in compact:
            return "CERAI HIDUP", 0.6
        if "CERAI" in compact and "MATI" in compact:
            return "CERAI MATI", 0.6
        return None, 0.0
    best_item = None
    best_score = 0.0
    for item in _KTP_STATUS:
        score = float(fuzz.ratio(compact, item))
        if score > best_score:
            best_score = score
            best_item = item
    if best_item and best_score >= 70:
        conf = 0.8 if best_score >= 88 else 0.65
        return best_item, conf
    return None, 0.0


def _sixteen_digit_runs(text: str) -> list[str]:
    """Urutan 16 digit yang sudah tertulis; tidak menggabungkan angka antar baris."""
    runs: list[str] = []
    for m in re.finditer(r"\d[\d \t\-]{14,}\d", text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= 16:
            runs.append(digits[:16])
    return list(dict.fromkeys(runs))


def _nik_after_label(lines: list[str]) -> str | None:
    for i, line in enumerate(lines):
        if not re.match(r"^(?:#+\s*)?NIK\b", line, re.I):
            continue
        same = re.sub(r"\D", "", line)
        if re.fullmatch(r"\d{16}", same):
            return same
        window = "\n".join(lines[i : i + 6])
        runs = _sixteen_digit_runs(window)
        if runs:
            return runs[0]
    return None


def extract_ktp_fields_from_ocr_text(text: str) -> dict[str, Any]:
    """Terapkan aturan prompt KTP pada teks OCR (tanpa mengarang karakter)."""
    blob = _flatten_ktp_ocr_text(text)
    lines = [re.sub(r"^#+\s*", "", ln).strip() for ln in blob.splitlines() if ln.strip()]

    nik: str | None = None
    nik_conf = 0.0
    labeled = _nik_after_label(lines)
    if labeled:
        nik, nik_conf = labeled, 0.9
    if nik is None:
        runs = _sixteen_digit_runs(blob)
        if len(runs) == 1:
            nik, nik_conf = runs[0], 0.7

    nama: str | None = None
    nama_conf = 0.0
    label_name = re.compile(
        r"^(Nama|Name|Nema|Nicht|Namaa)\s*[:.\-]?\s*(.*)$",
        re.I,
    )
    for i, line in enumerate(lines):
        m = label_name.match(line)
        if not m:
            continue
        inline = _name_from_line(m.group(2) or "")
        if inline:
            nama, nama_conf = inline, 0.85
            break
        if i + 1 < len(lines):
            nxt = _name_from_line(lines[i + 1])
            if nxt:
                nama, nama_conf = nxt, 0.8
                break

    status: str | None = None
    status_conf = 0.0
    labeled_status = re.search(
        r"Status\s*Perkawinan\w*\s*[:.\-]?\s*([A-Za-z ]{3,40})",
        blob,
        re.I,
    )
    if labeled_status:
        status, status_conf = _match_marital_status(labeled_status.group(1))
    if status is None:
        for line in lines:
            if re.search(r"kawin|cerai", line, re.I):
                status, status_conf = _match_marital_status(line)
                if status:
                    break

    return {
        "nik": nik,
        "nama": nama,
        "status_perkawinan": status,
        "confidence": {
            "nik": nik_conf,
            "nama": nama_conf,
            "status_perkawinan": status_conf,
        },
    }


def _ocr_payload_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("markdown", "text"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return "\n".join(parts)


def _ktp_from_remote_ocr(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(payload.get("nik"), str) or payload.get("status_perkawinan"):
        extracted = normalize_ktp_extraction(payload)
        raw = payload.get("raw_model_text")
        if not isinstance(raw, str):
            raw = _ocr_payload_text(payload)
        return extracted, raw
    raw = _ocr_payload_text(payload)
    extracted = normalize_ktp_extraction(extract_ktp_fields_from_ocr_text(raw))
    return extracted, raw


def _jpeg_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Gagal menyiapkan gambar untuk inferensi VL.")
    return buf.tobytes()


def _ktp_payload(
    extracted: dict[str, Any],
    *,
    timing: dict[str, Any],
    raw_text: str,
    backend: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = _pipeline_version()
    out: dict[str, Any] = {
        **extracted,
        "model": _model_label(version),
        "pipeline_version": version,
        "backend": backend,
        "paddle_device": resolve_paddle_device() if paddlepaddle_importable() else None,
        "timing": timing,
        "raw_model_text": raw_text,
    }
    if extra:
        out.update(extra)
    return out


def _ktp_via_openai(
    image_bytes: bytes,
    *,
    timing: dict[str, Any],
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[dict[str, Any], str]:
    from systems.validation.llm_fallback import complete_vision

    t0 = time.perf_counter()
    content, meta = complete_vision(
        image_bytes,
        KTP_EXTRACT_PROMPT,
        image_mime="image/jpeg",
        max_tokens=512,
        temperature=0.0,
        base_url=base_url,
        model=model,
    )
    timing["predict_s"] = round(time.perf_counter() - t0, 3)
    timing["vision_meta"] = {
        "base_url": meta.get("base_url"),
        "model": meta.get("model"),
        "latency_s": meta.get("latency_s"),
        "error": meta.get("error"),
    }
    if not content:
        raise RuntimeError(meta.get("error") or "Endpoint vision tidak mengembalikan teks.")
    extracted = normalize_ktp_extraction(_extract_json_object(content))
    return extracted, content


def _ktp_via_native(
    bgr: np.ndarray, timing: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """PaddleOCR-VL-1.6: layout + OCR, lalu aturan field KTP pada teks hasil."""
    t1 = time.perf_counter()
    pipe = get_vl_pipeline()
    timing["get_pipeline_s"] = round(time.perf_counter() - t1, 3)
    with _infer_lock:
        t2 = time.perf_counter()
        raw_pages = pipe.predict(bgr)
        timing["predict_s"] = round(time.perf_counter() - t2, 3)
        if not raw_pages:
            raise RuntimeError("PaddleOCR-VL tidak mengembalikan hasil.")
        t3 = time.perf_counter()
        parsed_pages = pipe.restructure_pages(
            raw_pages,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False,
        )
        timing["restructure_pages_s"] = round(time.perf_counter() - t3, 3)
    page = parsed_pages[0]
    markdown_text = (page.markdown or {}).get("markdown_texts") or ""
    json_inner = page.json.get("res") or {}
    plain = _plain_text_from_json_res(json_inner if isinstance(json_inner, dict) else {})
    raw_text = "\n".join(part for part in (markdown_text, plain) if part)
    extracted = normalize_ktp_extraction(extract_ktp_fields_from_ocr_text(raw_text))
    return extracted, raw_text, {"vl_markdown": markdown_text, "vl_text": plain}


def run_paddleocr_vl_ktp(image_bytes: bytes) -> dict[str, Any]:
    """Satu gambar KTP + prompt ekstraksi → JSON nik, nama, status_perkawinan.

    Tidak memakai deteksi layout. Urutan: Paddle native (jika ada) → OCR_VL_SERVER_URL
    → OCR remote `/ocr-vl-ktp` → LLM vision fallback.
    """
    from systems.ocr.remote_client import (
        ocr_remote_enabled,
        run_remote_ocr_fast,
        run_remote_ocr_vl,
        run_remote_ocr_vl_ktp,
    )

    t_wall0 = time.perf_counter()
    timing: dict[str, Any] = {}
    errors: list[str] = []

    t0 = time.perf_counter()
    bgr = _decode_bgr(image_bytes)
    timing["decode_image_s"] = round(time.perf_counter() - t0, 3)
    max_side = _ktp_max_long_side()
    t0r = time.perf_counter()
    bgr, resize_meta = _maybe_downscale_bgr(bgr, max_side)
    jpeg = _jpeg_bytes(bgr)
    timing["resize_s"] = round(time.perf_counter() - t0r, 3)
    timing["resize"] = resize_meta
    timing["paddle_device"] = resolve_paddle_device() if paddlepaddle_importable() else None

    if _paddleocr_vl_importable():
        try:
            extracted, raw_text, extra = _ktp_via_native(bgr, timing)
            timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)
            return _ktp_payload(
                extracted,
                timing=timing,
                raw_text=raw_text,
                backend="paddleocr-vl-native",
                extra=extra,
            )
        except Exception as e:
            logger.warning("KTP native PaddleOCR-VL gagal: %s", e)
            errors.append(f"native: {e}")

    vl_server = (os.environ.get("OCR_VL_SERVER_URL") or "").strip().rstrip("/")
    vl_model = (os.environ.get("OCR_VL_API_MODEL_NAME") or os.environ.get("OCR_VL_MODEL") or "").strip()
    if vl_server:
        try:
            extracted, raw_text = _ktp_via_openai(
                jpeg,
                timing=timing,
                base_url=vl_server,
                model=vl_model or "PaddlePaddle/PaddleOCR-VL",
            )
            timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)
            return _ktp_payload(
                extracted,
                timing=timing,
                raw_text=raw_text,
                backend="ocr-vl-server",
                extra={"vl_server_url": vl_server},
            )
        except Exception as e:
            logger.warning("KTP OCR_VL_SERVER_URL gagal: %s", e)
            errors.append(f"ocr_vl_server: {e}")

    if ocr_remote_enabled():
        try:
            remote = run_remote_ocr_vl_ktp(jpeg, filename="ktp.jpg")
            extracted, raw_text = _ktp_from_remote_ocr(remote)
            timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)
            timing["ocr_remote"] = remote.get("timing")
            return _ktp_payload(
                extracted,
                timing=timing,
                raw_text=raw_text,
                backend=str(remote.get("backend") or "ocr-remote-ktp"),
            )
        except Exception as e:
            logger.warning("KTP OCR remote /ocr-vl-ktp gagal: %s", e)
            errors.append(f"ocr_remote_ktp: {e}")
        try:
            remote = run_remote_ocr_fast(jpeg, filename="ktp.jpg")
            extracted, raw_text = _ktp_from_remote_ocr(remote)
            timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)
            timing["ocr_remote"] = remote.get("timing")
            return _ktp_payload(
                extracted,
                timing=timing,
                raw_text=raw_text,
                backend="ocr-remote-fast+parse",
                extra={"ocr_remote_model": remote.get("model")},
            )
        except Exception as e:
            logger.warning("KTP OCR remote /ocr-fast gagal: %s", e)
            errors.append(f"ocr_remote_fast: {e}")
        try:
            remote = run_remote_ocr_vl(jpeg, filename="ktp.jpg")
            extracted, raw_text = _ktp_from_remote_ocr(remote)
            timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)
            timing["ocr_remote"] = remote.get("timing")
            return _ktp_payload(
                extracted,
                timing=timing,
                raw_text=raw_text,
                backend="ocr-remote-vl+parse",
                extra={"ocr_remote_model": remote.get("model")},
            )
        except Exception as e:
            logger.warning("KTP OCR remote /ocr gagal: %s", e)
            errors.append(f"ocr_remote_vl: {e}")

    try:
        extracted, raw_text = _ktp_via_openai(jpeg, timing=timing)
        timing["total_wall_s"] = round(time.perf_counter() - t_wall0, 3)
        return _ktp_payload(
            extracted,
            timing=timing,
            raw_text=raw_text,
            backend="llm-vision",
        )
    except Exception as e:
        logger.warning("KTP LLM vision gagal: %s", e)
        errors.append(f"llm_vision: {e}")

    detail = " | ".join(errors) if errors else "tidak ada backend VL"
    raise RuntimeError(
        "Ekstraksi KTP tidak bisa dijalankan: PaddlePaddle tidak tersedia di mesin ini, "
        "dan backend cadangan (OCR_VL_SERVER_URL / OCR_REMOTE_BASE_URL / LLM_FALLBACK) gagal. "
        + detail
    )
