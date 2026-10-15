"""Tes OCR AI: baca surat keterangan kesehatan dari gambar + validasi terstruktur."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.request
from datetime import date
from typing import Any
from urllib.parse import urlparse

from systems.validation.keterangan_kesehatan import evaluate_keterangan_kesehatan_gates
from systems.validation.llm_fallback import (
    _extract_json_object,
    complete_vision,
)
from systems.validation.skck_validity import format_id_date

_DEFAULT_BASE_URL = "http://172.21.13.26:8080/v1"
_DEFAULT_MODEL = "ornith-1.5-9b"

HEALTH_LETTER_SYSTEM_PROMPT = """Kamu asisten validasi dokumen kesehatan Indonesia.
Tugas: baca foto surat keterangan kesehatan / surat keterangan dokter / surat keterangan sehat / hasil MCU.
Jangan mengarang data yang tidak terlihat di gambar. Jika tidak terbaca, isi null atau string kosong.

Kembalikan HANYA satu objek JSON (tanpa markdown) dengan kunci persis:
- document_type_label (string): mis. "Surat Keterangan Kesehatan"
- is_health_certificate (boolean): true hanya jika ini surat keterangan sehat/dokter/MCU, bukan vaksin/KTP
- holder_name (string): nama pasien/yang diperiksa
- identity_number (string): NIK jika ada (hanya digit), else ""
- issuer_type (string): salah satu "rumah_sakit", "klinik", "puskesmas", "bidan", "lain".
  "bidan" jika surat dari bidan atau praktik bidan, meskipun ada kata klinik.
  "lain" jika penerbit tidak terbaca.
- clinic_name (string): nama rumah sakit, klinik, atau puskesmas
- city (string): kota/tempat surat diterbitkan
- doctor_name (string): nama dokter pemeriksa
- health_conclusion (string): salah satu "sehat", "tidak_sehat", "tidak_jelas"
- health_conclusion_text (string): kutipan kesimpulan dari surat
- issued_date (string|null): tanggal dibuat/tanggal pemeriksaan dalam YYYY-MM-DD. null jika tidak terbaca
- issued_date_raw (string): tulisannya persis di surat (mis. "Bekasi, 15 Juni 2026")
- valid_until (string|null): tanggal kadaluarsa YYYY-MM-DD jika tertulis, else null
- validity_duration_text (string): mis. "berlaku 3 bulan" jika ada
- confidence (number 0-100)
- notes (string): alasan singkat bahasa Indonesia, maksimal 12 kata
"""


def _quote_bare_json_value(line: str) -> str:
    matched = re.match(r'^(\s*"[^"]+"\s*:\s*)(.*)$', line)
    if not matched:
        return line
    prefix, rest = matched.group(1), matched.group(2).rstrip()
    comma = ""
    body = rest
    if body.endswith(","):
        comma = ","
        body = body[:-1].rstrip()
    if not body or body[0] in '"[{' or body in {"true", "false", "null"}:
        return line
    if re.fullmatch(r"-?\d+(?:\.\d+)?", body):
        if len(body) > 1 and body[0] == "0":
            return f'{prefix}"{body}"{comma}'
        return line
    body = body.replace("\\", "\\\\").replace('"', '\\"')
    return f'{prefix}"{body}"{comma}'


def _loose_json_object(text: str) -> dict[str, Any] | None:
    """Terima jawaban model yang menyerupai JSON: bullet, kunci tanpa kutip, atau tanpa kurung."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    raw = re.sub(r"^json\s*", "", raw, flags=re.I)
    raw = re.sub(r"(?m)^\s*-\s*", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        chunk = raw[start : end + 1]
    elif ":" not in raw:
        return None
    else:
        chunk = "{\n" + raw + "\n}"
    lines = []
    for line in chunk.splitlines():
        line = re.sub(r'^(\s*)([A-Za-z_][\w]*)\s*:', r'\1"\2":', line)
        lines.append(_quote_bare_json_value(line))
    joined: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        nxt = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if stripped and stripped not in {"{", "}", "[", "]"} and nxt and nxt not in {"{", "}", "[", "]"}:
            if not stripped.endswith((",", "{", "[")):
                line = line.rstrip() + ","
        joined.append(line)
    chunk = re.sub(r",\s*([}\]])", r"\1", "\n".join(joined))
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key).casefold(): value for key, value in parsed.items()}


def _strip_closed_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()


def _parse_health_json(text: str) -> dict[str, Any] | None:
    text = _strip_closed_think(text)
    obj = _extract_json_object(text)
    if obj:
        return obj
    loose = _loose_json_object(text)
    if loose:
        return loose
    raw = (text or "").strip()
    start = raw.find("{")
    if start < 0:
        return None
    chunk = raw[start:]
    if chunk.count('"') % 2 == 1:
        chunk += '"'
    chunk = re.sub(r",\s*$", "", chunk.rstrip())
    for suffix in ("}", "]}", '"}'):
        try:
            parsed = json.loads(chunk + suffix)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def health_letter_user_prompt(*, expected_name: str = "", expected_nik: str = "") -> str:
    lines = [
        "Baca gambar surat keterangan kesehatan ini.",
        "Surat harus dari rumah sakit, klinik, atau puskesmas. Surat bidan tidak lolos.",
        "Ambil nama, kesimpulan sehat/tidak, tanggal pemeriksaan atau tanggal surat, "
        "dan masa berlaku jika ada.",
        "Validasi: penerbit, nama, tertulis SEHAT, dan tanggal pemeriksaan terbaca.",
    ]
    if expected_name.strip():
        lines.append(f"Nama referensi pemegang: {expected_name.strip()}")
    if expected_nik.strip():
        lines.append(f"NIK referensi: {expected_nik.strip()}")
    lines.append("Kembalikan JSON saja.")
    return "\n".join(lines)


def health_ai_base_url(override: str | None = None) -> str:
    raw = (override or os.environ.get("LLM_HEALTH_AI_BASE_URL") or _DEFAULT_BASE_URL).strip()
    return raw.rstrip("/")


def health_ai_model(override: str | None = None) -> str:
    return (override or os.environ.get("LLM_HEALTH_AI_MODEL") or _DEFAULT_MODEL).strip()


def _host_is_private(host: str) -> bool:
    host = (host or "").strip().strip("[]")
    if host in {"localhost"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (OSError, ValueError):
            return False
    return bool(ip.is_private or ip.is_loopback)


def assert_allowed_llm_url(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL LLM harus http atau https.")
    host = parsed.hostname or ""
    allow_any = (os.environ.get("LLM_HEALTH_AI_ALLOW_ANY_URL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_any and not _host_is_private(host):
        raise ValueError(
            f"URL LLM '{host}' ditolak. Hanya jaringan lokal (172/10/192.168/localhost)."
        )
    path = (parsed.path or "").rstrip("/")
    if path in {"", "/"}:
        return f"{parsed.scheme}://{parsed.netloc}/v1".rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")


def list_health_ai_models(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "-"}:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if not m:
        return None
    try:
        date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _synthetic_ocr(parsed: dict[str, Any]) -> str:
    conclusion = str(parsed.get("health_conclusion") or "").strip().casefold()
    if conclusion == "sehat":
        sehat_line = "Berdasarkan hasil pemeriksaan, dinyatakan SEHAT."
    elif conclusion == "tidak_sehat":
        sehat_line = "Dinyatakan tidak sehat."
    else:
        sehat_line = str(parsed.get("health_conclusion_text") or "").strip()
    city = str(parsed.get("city") or "").strip()
    issued_raw = str(parsed.get("issued_date_raw") or "").strip()
    issued_iso = _iso_date(parsed.get("issued_date"))
    place_line = issued_raw
    if not place_line and (city or issued_iso):
        place_line = ", ".join(p for p in (city, issued_iso or "") if p)
    until = _iso_date(parsed.get("valid_until"))
    duration = str(parsed.get("validity_duration_text") or "").strip()
    until_line = ""
    if duration:
        until_line = duration
        if until:
            until_line += f" yaitu sampai dengan tanggal {until}"
    elif until:
        until_line = f"berlaku sampai dengan tanggal {until}"
    title = "SURAT KETERANGAN KESEHATAN"
    if parsed.get("is_health_certificate") is False:
        title = str(parsed.get("document_type_label") or "DOKUMEN").strip() or "DOKUMEN"
    parts = [
        title,
        str(parsed.get("clinic_name") or "").strip(),
        f"Nama: {parsed.get('holder_name') or ''}",
        sehat_line,
        until_line,
        place_line,
        f"Dokter: {parsed.get('doctor_name') or ''}",
        "Dokter pemeriksa",
    ]
    return "\n".join(p for p in parts if p)


def _issuer_type(parsed: dict[str, Any]) -> str:
    explicit = str(parsed.get("issuer_type") or "").strip().casefold().replace(" ", "_")
    blob = " ".join(
        str(parsed.get(key) or "")
        for key in ("clinic_name", "document_type_label", "doctor_name", "notes", "issuer_type")
    )
    compact = re.sub(r"[^a-z]", "", blob.casefold())
    if "bidan" in compact or explicit == "bidan":
        return "bidan"
    if "puskesmas" in compact or explicit == "puskesmas":
        return "puskesmas"
    if any(mark in compact for mark in ("rumahsakit", "rsud", "rsu", "hospital")) or explicit in {
        "rumah_sakit",
        "rs",
    }:
        return "rumah_sakit"
    if "klinik" in compact or "praktikdokter" in compact or explicit == "klinik":
        return "klinik"
    if explicit in {"rumah_sakit", "klinik", "puskesmas"}:
        return explicit
    return "lain"


def _normalize_conclusion(value: Any) -> str:
    raw = str(value or "").strip().casefold().replace(" ", "_")
    if raw in {"sehat", "healthy", "fit"}:
        return "sehat"
    if raw in {"tidak_sehat", "tidak-sehat", "unfit", "sakit"}:
        return "tidak_sehat"
    return "tidak_jelas"


def analyze_health_letter_image(
    image_bytes: bytes,
    *,
    image_mime: str = "image/jpeg",
    expected_name: str = "",
    expected_nik: str = "",
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    base_url = assert_allowed_llm_url(health_ai_base_url(llm_base_url))
    model = health_ai_model(llm_model)
    prompt = (user_prompt or "").strip() or health_letter_user_prompt(
        expected_name=expected_name,
        expected_nik=expected_nik,
    )
    sys_prompt = (system_prompt or "").strip() or HEALTH_LETTER_SYSTEM_PROMPT

    from systems.preprocessing.pipeline import encode_image_bytes_for_ocr_remote_safe

    prepared, prep_meta = encode_image_bytes_for_ocr_remote_safe(
        image_bytes, max_long_side=1280
    )
    thinking_on = (os.environ.get("LLM_HEALTH_AI_ENABLE_THINKING") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    extra: dict[str, Any] = {}
    if not thinking_on:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    content, meta = complete_vision(
        prepared,
        prompt,
        image_mime="image/jpeg",
        system_text=sys_prompt,
        max_tokens=int(os.environ.get("LLM_HEALTH_AI_MAX_TOKENS", "1536")),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120")),
        extra=extra or None,
    )
    out: dict[str, Any] = {
        "llm": {
            "base_url": base_url,
            "model": model,
            "latency_s": meta.get("latency_s"),
            "success": meta.get("success"),
            "error": meta.get("error"),
            "attempted": meta.get("attempted"),
        },
        "prompt": {"system": sys_prompt, "user": prompt},
        "image_prepare": prep_meta,
        "raw_content": content,
        "extraction": None,
        "validation": None,
        "document_matched": False,
    }
    if not content:
        out["error"] = meta.get("error") or "empty_content"
        return out

    parsed = _parse_health_json(content)
    if not parsed:
        out["error"] = "invalid_json"
        return out

    parsed["issuer_type"] = _issuer_type(parsed)
    parsed["health_conclusion"] = _normalize_conclusion(parsed.get("health_conclusion"))
    parsed["issued_date"] = _iso_date(parsed.get("issued_date"))
    parsed["valid_until"] = _iso_date(parsed.get("valid_until"))
    parsed["issued_date_display"] = format_id_date(parsed.get("issued_date"))
    out["extraction"] = parsed

    synthetic = _synthetic_ocr(parsed)
    gate = evaluate_keterangan_kesehatan_gates(
        synthetic,
        profile_id="keterangan_kesehatan",
        today=today,
    ) or {}
    if parsed.get("issued_date") and not gate.get("exam_date"):
        gate["exam_date"] = parsed["issued_date"]
    if parsed.get("valid_until"):
        gate["valid_until"] = parsed["valid_until"]

    is_health = bool(parsed.get("is_health_certificate"))
    conclusion = parsed.get("health_conclusion")
    health_status_pass = conclusion == "sehat"
    if conclusion == "tidak_sehat":
        health_status_pass = False
    elif conclusion == "tidak_jelas":
        health_status_pass = bool(gate.get("health_status_pass"))

    identity_pass = None
    identity = None
    if expected_name.strip():
        from systems.validation.fuzzy_compare import compare_extracted_identity_scores

        holder = str(parsed.get("holder_name") or "")
        identity = compare_extracted_identity_scores(holder, expected_name)
        identity_pass = float(identity.get("identity_combined_score") or 0.0) >= 65.0 and bool(
            identity.get("token_gate_pass", True)
        )
        exp_nik = re.sub(r"\D", "", expected_nik or "")
        got_nik = re.sub(r"\D", "", str(parsed.get("identity_number") or ""))
        if len(exp_nik) == 16 and got_nik == exp_nik:
            identity_pass = True

    exam = gate.get("exam_date") or parsed.get("issued_date")
    until = gate.get("valid_until") or parsed.get("valid_until")
    as_of = today or date.today()
    expiry_pass = False
    if exam:
        expiry_pass = True
        try:
            exam_day = date.fromisoformat(str(exam)[:10])
        except ValueError:
            exam_day = None
        if exam_day and exam_day > as_of:
            expiry_pass = False
        if until:
            try:
                expiry_pass = date.fromisoformat(str(until)[:10]) >= as_of
            except ValueError:
                expiry_pass = True

    issuer = str(parsed.get("issuer_type") or "lain")
    issuer_pass = issuer in {"rumah_sakit", "klinik", "puskesmas"}
    matched = is_health and issuer_pass and health_status_pass and expiry_pass
    if identity_pass is not None:
        matched = matched and identity_pass

    reasons: list[str] = []
    if not is_health:
        reasons.append("Bukan surat keterangan kesehatan.")
    elif not issuer_pass:
        reasons.append("Surat harus dari rumah sakit, klinik, atau puskesmas, bukan bidan.")
    if identity_pass is False:
        reasons.append("Nama pada surat tidak cocok dengan referensi.")
    if not health_status_pass:
        reasons.append("Tidak tertera dinyatakan SEHAT.")
    if not exam:
        reasons.append("Tanggal dibuat tidak terbaca.")
    elif not expiry_pass:
        reasons.append("Surat kesehatan kadaluarsa.")
    if not reasons:
        reasons.append("Surat keterangan kesehatan lolos: SEHAT, nama cocok, tanggal dibuat terbaca.")

    blockers: list[str] = []
    if not is_health:
        blockers.append("DOCUMENT_TYPE")
    if not issuer_pass:
        blockers.append("HEALTH_ISSUER")
    if not health_status_pass:
        blockers.append("HEALTH_STATUS")
    if not expiry_pass:
        blockers.append("HEALTH_CHECK_EXPIRY")
    if identity_pass is False:
        blockers.append("IDENTITY")
    if not exam:
        expiry_reason = "Tanggal pemeriksaan tidak terbaca."
    elif not expiry_pass:
        expiry_reason = "Surat kesehatan kadaluarsa."
    else:
        expiry_reason = ""

    out["validation"] = {
        "document_profile_id": "keterangan_kesehatan",
        "validation_engine": "ornith",
        "document_type_pass": is_health,
        "issuer_type": issuer,
        "issuer_pass": issuer_pass,
        "holder_name": str(parsed.get("holder_name") or ""),
        "health_exam_date": exam,
        "health_check": {
            "exam_date": exam,
            "valid_until": until,
            "reason": expiry_reason,
        },
        "explanation": {"primary_blockers": blockers},
        "identity_pass": identity_pass,
        "identity": identity,
        "health_status_pass": health_status_pass,
        "health_expiry_pass": expiry_pass,
        "issued_date": exam,
        "issued_date_display": format_id_date(exam) if exam else None,
        "valid_until": gate.get("valid_until") or parsed.get("valid_until"),
        "health_conclusion": conclusion,
        "document_matched": matched,
        "summary": reasons[0],
        "detail_lines": reasons,
        "gate": {
            "exam_date": gate.get("exam_date"),
            "valid_until": gate.get("valid_until"),
            "health_status_pass": gate.get("health_status_pass"),
            "health_expiry_pass": gate.get("health_expiry_pass"),
        },
        "synthetic_ocr": synthetic,
    }
    out["document_matched"] = matched
    out["error"] = None
    return out


def overlay_health_ai_on_validation(validation: dict[str, Any], ai_out: dict[str, Any]) -> dict[str, Any]:
    """Tempel hasil Ornith ke payload validasi pipeline (tanggal + mesin OCR)."""
    out = dict(validation)
    ext = ai_out.get("extraction") if isinstance(ai_out.get("extraction"), dict) else {}
    val = ai_out.get("validation") if isinstance(ai_out.get("validation"), dict) else {}
    out["ocr_engine"] = "ornith"
    out["health_ai"] = {"llm": ai_out.get("llm"), "extraction": ext}
    exam = val.get("issued_date") or ext.get("issued_date") or out.get("health_exam_date")
    until = val.get("valid_until") or ext.get("valid_until") or out.get("health_valid_until")
    if val.get("issuer_type"):
        out["issuer_type"] = val.get("issuer_type")
        out["issuer_pass"] = val.get("issuer_pass")
    if exam:
        out["health_exam_date"] = exam
        out["issued_date"] = exam
        out["issued_date_display"] = format_id_date(str(exam))
        hc = dict(out.get("health_check") or {})
        hc["exam_date"] = exam
        if until:
            hc["valid_until"] = until
            out["health_valid_until"] = until
        out["health_check"] = hc
        exam_day = None
        try:
            exam_day = date.fromisoformat(str(exam)[:10])
        except ValueError:
            exam_day = None
        if out.get("health_expiry_pass") is False and not until and exam_day and exam_day <= date.today():
            out["health_expiry_pass"] = True
            explanation = dict(out.get("explanation") or {})
            blockers = [
                b
                for b in (explanation.get("primary_blockers") or [])
                if b != "HEALTH_CHECK_EXPIRY"
            ]
            explanation["primary_blockers"] = blockers
            out["explanation"] = explanation
            if (
                out.get("document_type_pass")
                and out.get("health_status_pass")
                and (out.get("identity_pass") is not False)
            ):
                out["document_matched"] = True
    if ext.get("is_health_certificate") is False:
        out["document_type_pass"] = False
        out["document_matched"] = False
    if val.get("issuer_pass") is False:
        out["document_matched"] = False
        explanation = dict(out.get("explanation") or {})
        blockers = list(explanation.get("primary_blockers") or [])
        if "HEALTH_ISSUER" not in blockers:
            blockers.append("HEALTH_ISSUER")
        explanation["primary_blockers"] = blockers
        out["explanation"] = explanation
    return out


def health_ai_config() -> dict[str, Any]:
    base = health_ai_base_url()
    try:
        base = assert_allowed_llm_url(base)
        url_ok = True
        url_error = None
    except ValueError as e:
        url_ok = False
        url_error = str(e)
    models = list_health_ai_models(base) if url_ok else []
    return {
        "base_url": base,
        "model": health_ai_model(),
        "models": models,
        "url_ok": url_ok,
        "url_error": url_error,
        "system_prompt": HEALTH_LETTER_SYSTEM_PROMPT,
        "user_prompt_template": health_letter_user_prompt(
            expected_name="{expected_name}",
            expected_nik="{expected_nik}",
        ),
    }
