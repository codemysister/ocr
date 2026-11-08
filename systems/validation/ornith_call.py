"""Panggilan vision Ornith yang dipakai beberapa profil validasi."""

from __future__ import annotations

import os
import re
from typing import Any

from systems.validation.keterangan_kesehatan_ai import (
    _parse_health_json,
    assert_allowed_llm_url,
    health_ai_base_url,
    health_ai_model,
)
from systems.validation.llm_fallback import complete_vision
from systems.validation.rekening_ai import rekening_holder_name_matches


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().casefold()
    return text in {"true", "ya", "yes", "1"}


def nik16(value: str) -> str:
    digits = digits_only(value)
    return digits if len(digits) == 16 else ""


def decide_identity(
    *,
    expected_name: str,
    holder: str,
    expected_nik: str = "",
    nik: str = "",
    use_nik: bool = True,
) -> dict[str, Any]:
    """Nama boleh terpotong. NIK 16 digit yang sama dengan referensi juga lolos."""
    exp_name = (expected_name or "").strip()
    exp_nik = nik16(expected_nik) if use_nik else ""
    got_nik = nik16(nik) if use_nik else ""
    if exp_name and holder:
        name_info = rekening_holder_name_matches(exp_name, holder)
    elif exp_name:
        name_info = {"pass": False, "method": "none", "identity": None}
    else:
        name_info = {"pass": None, "method": "skipped", "identity": None}
    name_pass = name_info.get("pass")
    nik_pass: bool | None = None
    if exp_nik and got_nik:
        nik_pass = got_nik == exp_nik
    elif exp_nik:
        nik_pass = False

    if nik_pass is True:
        identity_pass: bool | None = True
        method = "nik" if name_pass is not True else name_info.get("method")
    elif name_pass is True:
        identity_pass = True
        method = name_info.get("method")
    elif exp_name or exp_nik:
        identity_pass = False
        method = name_info.get("method")
    else:
        identity_pass = None
        method = "skipped"
    return {
        "name_pass": name_pass,
        "nik_pass": nik_pass,
        "identity_pass": identity_pass,
        "method": method or "none",
        "identity": name_info.get("identity") if isinstance(name_info.get("identity"), dict) else None,
        "nik": got_nik,
    }


def run_ornith_json(
    image_bytes: bytes,
    *,
    user_prompt: str,
    system_text: str,
    env_prefix: str,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Kirim gambar ke Ornith dan parse satu objek JSON."""
    base_url = assert_allowed_llm_url(
        health_ai_base_url(os.environ.get(f"{env_prefix}_BASE_URL"))
    )
    model = health_ai_model(os.environ.get(f"{env_prefix}_MODEL"))
    from systems.preprocessing.pipeline import encode_image_bytes_for_ocr_remote_safe

    prepared, prep_meta = encode_image_bytes_for_ocr_remote_safe(image_bytes, max_long_side=1280)
    thinking_on = (os.environ.get("LLM_HEALTH_AI_ENABLE_THINKING") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    extra: dict[str, Any] | None = None
    if not thinking_on:
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    content, meta = complete_vision(
        prepared,
        user_prompt,
        image_mime="image/jpeg",
        system_text=system_text,
        max_tokens=int(os.environ.get(f"{env_prefix}_MAX_TOKENS", str(max_tokens))),
        temperature=0.0,
        base_url=base_url,
        model=model,
        timeout_s=float(
            os.environ.get(f"{env_prefix}_TIMEOUT_S", os.environ.get("LLM_HEALTH_AI_TIMEOUT_S", "120"))
        ),
        extra=extra,
    )
    llm = {
        "base_url": base_url,
        "model": model,
        "latency_s": meta.get("latency_s"),
        "success": meta.get("success"),
        "error": meta.get("error"),
    }
    if not content:
        return {"error": meta.get("error") or "empty_content", "llm": llm, "image_prepare": prep_meta}
    parsed = _parse_health_json(content)
    if not parsed:
        return {
            "error": "invalid_json",
            "llm": llm,
            "raw_content": content,
            "image_prepare": prep_meta,
        }
    return {
        "error": None,
        "llm": llm,
        "parsed": parsed,
        "image_prepare": prep_meta,
        "raw_content": content,
    }
