"""Satu file JSON berisi ringkasan operasi terakhir untuk review & tuning (overwrite tiap kali)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO_ROOT / "logs" / "last_tuning.json"


def _log_path() -> Path:
    raw = (os.environ.get("LAST_TUNING_LOG_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_LOG


def _truncate(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def summarize_text_fields(
    d: dict[str, Any],
    *,
    text_keys: tuple[str, ...] = ("text", "markdown"),
    preview_chars: int = 240,
) -> dict[str, Any]:
    """Salinan dangkal: panjang teks + cuplikan singkat (tanpa `result_json` besar)."""
    out: dict[str, Any] = {}
    skip = frozenset(text_keys) | {"result_json", "image_base64"}
    for k, v in d.items():
        if k in skip:
            continue
        out[k] = v
    for k in text_keys:
        if k not in d:
            continue
        t = d[k]
        if isinstance(t, str):
            out[f"{k}_chars"] = len(t)
            out[f"{k}_preview"] = _truncate(t.replace("\n", " "), preview_chars)
        else:
            out[k] = v
    if "lines" in d and isinstance(d["lines"], list):
        out["lines_count"] = len(d["lines"])
    return out


def write_last_tuning_log(record: dict[str, Any]) -> None:
    """
    Timpa file log dengan record terbaru. Tidak memicu error ke permintaan HTTP jika penulisan gagal.
    """
    path = _log_path()
    payload: dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        logger.exception("last_tuning_log: gagal menulis %s", path)


def summarize_validation_result(payload: dict[str, Any], *, max_keywords: int = 40) -> dict[str, Any]:
    """Potong daftar keyword panjang; sisanya untuk tuning."""
    keys = (
        "document_matched",
        "document_type",
        "document_profile_id",
        "document_type_pass",
        "identity_pass",
        "document_type_aggregate_pass_ratio",
        "aggregate_min_pass_ratio",
        "identity_min_score",
        "explanation",
        "name_extraction",
        "identity",
        "document_type_components_count",
        "keywords_from_profile",
        "verdict",
        "is_own_document",
        "document_type_current",
        "document_type_current_label",
    )
    out: dict[str, Any] = {k: payload[k] for k in keys if k in payload}
    if "keywords" in payload and isinstance(payload["keywords"], list):
        kws = payload["keywords"]
        out["keywords"] = kws[:max_keywords]
        if len(kws) > max_keywords:
            out["keywords_truncated_after"] = max_keywords
    return out


def log_safe_failure(
    *,
    subsystem: str,
    method: str,
    path: str,
    http_status: int,
    detail: Any,
) -> None:
    """Ringkas error HTTP / validasi."""
    err: Any = detail
    if isinstance(detail, dict):
        err = detail
    elif detail is not None:
        err = str(detail)
    write_last_tuning_log(
        {
            "success": False,
            "subsystem": subsystem,
            "method": method,
            "path": path,
            "http_status": http_status,
            "error": err,
        }
    )
