"""Log persisten untuk dokumen yang lolos validasi (preview di /dataset-successes)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from systems.observability.validation_failure_log import (
    _infer_folder,
    _ocr_text_from_result,
    _pipeline_response_from_result,
    _trim_log_file,
    _truncate,
)
from systems.validation.skck_validity import format_id_date

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO_ROOT / "logs" / "validation_successes.jsonl"
_CLEARED_STAMP = _REPO_ROOT / "logs" / "validation_successes.cleared"


def _log_path() -> Path:
    raw = (os.environ.get("VALIDATION_SUCCESS_LOG_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_LOG


def _max_entries() -> int:
    try:
        return max(50, int(os.environ.get("VALIDATION_SUCCESS_LOG_MAX_ENTRIES", "2000")))
    except ValueError:
        return 2000


def append_validation_success(entry: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        _trim_log_file(path, _max_entries())
    except Exception:
        logger.exception("validation_success_log: gagal menulis %s", path)
    return record


def _success_reason(validation: dict[str, Any]) -> str:
    explanation = validation.get("explanation") if isinstance(validation.get("explanation"), dict) else {}
    summary = explanation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return "Validasi lolos (document_matched=true)."


def record_success_from_pipeline_result(
    result: Any,
    *,
    filename: str,
    document_type: str,
    expected_name: str = "",
    expected_bank: str = "",
    expected_account: str = "",
    job_position: str = "",
    source: str = "pipeline",
) -> dict[str, Any] | None:
    if not getattr(result, "ok", False) or not getattr(result, "document_matched", False):
        return None

    payload = getattr(result, "payload", None) or {}
    validation = payload.get("validation") if isinstance(payload, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    marital = validation.get("marital_status")
    marital_label = None
    if isinstance(marital, dict):
        marital_label = marital.get("label") or marital.get("status")
    elif isinstance(marital, str):
        marital_label = marital

    ocr_text = _ocr_text_from_result(result)
    timing = getattr(getattr(result, "timing", None), "as_dict", lambda: {})()
    file_name = Path(filename or "").name or filename
    folder = _infer_folder(file_name)
    name_extraction = validation.get("name_extraction") if isinstance(validation.get("name_extraction"), dict) else {}

    entry = {
        "source": source,
        "folder": folder,
        "file": file_name,
        "document_type": (document_type or "").strip() or validation.get("document_profile_id"),
        "expected_name": (expected_name or "").strip() or None,
        "extracted_name": name_extraction.get("candidate_raw"),
        "expected_bank": (expected_bank or "").strip() or None,
        "expected_account": (expected_account or "").strip() or None,
        "job_position": (job_position or "").strip() or validation.get("job_position"),
        "identity_pass": validation.get("identity_pass"),
        "identity_or_account_pass": validation.get("identity_or_account_pass"),
        "bank_pass": validation.get("bank_pass"),
        "account_pass": validation.get("account_pass"),
        "marital_pass": validation.get("marital_pass"),
        "marital_status": marital_label,
        "skck_expiry_pass": validation.get("skck_expiry_pass"),
        "skck_valid_until": validation.get("skck_valid_until"),
        "skck_valid_from": validation.get("skck_valid_from"),
        "health_status_pass": validation.get("health_status_pass"),
        "health_expiry_pass": validation.get("health_expiry_pass"),
        "health_exam_date": validation.get("health_exam_date"),
        "health_valid_until": validation.get("health_valid_until"),
        "issued_date": validation.get("issued_date")
        or validation.get("health_exam_date")
        or validation.get("skck_valid_from"),
        "issued_date_display": validation.get("issued_date_display"),
        "jkn_status_label": validation.get("jkn_status_label"),
        "jkn_status_display": validation.get("jkn_status_display"),
        "detected_bank": (validation.get("bank_detection") or {}).get("detected_bank")
        if isinstance(validation.get("bank_detection"), dict)
        else None,
        "success_kind": "validation_pass",
        "success_reason": _success_reason(validation),
        "ocr_text": _truncate(ocr_text or "") or None,
        "pipeline_response": _pipeline_response_from_result(result),
        "document_matched": True,
        "timing": timing,
        "error_code": None,
        "error_message": None,
    }
    if entry.get("issued_date") and not entry.get("issued_date_display"):
        entry["issued_date_display"] = format_id_date(entry["issued_date"])
    llm = validation.get("llm_fallback")
    if isinstance(llm, dict) and llm.get("attempted"):
        entry["llm_fallback"] = llm
        if isinstance(validation.get("llm_annotation"), dict):
            entry["llm_annotation"] = validation["llm_annotation"]
        if isinstance(validation.get("llm_validation"), dict):
            entry["llm_validation"] = validation["llm_validation"]
    return append_validation_success(entry)


def read_validation_successes(*, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    path = _log_path()
    if not path.is_file():
        return []
    limit = max(1, min(limit, 2000))
    offset = max(0, offset)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[offset : offset + limit]


def count_validation_successes() -> int:
    path = _log_path()
    if not path.is_file():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def clear_validation_successes() -> int:
    path = _log_path()
    n = 0
    if path.is_file():
        try:
            n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("validation_success_log: gagal hapus %s", path)
            return 0
    try:
        _CLEARED_STAMP.parent.mkdir(parents=True, exist_ok=True)
        _CLEARED_STAMP.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    except OSError:
        logger.exception("validation_success_log: gagal tulis stamp clear")
    return n


def success_log_stats() -> dict[str, Any]:
    rows = read_validation_successes(limit=2000, offset=0)
    by_type: dict[str, int] = {}
    for r in rows:
        key = str(r.get("document_type") or "lainnya")
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "total": count_validation_successes(),
        "in_recent_window": len(rows),
        "by_document_type": by_type,
        "log_path": str(_log_path()),
    }
