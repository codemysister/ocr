"""Log persisten untuk dokumen yang gagal pipeline / validasi (preview di /dataset-failures)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from systems.validation.skck_validity import format_id_date

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO_ROOT / "logs" / "validation_failures.jsonl"
_CLEARED_STAMP = _REPO_ROOT / "logs" / "validation_failures.cleared"


def _log_path() -> Path:
    raw = (os.environ.get("VALIDATION_FAILURE_LOG_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_LOG


def _max_entries() -> int:
    try:
        return max(50, int(os.environ.get("VALIDATION_FAILURE_LOG_MAX_ENTRIES", "2000")))
    except ValueError:
        return 2000


def _truncate(text: str, max_len: int = 8000) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _infer_folder(filename: str) -> str:
    name = Path(filename or "").name
    if not name:
        return ""
    dataset = _REPO_ROOT / "dataset"
    if "_" in name:
        prefix = name.split("_", 1)[0]
        if prefix and (dataset / prefix / name).is_file():
            return prefix
    if dataset.is_dir():
        for folder in dataset.iterdir():
            if folder.is_dir() and (folder / name).is_file():
                return folder.name
    return ""


def _ocr_text_from_result(result: Any) -> str | None:
    payload = getattr(result, "payload", None) or {}
    if not isinstance(payload, dict):
        return None
    ocr = payload.get("ocr")
    if not isinstance(ocr, dict):
        return None
    text = ocr.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    markdown = ocr.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()
    return None


def _pipeline_response_from_result(result: Any) -> dict[str, Any] | None:
    payload = getattr(result, "payload", None) or {}
    if not isinstance(payload, dict):
        return None
    out: dict[str, Any] = {}
    validation = payload.get("validation")
    verdict = payload.get("verdict")
    if isinstance(validation, dict):
        out["validation"] = validation
    if isinstance(verdict, dict):
        out["verdict"] = verdict
    ocr = payload.get("ocr")
    if isinstance(ocr, dict):
        ocr_copy = {k: v for k, v in ocr.items() if k != "timing"}
        if ocr_copy:
            out["ocr"] = ocr_copy
    if payload.get("validation_mode"):
        out["validation_mode"] = payload["validation_mode"]
    return out or None


def _failure_detail(result: Any) -> tuple[str | None, str | None]:
    if getattr(result, "ok", False) and getattr(result, "document_matched", False):
        return None, None
    if not getattr(result, "ok", False):
        kind = getattr(result, "error_kind", None) or "pipeline_error"
        return kind, getattr(result, "error", None) or "Pipeline gagal tanpa pesan."

    payload = getattr(result, "payload", None) or {}
    validation = payload.get("validation") if isinstance(payload, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    explanation = validation.get("explanation") if isinstance(validation.get("explanation"), dict) else {}
    parts: list[str] = []
    summary = explanation.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    blockers = explanation.get("primary_blockers")
    if isinstance(blockers, list) and blockers:
        parts.append("Blocker: " + ", ".join(str(b) for b in blockers))
    reason = " — ".join(parts) if parts else "Validasi gagal (document_matched=false)."
    return "validation_fail", reason


def _trim_log_file(path: Path, max_entries: int) -> None:
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= max_entries:
        return
    kept = lines[-max_entries:]
    try:
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except OSError:
        logger.exception("validation_failure_log: gagal trim %s", path)


def append_validation_failure(entry: dict[str, Any]) -> dict[str, Any]:
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
        logger.exception("validation_failure_log: gagal menulis %s", path)
    return record


def record_from_pipeline_result(
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
    skip_kinds = {"empty_file", "empty_document_type", "unknown_document_type"}
    kind, reason = _failure_detail(result)
    if not kind:
        return None
    if kind in skip_kinds:
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

    entry = {
        "source": source,
        "folder": folder,
        "file": file_name,
        "document_type": (document_type or "").strip() or validation.get("document_profile_id"),
        "expected_name": (expected_name or "").strip() or None,
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
        "detected_bank": (validation.get("bank_detection") or {}).get("detected_bank")
        if isinstance(validation.get("bank_detection"), dict)
        else None,
        "failure_kind": kind,
        "failure_reason": reason,
        "ocr_text": _truncate(ocr_text or "") or None,
        "pipeline_response": _pipeline_response_from_result(result),
        "document_matched": getattr(result, "document_matched", False),
        "timing": timing,
        "error": getattr(result, "error", None),
        "error_kind": getattr(result, "error_kind", None),
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
    return append_validation_failure(entry)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        return []
    return rows


def maybe_seed_from_ktp_logs() -> int:
    """Isi log dari tes KTP terakhir bila belum ada file log (dan belum pernah di-clear)."""
    if _log_path().is_file() or _CLEARED_STAMP.is_file():
        return 0
    timing_path = _REPO_ROOT / "logs" / "ktp_199_timing.jsonl"
    helper_path = _REPO_ROOT / "logs" / "ktp_helper_benchmark.jsonl"
    timing_rows = _read_jsonl(timing_path)
    if not timing_rows:
        return 0
    helper_by_file = {r.get("file"): r for r in _read_jsonl(helper_path) if r.get("file")}
    seeded = 0
    for row in timing_rows:
        http_ok = row.get("http") == 200
        matched = row.get("document_matched") is True
        if http_ok and matched:
            continue
        helper = helper_by_file.get(row.get("file")) or {}
        blockers = helper.get("blockers") if isinstance(helper.get("blockers"), list) else []
        if not http_ok:
            kind = "pipeline_error"
            reason = row.get("error") or helper.get("error") or f"HTTP {row.get('http')}"
        else:
            kind = "validation_fail"
            parts = []
            if blockers:
                parts.append("Blocker: " + ", ".join(str(b) for b in blockers if b))
            if helper.get("identity_pass") is False:
                parts.append("identitas gagal")
            if helper.get("marital_pass") is False:
                parts.append("status perkawinan gagal")
            reason = " — ".join(parts) if parts else "Validasi gagal (document_matched=false)."
        file_name = row.get("file") or ""
        append_validation_failure(
            {
                "source": "ktp_199_timing",
                "folder": _infer_folder(file_name) or "ktp",
                "file": file_name,
                "document_type": "ktp",
                "expected_name": helper.get("expected_name"),
                "job_position": "helper",
                "identity_pass": helper.get("identity_pass"),
                "marital_pass": helper.get("marital_pass"),
                "marital_status": helper.get("marital_status"),
                "failure_kind": kind,
                "failure_reason": reason,
                "ocr_text": None,
                "pipeline_response": None,
                "document_matched": False,
                "timing": {
                    "ocr": row.get("ocr_s"),
                    "total": row.get("pipeline_total_s") or row.get("client_wall_s"),
                },
                "extracted_name": helper.get("extracted_name"),
                "error": row.get("error") or helper.get("error"),
            }
        )
        seeded += 1
    return seeded


def read_validation_failures(*, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    maybe_seed_from_ktp_logs()
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


def count_validation_failures() -> int:
    maybe_seed_from_ktp_logs()
    path = _log_path()
    if not path.is_file():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def clear_validation_failures() -> int:
    path = _log_path()
    n = 0
    if path.is_file():
        try:
            n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("validation_failure_log: gagal hapus %s", path)
            return 0
    try:
        _CLEARED_STAMP.parent.mkdir(parents=True, exist_ok=True)
        _CLEARED_STAMP.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    except OSError:
        logger.exception("validation_failure_log: gagal tulis stamp clear")
    return n


def failure_log_stats() -> dict[str, Any]:
    rows = read_validation_failures(limit=2000, offset=0)
    validation = sum(1 for r in rows if (r.get("failure_kind") or "validation_fail") == "validation_fail")
    pipeline = len(rows) - validation
    return {
        "total": count_validation_failures(),
        "in_recent_window": len(rows),
        "validation_fail": validation,
        "pipeline_error": pipeline,
        "log_path": str(_log_path()),
    }
