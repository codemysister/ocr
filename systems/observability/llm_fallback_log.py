"""Log persisten untuk setiap kegagalan Paddle yang memicu fallback AI vision."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO_ROOT / "logs" / "llm_fallback_errors.jsonl"


def _log_path() -> Path:
    raw = (os.environ.get("LLM_FALLBACK_LOG_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_LOG


def _max_entries() -> int:
    try:
        return max(50, int(os.environ.get("LLM_FALLBACK_LOG_MAX_ENTRIES", "500")))
    except ValueError:
        return 500


def _truncate(text: str, max_len: int = 400) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _paddle_failure_summary(detail: dict[str, Any]) -> str:
    explanation = detail.get("explanation") if isinstance(detail.get("explanation"), dict) else {}
    summary = explanation.get("summary") if isinstance(explanation, dict) else None
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    blockers = explanation.get("primary_blockers") if isinstance(explanation, dict) else None
    if isinstance(blockers, list) and blockers:
        return "Blocker: " + ", ".join(str(b) for b in blockers)
    return "Validasi Paddle gagal (document_matched=false)."


def append_llm_fallback_log(entry: dict[str, Any]) -> dict[str, Any]:
    """Tambahkan satu baris JSONL; kembalikan record lengkap dengan id & timestamp."""
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
        logger.exception("llm_fallback_log: gagal menulis %s", path)
    return record


def record_pipeline_llm_fallback(
    *,
    paddle_detail: dict[str, Any],
    final_detail: dict[str, Any],
    filename: str,
    document_type: str,
    document_profile_id: str,
    expected_name: str,
    expected_nik: str,
    ocr_mode: str | None,
    ocr_text: str | None,
    source: str = "pipeline",
) -> dict[str, Any] | None:
    """Catat bila fallback AI vision dipanggil setelah Paddle gagal."""
    fb = final_detail.get("llm_fallback")
    if not isinstance(fb, dict) or not fb.get("attempted"):
        return None

    final_matched = bool(final_detail.get("document_matched"))
    entry = {
        "source": source,
        "filename": filename,
        "document_type": document_type,
        "document_profile_id": document_profile_id,
        "expected_name": expected_name,
        "expected_nik": expected_nik,
        "ocr_mode": ocr_mode,
        "paddle_matched": bool(paddle_detail.get("document_matched")),
        "paddle_failure_summary": _paddle_failure_summary(paddle_detail),
        "paddle_document_type_pass": paddle_detail.get("document_type_pass"),
        "paddle_identity_pass": paddle_detail.get("identity_pass"),
        "ocr_text_preview": _truncate(ocr_text or "", 600),
        "llm_fallback": fb,
        "llm_validation": final_detail.get("llm_validation"),
        "llm_annotation": final_detail.get("llm_annotation"),
        "final_matched": final_matched,
        "rescued": final_matched and not bool(paddle_detail.get("document_matched")),
    }
    return append_llm_fallback_log(entry)


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
        logger.exception("llm_fallback_log: gagal trim %s", path)


def read_llm_fallback_logs(*, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    """Baca entri terbaru dulu (newest first)."""
    path = _log_path()
    if not path.is_file():
        return []
    limit = max(1, min(limit, 500))
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


def count_llm_fallback_logs() -> int:
    path = _log_path()
    if not path.is_file():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def clear_llm_fallback_logs() -> int:
    """Hapus semua log; kembalikan jumlah baris yang dihapus."""
    path = _log_path()
    if not path.is_file():
        return 0
    try:
        n = count_llm_fallback_logs()
        path.unlink(missing_ok=True)
        return n
    except OSError:
        logger.exception("llm_fallback_log: gagal hapus %s", path)
        return 0


def log_stats() -> dict[str, Any]:
    rows = read_llm_fallback_logs(limit=500, offset=0)
    rescued = sum(1 for r in rows if r.get("rescued"))
    failed = sum(1 for r in rows if r.get("llm_fallback", {}).get("attempted") and not r.get("final_matched"))
    return {
        "total": count_llm_fallback_logs(),
        "in_recent_window": len(rows),
        "rescued_recent": rescued,
        "failed_recent": failed,
        "log_path": str(_log_path()),
    }
