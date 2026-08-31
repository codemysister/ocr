"""Benchmark OCR pipeline terhadap file di folder dataset/."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from app.pipeline_runner import OcrMode, PipelineResult, run_pipeline_bytes
from systems.validation.document_profiles import profile_label, resolve_keywords

_NIK_SUFFIX_RE = re.compile(r"_(\d{16})$")
_DATASET_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
_MAX_LIMIT_PER_TYPE = 500


def dataset_root() -> Path:
    env = os.environ.get("DATASET_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "dataset"


def parse_dataset_filename(folder: str, filename: str) -> str | None:
    """
    Ekstrak expected_name dari pola `{folder}_{Nama}_{NIK16}.ext`.
    Mengembalikan None jika pola tidak cocok.
    """
    stem = Path(filename).stem
    prefix = folder + "_"
    if stem.casefold().startswith(prefix.casefold()):
        stem = stem[len(prefix) :]
    elif stem.startswith(prefix):
        stem = stem[len(prefix) :]
    else:
        return None

    m = _NIK_SUFFIX_RE.search(stem)
    if not m:
        return None
    name_part = stem[: m.start()].strip()
    return name_part or None


def resolve_dataset_file(folder: str, filename: str) -> Path:
    """Path aman ke file di dataset/ (tolak traversal)."""
    if not folder.strip() or not filename.strip():
        raise ValueError("folder dan file wajib.")
    if any(x in folder for x in ("..", "/", "\\")) or any(x in filename for x in ("..", "/", "\\")):
        raise ValueError("Path tidak valid.")
    root = dataset_root().resolve()
    file_path = (root / folder / filename).resolve()
    if not str(file_path).startswith(str(root)):
        raise ValueError("Path di luar dataset.")
    if not file_path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {folder}/{filename}")
    if file_path.suffix.casefold() not in _DATASET_EXTS:
        raise ValueError("Ekstensi file tidak didukung.")
    return file_path


def _ocr_text_from_result(result: PipelineResult) -> str | None:
    """Ambil teks OCR mentah dari hasil pipeline (jika ada)."""
    if not result.payload:
        return None
    ocr = result.payload.get("ocr")
    if not isinstance(ocr, dict):
        return None
    text = ocr.get("text")
    if isinstance(text, str) and text.strip():
        return text
    markdown = ocr.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()
    return None


def _pipeline_response_from_result(result: PipelineResult) -> dict[str, Any] | None:
    """Ringkasan respons pipeline yang sudah diolah (validasi / verdict / OCR meta)."""
    if not result.payload:
        return None
    payload = result.payload
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


def _llm_fallback_from_result(result: PipelineResult) -> dict[str, Any] | None:
    """Ringkasan fallback AI lokal dari hasil validasi (jika dipanggil)."""
    if not result.payload:
        return None
    validation = result.payload.get("validation")
    if not isinstance(validation, dict):
        return None
    fb = validation.get("llm_fallback")
    if not isinstance(fb, dict) or not fb.get("attempted"):
        return None
    out: dict[str, Any] = {"llm_fallback": fb}
    ann = validation.get("llm_annotation")
    if isinstance(ann, dict):
        out["llm_annotation"] = ann
    elif isinstance(fb.get("annotation"), dict):
        out["llm_annotation"] = fb["annotation"]
    llm_val = validation.get("llm_validation")
    if isinstance(llm_val, dict):
        out["llm_validation"] = llm_val
    elif isinstance(fb.get("validation"), dict):
        out["llm_validation"] = fb["validation"]
    return out


def _review_fields_from_result(result: PipelineResult) -> dict[str, Any]:
    """Field tambahan untuk review kegagalan di UI benchmark."""
    out: dict[str, Any] = {}
    ocr_text = _ocr_text_from_result(result)
    if ocr_text:
        out["ocr_text"] = ocr_text
    pipeline_response = _pipeline_response_from_result(result)
    if pipeline_response:
        out["pipeline_response"] = pipeline_response
    llm_fb = _llm_fallback_from_result(result)
    if llm_fb:
        out.update(llm_fb)
    return out


def _failure_detail(result: PipelineResult) -> tuple[str | None, str | None]:
    """Kembalikan (failure_kind, alasan) bila gagal; (None, None) bila lolos."""
    if result.ok and result.document_matched:
        return None, None

    if not result.ok:
        kind = result.error_kind or "pipeline_error"
        return kind, result.error or "Pipeline gagal tanpa pesan."

    payload = result.payload or {}
    if payload.get("validation_mode") == "cv":
        cv_match = payload.get("cv_match") or {}
        summary = cv_match.get("summary")
        parts: list[str] = []
        if isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())
        dims = cv_match.get("dimensions") or {}
        for dim_name, dim in dims.items():
            if not isinstance(dim, dict) or dim.get("pass") is not False:
                continue
            pct = dim.get("percent")
            parts.append(f"{dim_name} gagal ({pct}%)" if pct is not None else f"{dim_name} gagal")
        reason = " — ".join(parts) if parts else "CV match gagal."
        return "validation_fail", reason

    validation = payload.get("validation") or {}
    explanation = validation.get("explanation") or {}
    verdict = validation.get("verdict") or payload.get("verdict") or {}

    parts: list[str] = []
    summary = explanation.get("summary") if isinstance(explanation, dict) else None
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    elif isinstance(verdict, dict):
        vs = verdict.get("summary")
        if isinstance(vs, str) and vs.strip():
            parts.append(vs.strip())

    blockers = explanation.get("primary_blockers") if isinstance(explanation, dict) else None
    if isinstance(blockers, list) and blockers:
        parts.append("Blocker: " + ", ".join(str(b) for b in blockers))

    detail_lines = explanation.get("detail_lines") if isinstance(explanation, dict) else None
    if isinstance(detail_lines, list):
        for line in detail_lines[:2]:
            if isinstance(line, str) and line.strip():
                parts.append(line.strip())

    reason = " — ".join(parts) if parts else "Validasi gagal (document_matched=false)."
    return "validation_fail", reason


@dataclass
class DatasetFolderInfo:
    folder: str
    file_count: int
    document_type: str | None
    supported: bool
    label: str


def list_dataset_folders(root: Path | None = None) -> list[DatasetFolderInfo]:
    base = root or dataset_root()
    if not base.is_dir():
        return []

    rows: list[DatasetFolderInfo] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        files = [
            p.name
            for p in entry.iterdir()
            if p.is_file() and p.suffix.casefold() in _DATASET_EXTS
        ]
        resolved = resolve_keywords(entry.name)
        if resolved:
            cid, _ = resolved
            rows.append(
                DatasetFolderInfo(
                    folder=entry.name,
                    file_count=len(files),
                    document_type=cid,
                    supported=True,
                    label=profile_label(cid),
                )
            )
        else:
            rows.append(
                DatasetFolderInfo(
                    folder=entry.name,
                    file_count=len(files),
                    document_type=None,
                    supported=False,
                    label=entry.name,
                )
            )
    return rows


def _list_files_for_folder(folder_path: Path, limit: int, offset: int = 0) -> list[Path]:
    files = sorted(
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.casefold() in _DATASET_EXTS
    )
    cap = min(max(limit, 0), _MAX_LIMIT_PER_TYPE)
    off = max(offset, 0)
    return files[off : off + cap]


@dataclass
class BenchmarkSelection:
    folder: str
    enabled: bool = False
    limit: int = 20
    offset: int = 0
    files: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    selections: list[BenchmarkSelection]
    ocr_mode: OcrMode = "fast"
    pp_ocr_tier: str = "medium"
    use_expected_name: bool = True
    enable_preprocess: bool = False
    skip_passthrough: bool = False


@dataclass
class _TimingAgg:
    preprocess: list[float] = field(default_factory=list)
    ocr: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)

    def add(self, timing: dict[str, float]) -> None:
        for key in ("preprocess", "ocr", "validation", "total"):
            val = timing.get(key)
            if isinstance(val, (int, float)):
                getattr(self, key).append(float(val))

    def avg(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in ("preprocess", "ocr", "validation", "total"):
            vals = getattr(self, key)
            out[key] = round(sum(vals) / len(vals), 3) if vals else 0.0
        return out


@dataclass
class _StatsBucket:
    total: int = 0
    validation_pass: int = 0
    validation_fail: int = 0
    pipeline_error: int = 0
    timing: _TimingAgg = field(default_factory=_TimingAgg)

    def ratios(self) -> dict[str, float]:
        if self.total == 0:
            return {"success_ratio": 0.0, "failure_ratio": 0.0, "error_ratio": 0.0}
        return {
            "success_ratio": round(self.validation_pass / self.total, 4),
            "failure_ratio": round(self.validation_fail / self.total, 4),
            "error_ratio": round(self.pipeline_error / self.total, 4),
        }

    def to_dict(self, *, folder: str | None = None, document_type: str | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "total": self.total,
            "validation_pass": self.validation_pass,
            "validation_fail": self.validation_fail,
            "pipeline_error": self.pipeline_error,
            **self.ratios(),
            "timing_avg_s": self.timing.avg(),
        }
        if folder is not None:
            d["folder"] = folder
        if document_type is not None:
            d["document_type"] = document_type
        return d


def _process_result(
    result: PipelineResult,
    bucket: _StatsBucket,
) -> tuple[bool, bool | None]:
    """Return (pipeline_ok, document_matched)."""
    t = result.timing.as_dict()
    bucket.timing.add(t)
    bucket.total += 1

    if not result.ok:
        bucket.pipeline_error += 1
        return False, None

    matched = result.document_matched
    if matched:
        bucket.validation_pass += 1
    else:
        bucket.validation_fail += 1
    return True, matched


def _parse_specific_file_ref(folder: str, ref: str) -> tuple[str, str]:
    """Parse referensi file: `nama.jpg` atau `folder/nama.jpg`."""
    ref = ref.strip().replace("\\", "/")
    if not ref or ref.startswith("#"):
        raise ValueError("Referensi file kosong.")
    if "/" in ref:
        parts = ref.split("/", 1)
        job_folder, job_file = parts[0].strip(), parts[1].strip()
        if not job_folder or not job_file:
            raise ValueError(f"Referensi tidak valid: {ref}")
        return job_folder, job_file
    if not folder.strip():
        raise ValueError(f"Folder wajib untuk file `{ref}` (atau tulis folder/file).")
    return folder.strip(), ref


def _collect_jobs_for_selection(
    sel: BenchmarkSelection,
    root: Path,
    *,
    use_expected_name: bool,
) -> tuple[list[tuple[str, Path, str | None]], list[str]]:
    """
    Kumpulkan job dari selection (file spesifik atau limit/offset).
    Return (jobs, error_messages).
    """
    jobs: list[tuple[str, Path, str | None]] = []
    errors: list[str] = []

    if sel.files:
        seen: set[tuple[str, str]] = set()
        for ref in sel.files:
            try:
                job_folder, job_file = _parse_specific_file_ref(sel.folder, ref)
            except ValueError as e:
                errors.append(str(e))
                continue
            key = (job_folder.casefold(), job_file.casefold())
            if key in seen:
                continue
            seen.add(key)
            try:
                fp = resolve_dataset_file(job_folder, job_file)
            except FileNotFoundError:
                errors.append(f"File tidak ditemukan: {job_folder}/{job_file}")
                continue
            except ValueError as e:
                errors.append(str(e))
                continue
            expected = None
            if use_expected_name:
                expected = parse_dataset_filename(job_folder, fp.name)
            jobs.append((job_folder, fp, expected))
        return jobs, errors

    if not sel.enabled or sel.limit <= 0:
        return [], errors

    folder_path = root / sel.folder
    if not folder_path.is_dir():
        errors.append(f"Folder tidak ditemukan: {sel.folder}")
        return [], errors

    for fp in _list_files_for_folder(folder_path, sel.limit, sel.offset):
        expected = None
        if use_expected_name:
            expected = parse_dataset_filename(sel.folder, fp.name)
        jobs.append((sel.folder, fp, expected))
    return jobs, errors


def run_benchmark(config: BenchmarkConfig) -> Generator[str, None, None]:
    """Yield baris NDJSON: progress, result, summary."""
    root = dataset_root()
    global_stats = _StatsBucket()
    by_folder: dict[str, _StatsBucket] = defaultdict(_StatsBucket)
    folder_meta: dict[str, tuple[str | None, str | None]] = {}
    all_results: list[dict[str, Any]] = []

    # Kumpulkan job
    jobs: list[tuple[str, Path, str | None]] = []
    for sel in config.selections:
        batch_jobs, batch_errors = _collect_jobs_for_selection(
            sel,
            root,
            use_expected_name=config.use_expected_name,
        )
        for msg in batch_errors:
            yield json.dumps(
                {
                    "type": "error",
                    "folder": sel.folder,
                    "message": msg,
                },
                ensure_ascii=False,
            ) + "\n"
        for folder, fp, expected in batch_jobs:
            if folder not in folder_meta:
                resolved = resolve_keywords(folder)
                doc_type = resolved[0] if resolved else None
                folder_meta[folder] = (doc_type, profile_label(doc_type) if doc_type else None)
            jobs.append((folder, fp, expected))

    total_jobs = len(jobs)
    for idx, (folder, file_path, expected_name) in enumerate(jobs, start=1):
        yield json.dumps(
            {
                "type": "progress",
                "folder": folder,
                "file": file_path.name,
                "index": idx,
                "total": total_jobs,
            },
            ensure_ascii=False,
        ) + "\n"

        raw = file_path.read_bytes()
        resolved = resolve_keywords(folder)
        doc_type = resolved[0] if resolved else folder

        result = run_pipeline_bytes(
            raw,
            document_type=doc_type,
            expected_name=expected_name or "",
            filename=file_path.name,
            ocr_mode=config.ocr_mode,
            pp_ocr_tier=config.pp_ocr_tier,
            enable_preprocess=config.enable_preprocess,
            skip_passthrough=config.skip_passthrough,
        )

        bucket = by_folder[folder]
        pipeline_ok, matched = _process_result(result, bucket)
        _process_result(result, global_stats)

        row: dict[str, Any] = {
            "type": "result",
            "folder": folder,
            "file": file_path.name,
            "document_type": doc_type,
            "expected_name": expected_name,
            "pipeline_ok": pipeline_ok,
            "document_matched": matched,
            "timing": result.timing.as_dict(),
        }
        llm_fb = _llm_fallback_from_result(result)
        if llm_fb:
            row.update(llm_fb)
        fail_kind, fail_reason = _failure_detail(result)
        if fail_kind:
            row["failure_kind"] = fail_kind
            row["failure_reason"] = fail_reason
            row.update(_review_fields_from_result(result))
        elif llm_fb:
            row.update(_review_fields_from_result(result))
        if not result.ok:
            row["error"] = result.error
            row["error_kind"] = result.error_kind
        all_results.append(row)

        yield json.dumps(row, ensure_ascii=False) + "\n"

    by_folder_list = []
    for folder, bucket in sorted(by_folder.items()):
        doc_type, _ = folder_meta.get(folder, (None, None))
        by_folder_list.append(bucket.to_dict(folder=folder, document_type=doc_type))

    failures = [
        {
            "folder": r["folder"],
            "file": r["file"],
            "document_type": r.get("document_type"),
            "expected_name": r.get("expected_name"),
            "failure_kind": r.get("failure_kind"),
            "failure_reason": r.get("failure_reason") or r.get("error"),
            "ocr_text": r.get("ocr_text"),
            "pipeline_response": r.get("pipeline_response"),
            "llm_fallback": r.get("llm_fallback"),
            "llm_annotation": r.get("llm_annotation"),
            "llm_validation": r.get("llm_validation"),
            "document_matched": r.get("document_matched"),
        }
        for r in all_results
        if r.get("failure_kind")
    ]

    llm_fallback_runs = [
        {
            "folder": r["folder"],
            "file": r["file"],
            "document_type": r.get("document_type"),
            "expected_name": r.get("expected_name"),
            "document_matched": r.get("document_matched"),
            "llm_fallback": r.get("llm_fallback"),
            "llm_annotation": r.get("llm_annotation"),
            "llm_validation": r.get("llm_validation"),
            "ocr_text": r.get("ocr_text"),
            "pipeline_response": r.get("pipeline_response"),
        }
        for r in all_results
        if isinstance(r.get("llm_fallback"), dict) and r["llm_fallback"].get("attempted")
    ]
    llm_rescued = sum(
        1 for r in llm_fallback_runs if r.get("document_matched") is True
    )

    summary = {
        "type": "summary",
        "stats": {
            **global_stats.to_dict(),
            "by_folder": by_folder_list,
            "failures": failures,
            "llm_fallback_runs": llm_fallback_runs,
            "llm_fallback_attempted": len(llm_fallback_runs),
            "llm_fallback_success": sum(
                1 for r in llm_fallback_runs if r.get("llm_fallback", {}).get("success")
            ),
            "llm_fallback_rescued": llm_rescued,
        },
        "results": all_results,
        "config": {
            "ocr_mode": config.ocr_mode,
            "pp_ocr_tier": config.pp_ocr_tier,
            "use_expected_name": config.use_expected_name,
            "enable_preprocess": config.enable_preprocess,
            "skip_passthrough": config.skip_passthrough,
            "total_jobs": total_jobs,
            "specific_files": any(bool(s.files) for s in config.selections),
        },
    }
    yield json.dumps(summary, ensure_ascii=False) + "\n"
