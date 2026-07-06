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


def _list_files_for_folder(folder_path: Path, limit: int) -> list[Path]:
    files = sorted(
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.casefold() in _DATASET_EXTS
    )
    cap = min(max(limit, 0), _MAX_LIMIT_PER_TYPE)
    return files[:cap]


@dataclass
class BenchmarkSelection:
    folder: str
    enabled: bool = True
    limit: int = 10


@dataclass
class BenchmarkConfig:
    selections: list[BenchmarkSelection]
    ocr_mode: OcrMode = "fast"
    pp_ocr_tier: str = "medium"
    use_expected_name: bool = True


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
        if not sel.enabled or sel.limit <= 0:
            continue
        folder_path = root / sel.folder
        if not folder_path.is_dir():
            yield json.dumps(
                {
                    "type": "error",
                    "folder": sel.folder,
                    "message": f"Folder tidak ditemukan: {sel.folder}",
                },
                ensure_ascii=False,
            ) + "\n"
            continue
        resolved = resolve_keywords(sel.folder)
        doc_type = resolved[0] if resolved else None
        folder_meta[sel.folder] = (doc_type, profile_label(doc_type) if doc_type else None)
        for fp in _list_files_for_folder(folder_path, sel.limit):
            expected = None
            if config.use_expected_name:
                expected = parse_dataset_filename(sel.folder, fp.name)
            jobs.append((sel.folder, fp, expected))

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
        if not result.ok:
            row["error"] = result.error
            row["error_kind"] = result.error_kind
        all_results.append(row)

        yield json.dumps(row, ensure_ascii=False) + "\n"

    by_folder_list = []
    for folder, bucket in sorted(by_folder.items()):
        doc_type, _ = folder_meta.get(folder, (None, None))
        by_folder_list.append(bucket.to_dict(folder=folder, document_type=doc_type))

    summary = {
        "type": "summary",
        "stats": {
            **global_stats.to_dict(),
            "by_folder": by_folder_list,
        },
        "results": all_results,
        "config": {
            "ocr_mode": config.ocr_mode,
            "pp_ocr_tier": config.pp_ocr_tier,
            "use_expected_name": config.use_expected_name,
            "total_jobs": total_jobs,
        },
    }
    yield json.dumps(summary, ensure_ascii=False) + "\n"
