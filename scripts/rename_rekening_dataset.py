#!/usr/bin/env python3
"""Rapikan nama file dataset rekening ke pola `rekening bank_{Nama}_{NoRek}.ext`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REKENING_DIR = ROOT / "dataset" / "rekening bank"
FOLDER_PREFIX = "rekening bank_"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
_ACCOUNT_TAIL_RE = re.compile(r"_(\d{10,13})$")
_TRAILING_NIK_RE = re.compile(r"_?\d{16}$")


def parse_rekening_parts(stem: str) -> tuple[str, str] | None:
    if not stem.startswith(FOLDER_PREFIX):
        return None
    body = stem[len(FOLDER_PREFIX) :]
    m = _ACCOUNT_TAIL_RE.search(body)
    if not m:
        return None
    acct = m.group(1)
    name = body[: m.start()]
    name = _TRAILING_NIK_RE.sub("", name).strip().rstrip("_")
    if not name or len(acct) < 10:
        return None
    return name, acct


def target_name(name: str, acct: str, suffix: str) -> str:
    return f"{FOLDER_PREFIX}{name}_{acct}{suffix}"


def main() -> int:
    if not REKENING_DIR.is_dir():
        print(f"folder tidak ditemukan: {REKENING_DIR}", file=sys.stderr)
        return 1

    renamed = 0
    skipped = 0
    errors: list[str] = []

    for path in sorted(REKENING_DIR.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in _EXTS:
            continue
        parsed = parse_rekening_parts(path.stem)
        if not parsed:
            errors.append(f"pola tidak dikenal: {path.name}")
            continue
        name, acct = parsed
        new_name = target_name(name, acct, path.suffix)
        if path.name == new_name:
            skipped += 1
            continue
        target = path.with_name(new_name)
        if target.exists() and target != path:
            errors.append(f"target sudah ada: {new_name}")
            continue
        path.rename(target)
        renamed += 1

    print(f"renamed={renamed} skipped={skipped} errors={len(errors)}")
    for err in errors[:20]:
        print(f"  - {err}")
    if len(errors) > 20:
        print(f"  ... dan {len(errors) - 20} error lainnya")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
