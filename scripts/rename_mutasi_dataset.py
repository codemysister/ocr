#!/usr/bin/env python3
"""Rapikan nama file dataset mutasi ke pola `mutasi rekening_{Nama}_{NoRek}.ext`."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUTASI_DIR = ROOT / "dataset" / "mutasi rekening"
MANIFEST = ROOT / "dataset" / "manifest.json"
FOLDER_PREFIX = "mutasi rekening_"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
_ACCOUNT_TAIL_RE = re.compile(r"_(\d{10,13})$")
_TRAILING_NIK_RE = re.compile(r"_?\d{16}$")
_NIK_TAIL_RE = re.compile(r"(\d{16})$")


def load_accounts_by_nik() -> dict[str, str]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for emp in data.get("employees") or []:
        if not isinstance(emp, dict):
            continue
        nik = re.sub(r"\D", "", str(emp.get("nik") or emp.get("nomor_ktp") or ""))
        acct = re.sub(r"\D", "", str(emp.get("nomor_rekening_bank") or ""))
        if len(nik) == 16 and len(acct) >= 10:
            out[nik] = acct
    return out


def parse_parts(stem: str, by_nik: dict[str, str]) -> tuple[str, str] | None:
    if not stem.startswith(FOLDER_PREFIX):
        return None
    body = stem[len(FOLDER_PREFIX) :]

    m_acct = _ACCOUNT_TAIL_RE.search(body)
    if m_acct:
        acct = m_acct.group(1)
        name = body[: m.start()]
        name = _TRAILING_NIK_RE.sub("", name).strip().rstrip("_")
        if name and len(acct) >= 10:
            return name, acct

    m_nik = _NIK_TAIL_RE.search(body)
    if not m_nik:
        return None
    nik = m_nik.group(1)
    acct = by_nik.get(nik)
    if not acct:
        return None
    name = body[: m_nik.start()].strip().rstrip("_")
    if not name:
        return None
    return name, acct


def target_name(name: str, acct: str, suffix: str) -> str:
    return f"{FOLDER_PREFIX}{name}_{acct}{suffix}"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"manifest tidak ditemukan: {MANIFEST}", file=sys.stderr)
        return 1
    if not MUTASI_DIR.is_dir():
        print(f"folder tidak ditemukan: {MUTASI_DIR}", file=sys.stderr)
        return 1

    by_nik = load_accounts_by_nik()
    renamed = 0
    skipped = 0
    errors: list[str] = []

    for path in sorted(MUTASI_DIR.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in _EXTS:
            continue
        parsed = parse_parts(path.stem, by_nik)
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
