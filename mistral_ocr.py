#!/usr/bin/env python3
"""
Coba Mistral OCR dari CLI (berbayar, butuh MISTRAL_API_KEY).

Contoh:
  export MISTRAL_API_KEY="sk-..."
  .venv/bin/python mistral_ocr.py path/ke/gambar.jpg
  .venv/bin/python mistral_ocr.py path/ke/dokumen.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from systems.ocr.mistral_runner import run_mistral_ocr


def main() -> int:
    parser = argparse.ArgumentParser(description="Tes Mistral OCR (cloud API)")
    parser.add_argument("file", type=Path, help="Gambar atau PDF lokal")
    parser.add_argument(
        "--table-format",
        choices=("markdown", "html"),
        default=None,
        help="Format tabel terpisah (default: inline markdown)",
    )
    parser.add_argument(
        "--include-image-base64",
        action="store_true",
        help="Sertakan base64 gambar potongan di respons API",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Simpan respons penuh ke file JSON",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"File tidak ditemukan: {args.file}", file=sys.stderr)
        return 1

    raw = args.file.read_bytes()
    try:
        result = run_mistral_ocr(
            raw,
            table_format=args.table_format,
            include_image_base64=args.include_image_base64,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Gagal: {e}", file=sys.stderr)
        return 1

    print("\n==============================")
    print("TEKS / MARKDOWN")
    print("==============================\n")
    print(result.get("text") or "(kosong)")

    usage = result.get("usage") or {}
    timing = result.get("timing") or {}

    print("\n==============================")
    print("USAGE & BIAYA (estimasi)")
    print("==============================\n")
    print(f"Model            : {result.get('model')}")
    print(f"Halaman diproses : {usage.get('pages_processed')}")
    print(f"Ukuran file      : {usage.get('doc_size_bytes') or len(raw)} bytes")
    print(f"Waktu API        : {timing.get('api_call_s')}s")
    print(f"Total wall       : {timing.get('total_wall_s')}s")
    print(f"Biaya estimasi   : ${usage.get('estimated_cost_usd')} (~Rp {usage.get('estimated_cost_idr'):,.0f})")

    log_data = {
        "system": "ocr_platform",
        "provider": "mistral",
        "model": result.get("model"),
        "input_file": str(args.file),
        "input_bytes": len(raw),
        "usage": usage,
        "timing": timing,
        "response_characters": len(result.get("text") or ""),
        "created_at": datetime.now().isoformat(),
    }

    print("\n==============================")
    print("LOG JSON")
    print("==============================\n")
    print(json.dumps(log_data, indent=2))

    with open("ai_usage_log.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

    if args.save_json:
        args.save_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRespons penuh disimpan ke {args.save_json}")

    print("\nLog ditambahkan ke ai_usage_log.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
