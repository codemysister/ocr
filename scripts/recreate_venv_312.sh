#!/usr/bin/env bash
# Buat ulang .venv dengan Python 3.12 saja (macOS + Homebrew).
# Dipakai jika `python` di venv masih menaut ke 3.14 padahal Anda ingin Paddle (wheel 3.12).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY312="${PY312:-$(command -v python3.12 || true)}"
if [[ -z "$PY312" ]]; then
  echo "python3.12 tidak di PATH. Pasang: brew install python@3.12" >&2
  exit 1
fi

echo "Menggunakan: $PY312 — $($PY312 --version)"
rm -rf .venv
"$PY312" -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
echo "Cek:"
.venv/bin/python -c "import sys; print('exe:', sys.executable); import paddle; print('paddle:', paddle.__version__)"
.venv/bin/python scripts/verify_ocr_deps.py
