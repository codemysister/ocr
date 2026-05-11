#!/usr/bin/env bash
# Setup + install + hentikan port lama + jalankan uvicorn (satu alur untuk pengembangan).
#
# Dipanggil dari root: bash scripts/dev_server.sh
# Atau: ./run.sh
#
# Opsi:
#   --no-install     Lewati pip (restart cepat).
#   --install-only   Hanya venv + pip, tanpa menjalankan server.
#   --no-kill        Jangan hentikan proses di PORT.
#   -h, --help       Bantuan singkat.
#
# Lingkungan:
#   PORT=8001              Port HTTP.
#   HOST=0.0.0.0           Bind host.
#   PYTHON=python3.12      Untuk membuat .venv jika belum ada.
#   INSTALL_REALESRGAN=1  Juga: pip install -r requirements-realesrgan.txt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
NO_INSTALL=0
INSTALL_ONLY=0
NO_KILL=0

for arg in "$@"; do
  case "$arg" in
    --no-install) NO_INSTALL=1 ;;
    --install-only) INSTALL_ONLY=1 ;;
    --no-kill) NO_KILL=1 ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
  esac
done

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    command -v "$PYTHON" >/dev/null && { echo "$PYTHON"; return; }
    echo "PYTHON=$PYTHON tidak ditemukan di PATH." >&2
    exit 1
  fi
  if command -v python3.12 >/dev/null 2>&1; then echo "python3.12"; return; fi
  if command -v python3 >/dev/null 2>&1; then echo "python3"; return; fi
  echo "Tidak ada python3.12 atau python3 di PATH." >&2
  exit 1
}

ensure_venv() {
  if [[ -x .venv/bin/python ]]; then
    return
  fi
  local py
  py="$(pick_python)"
  echo "[1/4] Membuat .venv dengan: $py ($($py --version 2>&1))"
  "$py" -m venv .venv
  .venv/bin/pip install -U pip setuptools wheel
}

kill_port() {
  if [[ "$NO_KILL" -eq 1 ]]; then
    echo "[3/4] Melewati hentikan port (--no-kill)."
    return
  fi
  if ! command -v lsof >/dev/null 2>&1; then
    echo "[3/4] lsof tidak ada; lewati hentikan port (pasang lsof atau set NO_KILL=1)." >&2
    return
  fi
  local pids
  pids="$(lsof -ti "TCP:${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "[3/4] Tidak ada proses di port ${PORT}."
    return
  fi
  echo "[3/4] Menghentikan proses di port ${PORT}: ${pids//$'\n'/ }"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -ti "TCP:${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "      Paksa hentikan: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 0.5
  fi
}

echo "=== OCR platform — dev server ==="
echo "Root: $ROOT"

ensure_venv

if [[ "$NO_INSTALL" -eq 0 ]]; then
  echo "[2/4] pip install -r requirements.txt …"
  .venv/bin/pip install -r requirements.txt
  if [[ "${INSTALL_REALESRGAN:-0}" == "1" ]]; then
    echo "      pip install -r requirements-realesrgan.txt …"
    .venv/bin/pip install -r requirements-realesrgan.txt
  fi
else
  echo "[2/4] Melewati pip (--no-install)."
fi

if [[ "$INSTALL_ONLY" -eq 1 ]]; then
  echo "Selesai (--install-only)."
  exit 0
fi

kill_port

echo "[4/4] uvicorn app.main:app --reload --host ${HOST} --port ${PORT}"
echo "      Buka: http://127.0.0.1:${PORT}/"
exec .venv/bin/uvicorn app.main:app --reload --host "${HOST}" --port "${PORT}"
