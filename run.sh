#!/usr/bin/env bash
# Entry point dari root repo: setup, install, jalankan ulang server.
# Semua opsi diteruskan ke scripts/dev_server.sh (lihat bash scripts/dev_server.sh --help).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/scripts/dev_server.sh" "$@"
