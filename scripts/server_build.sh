#!/usr/bin/env bash
# Build image di server Linux (native amd64). Tidak lewat Docker Hub.
#
# Pertama / ganti Paddle atau dataset:
#   REBUILD_BASE=1 bash scripts/server_build.sh
#
# Harian (kode saja):
#   git pull && bash scripts/server_build.sh
#   docker compose -f deploy/docker-compose.yml up -d
#
# RTX 50 (sm_120): PADDLE_GPU_INDEX=cu129 PADDLE_VERSION=3.3.1 REBUILD_BASE=1 bash scripts/server_build.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VARIANT="${VARIANT:-gpu}"
PADDLE_GPU_INDEX="${PADDLE_GPU_INDEX:-cu126}"
PADDLE_VERSION="${PADDLE_VERSION:-3.2.0}"
REBUILD_BASE="${REBUILD_BASE:-0}"
APP_IMAGE="${OCR_IMAGE:-ocr:gpu}"
BASE_IMAGE="${OCR_BASE_IMAGE:-ocr:gpu-base}"

if [[ "$VARIANT" != "gpu" ]]; then
  APP_IMAGE="${OCR_IMAGE:-ocr:latest}"
  BASE_IMAGE="${OCR_BASE_IMAGE:-ocr:latest-base}"
  BASE_DOCKERFILE="Dockerfile.base"
  APP_DOCKERFILE="Dockerfile"
  BUILD_ARGS=()
else
  BASE_DOCKERFILE="Dockerfile.gpu.base"
  APP_DOCKERFILE="Dockerfile.gpu"
  BUILD_ARGS=(
    --build-arg "PADDLE_GPU_INDEX=${PADDLE_GPU_INDEX}"
    --build-arg "PADDLE_VERSION=${PADDLE_VERSION}"
  )
fi

need_base=0
if [[ "$REBUILD_BASE" == "1" ]]; then
  need_base=1
elif ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Base ${BASE_IMAGE} belum ada di server — build sekali (Paddle + dataset)."
  need_base=1
fi

if [[ "$need_base" == "1" ]]; then
  echo "Building ${BASE_IMAGE} (${BASE_DOCKERFILE}) ..."
  docker build -f "$BASE_DOCKERFILE" "${BUILD_ARGS[@]}" -t "$BASE_IMAGE" .
fi

echo "Building ${APP_IMAGE} (${APP_DOCKERFILE}, kode saja) ..."
docker build -f "$APP_DOCKERFILE" --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "$APP_IMAGE" .

echo
echo "Siap: ${APP_IMAGE}"
echo "  docker compose -f deploy/docker-compose.yml up -d"
