#!/usr/bin/env bash
# Push image ke Docker Hub.
#
# Harian (kode saja, cepat):
#   DOCKERHUB_USER=codemysister VARIANT=gpu bash scripts/docker_push.sh
#
# Sekali / saat ganti requirements, Paddle, atau dataset:
#   REBUILD_BASE=1 VARIANT=gpu bash scripts/docker_push.sh
#
# CPU:  bash scripts/docker_push.sh
# RTX 50: VARIANT=gpu PADDLE_GPU_INDEX=cu129 REBUILD_BASE=1 bash scripts/docker_push.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${DOCKERHUB_USER:?Set DOCKERHUB_USER (akun Docker Hub), contoh: export DOCKERHUB_USER=codemysister}"

REPO="${DOCKERHUB_REPO:-ocr}"
VARIANT="${VARIANT:-cpu}"
PADDLE_GPU_INDEX="${PADDLE_GPU_INDEX:-cu126}"
REBUILD_BASE="${REBUILD_BASE:-0}"
IMAGE_NAME="${DOCKERHUB_USER}/${REPO}"

if [[ "$VARIANT" == "gpu" ]]; then
  APP_DOCKERFILE="Dockerfile.gpu"
  BASE_DOCKERFILE="Dockerfile.gpu.base"
  TAG="${TAG:-gpu}"
  BASE_TAG="${BASE_TAG:-gpu-base}"
  BUILD_ARGS=(--build-arg "PADDLE_GPU_INDEX=${PADDLE_GPU_INDEX}")
else
  APP_DOCKERFILE="Dockerfile"
  BASE_DOCKERFILE="Dockerfile.base"
  TAG="${TAG:-latest}"
  BASE_TAG="${BASE_TAG:-latest-base}"
  BUILD_ARGS=()
fi

APP_IMAGE="${IMAGE_NAME}:${TAG}"
BASE_IMAGE="${IMAGE_NAME}:${BASE_TAG}"

buildx_extra=()
if docker buildx build --help 2>&1 | grep -q -- '--provenance'; then
  buildx_extra+=(--provenance=false)
fi
if docker buildx build --help 2>&1 | grep -q -- '--sbom'; then
  buildx_extra+=(--sbom=false)
fi

ignorefile_app=()
if docker buildx build --help 2>&1 | grep -q -- '--ignorefile' && [[ -f .dockerignore.app ]]; then
  ignorefile_app=(--ignorefile .dockerignore.app)
fi

push_image() {
  local dockerfile="$1" image="$2"
  shift 2
  local extra=("$@")
  echo "Building ${image} (linux/amd64, ${dockerfile}) ..."
  if docker buildx version >/dev/null 2>&1; then
    docker buildx build \
      --platform linux/amd64 \
      -f "$dockerfile" \
      "${buildx_extra[@]}" \
      "${extra[@]}" \
      -t "$image" \
      --push \
      .
  else
    echo "buildx tidak ada — docker build + push (platform host)." >&2
    docker build -f "$dockerfile" "${extra[@]}" -t "$image" .
    docker push "$image"
  fi
}

if [[ "$REBUILD_BASE" == "1" ]]; then
  echo "REBUILD_BASE=1 — push ${BASE_IMAGE} (Paddle/pip + dataset, lama) ..."
  push_image "$BASE_DOCKERFILE" "$BASE_IMAGE" "${BUILD_ARGS[@]}"
else
  echo "Memakai base ${BASE_IMAGE} (tidak rebuild Paddle/dataset)."
  echo "Kalau base belum ada di Hub: REBUILD_BASE=1 VARIANT=${VARIANT} bash scripts/docker_push.sh"
  docker pull --platform linux/amd64 "$BASE_IMAGE"
fi

push_image "$APP_DOCKERFILE" "$APP_IMAGE" \
  "${ignorefile_app[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"

echo "Pushed ${APP_IMAGE} (FROM ${BASE_IMAGE})"
echo
if [[ "$VARIANT" == "gpu" ]]; then
  echo "Server (NVIDIA):"
  echo "  docker pull ${APP_IMAGE}"
  echo "  docker run -d --name checkinpro-ocr --gpus all -p 8001:8001 --restart unless-stopped ${APP_IMAGE}"
else
  echo "Server (CPU):"
  echo "  docker pull ${APP_IMAGE}"
  echo "  docker run -d --name checkinpro-ocr -p 8001:8001 --restart unless-stopped ${APP_IMAGE}"
fi
