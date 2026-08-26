#!/usr/bin/env bash
# Build image linux/amd64 lalu push ke Docker Hub.
#
#   docker login
#   export DOCKERHUB_USER=akunanda
#   bash scripts/docker_push.sh
#
# Opsional: DOCKERHUB_REPO=ocr  TAG=v1.0.0

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${DOCKERHUB_USER:?Set DOCKERHUB_USER (akun Docker Hub), contoh: export DOCKERHUB_USER=akunanda}"

REPO="${DOCKERHUB_REPO:-ocr}"
TAG="${TAG:-latest}"
IMAGE_NAME="${DOCKERHUB_USER}/${REPO}"
FULL="${IMAGE_NAME}:${TAG}"

echo "Building ${FULL} (linux/amd64) ..."

if docker buildx version >/dev/null 2>&1; then
  if [[ "$TAG" != "latest" ]]; then
    docker buildx build \
      --platform linux/amd64 \
      -t "$FULL" \
      -t "${IMAGE_NAME}:latest" \
      --push \
      .
  else
    docker buildx build \
      --platform linux/amd64 \
      -t "$FULL" \
      --push \
      .
  fi
else
  echo "buildx tidak ada — docker build + push (platform host)." >&2
  docker build -t "$FULL" .
  docker push "$FULL"
  if [[ "$TAG" != "latest" ]]; then
    docker tag "$FULL" "${IMAGE_NAME}:latest"
    docker push "${IMAGE_NAME}:latest"
  fi
fi

echo "Pushed ${FULL}"
echo
echo "Install di server:"
echo "  docker run -d --name checkinpro-ocr -p 8001:8001 --restart unless-stopped ${FULL}"
echo "  # atau salin deploy/docker-compose.yml lalu:"
echo "  OCR_IMAGE=${FULL} docker compose -f docker-compose.yml up -d"
