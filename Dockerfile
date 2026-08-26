# Image CPU harian: hanya kode. Base (pip + dataset) sudah di Hub.
# Push cepat: bash scripts/docker_push.sh
# Rebuild pip/dataset: REBUILD_BASE=1 bash scripts/docker_push.sh

ARG BASE_IMAGE=codemysister/ocr:latest-base
FROM ${BASE_IMAGE}

COPY app ./app
COPY systems ./systems
COPY scripts/verify_ocr_deps.py ./scripts/verify_ocr_deps.py

CMD ["sh", "-c", "exec uvicorn app.main:app --host ${HOST} --port ${PORT}"]
