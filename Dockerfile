# OCR platform — image production (linux/amd64) untuk Docker Hub / server.
# Build & push: bash scripts/docker_push.sh
# Server:       docker compose -f deploy/docker-compose.yml up -d

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-mistral.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt -r requirements-mistral.txt

COPY app ./app
COPY systems ./systems
COPY scripts/verify_ocr_deps.py ./scripts/verify_ocr_deps.py

RUN python scripts/verify_ocr_deps.py

ENV HOST=0.0.0.0 \
    PORT=8001 \
    OCR_DEVICE=gpu \
    OCR_FAST_LANG=latin \
    OCR_FAST_DOC_UNWARPING=1 \
    PREPROCESS_AUTO_ROTATE_QUARTERS=auto \
    PREPROCESS_PROJECTION_DESKEW=1 \
    PREPROCESS_FULL_BLEED_STRAIGHTEN=1 \
    PREPROCESS_CARD_WARP=1 \
    PREPROCESS_MIN_SIDE_TARGET=900

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf "http://127.0.0.1:${PORT}/systems/ocr/health" | grep -q '"paddlepaddle_importable": true' || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host ${HOST} --port ${PORT}"]
