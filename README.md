# OCR Platform

API preprocessing + OCR (PP-OCRv6) + validasi dokumen. **Cara utama: build di server** (linux/amd64 native). Jangan push image bergiga dari Mac ke Docker Hub.

## Install di server NVIDIA

```bash
git clone <repo> && cd ocr
# pertama kali (Paddle + dataset, sekali):
bash scripts/server_build.sh
docker compose -f deploy/docker-compose.yml up -d
```

Update kode (tidak download Paddle/dataset ulang):

```bash
git pull
bash scripts/server_build.sh
docker compose -f deploy/docker-compose.yml up -d
```

RTX 50 (sm_120): `PADDLE_GPU_INDEX=cu129 PADDLE_VERSION=3.3.1 REBUILD_BASE=1 bash scripts/server_build.sh` — Paddle 3.2.x tidak mendukung arsitektur 120.

## Opsional: Docker Hub (dari server, bukan dari Mac)

```bash
docker login
export DOCKERHUB_USER=codemysister
REBUILD_BASE=1 VARIANT=gpu bash scripts/docker_push.sh   # sekali
VARIANT=gpu bash scripts/docker_push.sh                    # harian
```

UI/API: `http://SERVER:8001/pipeline` — docs: `http://SERVER:8001/docs`

Inferensi pertama mengunduh model Paddle ke volume `paddle-models` (bisa beberapa menit).

## Lokal (masih pakai image)

```bash
docker compose up -d --build
```

Set `OCR_IMAGE=akunanda/ocr:latest` di `.env` jika ingin pull, bukan build.

## Catatan

- Tag `:latest` = **CPU**. Tag `:gpu` = **Paddle CUDA** (tidak perlu pip ulang di server).
- Dua tag: `:gpu-base` (Paddle + dataset, jarang) dan `:gpu` (kode, harian). Server cukup `docker pull …:gpu`; layer base tidak diunduh ulang.
- `REBUILD_BASE=1` hanya jika `requirements.txt`, wheel Paddle, atau `dataset/` berubah.
- Benchmark `/dataset-test` memakai `dataset/` di dalam image. Jangan mount volume kosong ke `/app/dataset`.
- Key opsional (Mistral, dll.): file `.env` di samping compose.
- CV search butuh OpenSearch: `docker compose -f docker-compose.cv.yml up -d` (dari repo).
