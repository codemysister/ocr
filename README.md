# OCR Platform

API preprocessing + OCR (PP-OCRv6) + validasi dokumen. **Cara utama: Docker image** (Docker Hub), tanpa Python/venv di server.

## Push ke Docker Hub

```bash
docker login
export DOCKERHUB_USER=akunanda          # ganti dengan akun Hub Anda
bash scripts/docker_push.sh             # image: akunanda/ocr:latest (linux/amd64)
```

Tag lain: `TAG=v1.0.0 bash scripts/docker_push.sh`

## Install di server

Tidak perlu clone repo. Setelah image ada di Hub:

```bash
docker pull akunanda/ocr:latest
docker run -d \
  --name checkinpro-ocr \
  -p 8001:8001 \
  --restart unless-stopped \
  -v paddle-models:/root/.paddlex \
  akunanda/ocr:latest
```

Atau salin `deploy/docker-compose.yml` ke server:

```bash
export OCR_IMAGE=akunanda/ocr:latest
docker compose -f docker-compose.yml pull
docker compose -f docker-compose.yml up -d
```

UI/API: `http://SERVER:8001/pipeline` — docs: `http://SERVER:8001/docs`

Inferensi pertama mengunduh model Paddle ke volume `paddle-models` (bisa beberapa menit).

## Lokal (masih pakai image)

```bash
docker compose up -d --build
```

Set `OCR_IMAGE=akunanda/ocr:latest` di `.env` jika ingin pull, bukan build.

## Catatan

- Image default **CPU PaddlePaddle**. `OCR_DEVICE=gpu` akan fallback ke CPU jika CUDA tidak ada di container.
- Benchmark `/dataset-test` butuh folder `dataset/` di host; `docker compose` mem-mount-nya ke `/app/dataset`.
- Key opsional (Mistral, dll.): file `.env` di samping compose.
- CV search butuh OpenSearch: `docker compose -f docker-compose.cv.yml up -d` (dari repo).
