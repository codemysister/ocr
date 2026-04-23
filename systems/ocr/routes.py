"""HTTP routes untuk sistem OCR."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter(prefix="/systems/ocr", tags=["ocr"])


@router.get("/health")
def ocr_health() -> dict:
    return {"status": "ok", "system": "ocr"}


@router.get("/")
def ocr_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
