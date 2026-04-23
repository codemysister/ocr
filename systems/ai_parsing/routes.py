"""HTTP routes untuk sistem AI parsing (opsional)."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter(prefix="/systems/ai-parsing", tags=["ai-parsing"])


@router.get("/health")
def ai_parsing_health() -> dict:
    return {"status": "ok", "system": "ai-parsing", "optional": True}


@router.get("/")
def ai_parsing_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
