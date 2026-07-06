"""HTTP routes untuk CV ingest + match."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from systems.cv.config import get_settings
from systems.cv.embedding.embedder import embedder_status
from systems.cv.index.opensearch_client import (
    OpenSearchUnavailableError,
    cluster_health,
    delete_by_doc_id,
    ensure_index,
    fetch_chunks_by_doc_id,
    is_opensearch_connection_error,
    list_documents,
    opensearch_unavailable_detail,
)
from systems.cv.ingest.pipeline import ingest_cv_bytes
from systems.cv.match.cv_matcher import match_cv_chunks
from systems.cv.models import CvMatchRequest, CvMatchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/systems/cv", tags=["cv"])


def _match_response(result: dict, *, doc_id: str | None = None) -> CvMatchResponse:
    dims = result.get("dimensions") or {}
    return CvMatchResponse(
        matched=bool(result.get("matched")),
        overall_percent=float(result.get("overall_percent") or 0.0),
        summary=str(result.get("summary") or ""),
        dimensions=dims,
        doc_id=doc_id,
    )


@router.get("/health")
def cv_health() -> dict:
    settings = get_settings()
    os_ok = False
    os_detail: dict = {}
    try:
        os_detail = cluster_health()
        os_ok = os_detail.get("status") in ("green", "yellow")
        ensure_index()
    except OpenSearchUnavailableError as e:
        os_detail = opensearch_unavailable_detail(exc=e)
    except Exception as e:
        if is_opensearch_connection_error(e):
            os_detail = opensearch_unavailable_detail(exc=e)
        else:
            os_detail = {"error": str(e)}

    return {
        "status": "ok" if os_ok else "degraded",
        "system": "cv",
        "opensearch": {"ok": os_ok, **os_detail},
        "index": settings.opensearch_index,
        "embedder": embedder_status(),
        "match_dimensions": ["nama", "pendidikan", "pengalaman"],
        "apis": [
            "/systems/cv/api/v1/ingest",
            "/systems/cv/api/v1/match",
            "/systems/cv/api/v1/documents",
        ],
    }


@router.post("/api/v1/ingest")
async def api_ingest_cv(
    file: UploadFile = File(..., description="CV: PDF, DOCX, MD, gambar"),
    expected_name: str = Form(""),
    education_query: str = Form(""),
    experience_query: str = Form(""),
) -> JSONResponse:
    raw = await file.read()
    filename = file.filename or "cv"
    try:
        result = ingest_cv_bytes(
            raw,
            filename=filename,
            expected_name=expected_name.strip(),
            education_query=education_query.strip(),
            experience_query=experience_query.strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OpenSearchUnavailableError as e:
        raise HTTPException(status_code=503, detail=opensearch_unavailable_detail(exc=e)) from e
    except Exception as e:
        if is_opensearch_connection_error(e):
            raise HTTPException(status_code=503, detail=opensearch_unavailable_detail(exc=e)) from e
        logger.exception("CV ingest failed for %s", filename)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(content=jsonable_encoder(result))


@router.post("/api/v1/match", response_model=CvMatchResponse)
def api_match_cv(body: CvMatchRequest) -> CvMatchResponse:
    if not body.doc_id:
        raise HTTPException(status_code=400, detail="doc_id wajib untuk match CV terindeks.")
    try:
        chunks = fetch_chunks_by_doc_id(body.doc_id)
    except OpenSearchUnavailableError as e:
        raise HTTPException(status_code=503, detail=opensearch_unavailable_detail(exc=e)) from e
    except Exception as e:
        if is_opensearch_connection_error(e):
            raise HTTPException(status_code=503, detail=opensearch_unavailable_detail(exc=e)) from e
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not chunks:
        raise HTTPException(status_code=404, detail=f"Tidak ada chunk untuk doc_id={body.doc_id}")

    result = match_cv_chunks(
        chunks,
        expected_name=body.expected_name.strip(),
        education_query=body.education_query.strip(),
        experience_query=body.experience_query.strip(),
    )
    return _match_response(result, doc_id=body.doc_id)


@router.get("/api/v1/documents")
def api_list_cv_documents() -> dict:
    try:
        docs = list_documents()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"documents": docs, "count": len(docs)}


@router.delete("/api/v1/documents/{doc_id}")
def api_delete_cv(doc_id: str) -> dict:
    try:
        deleted = delete_by_doc_id(doc_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"doc_id": doc_id, "chunks_deleted": deleted}
