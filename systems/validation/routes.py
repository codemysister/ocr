"""HTTP routes untuk validasi nama (fuzzy) hasil OCR vs input."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from systems.validation.document_profiles import list_supported_document_types, resolve_keywords
from systems.validation.fuzzy_compare import compare_ocr_name_to_expected, validate_document_ocr

STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter(prefix="/systems/validation", tags=["validation"])


class CompareNamesBody(BaseModel):
    ocr_text: str = Field(..., description="Teks / nama hasil OCR")
    expected_name: str = Field(..., description="Nama referensi yang diharapkan")
    threshold: float = Field(
        85.0,
        ge=0.0,
        le=100.0,
        description="Ambang skor 0–100; matched jika best_score >= threshold",
    )


class ValidateDocumentBody(BaseModel):
    ocr_text: str = Field(..., description="Teks penuh hasil OCR")
    document_type: str = Field(
        ...,
        min_length=1,
        description="Jenis dokumen, mis. KTP / NPWP — menentukan keyword di server",
    )
    expected_name: str = Field(
        "",
        description="Nama yang diharapkan; kosongkan jika hanya cek keyword profil",
    )
    threshold: float = Field(
        85.0,
        ge=0.0,
        le=100.0,
        description="Ambang fuzzy sama untuk tiap keyword dan nama",
    )

    @field_validator("document_type")
    @classmethod
    def document_type_not_blank(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("document_type tidak boleh kosong.")
        return s


@router.get("/health")
def validation_health() -> dict:
    return {
        "status": "ok",
        "system": "validation",
        "engine": "rapidfuzz",
        "document_profiles": list_supported_document_types(),
        "apis": [
            "/systems/validation/api/v1/compare-names",
            "/systems/validation/api/v1/validate-document",
        ],
    }


@router.get("/")
def validation_page() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "index.html tidak ditemukan.")
    return FileResponse(page)


@router.post("/api/v1/compare-names")
def api_compare_names(body: CompareNamesBody) -> JSONResponse:
    if not body.ocr_text.strip() and not body.expected_name.strip():
        raise HTTPException(
            status_code=400,
            detail="ocr_text dan expected_name tidak boleh keduanya kosong.",
        )

    result = compare_ocr_name_to_expected(
        body.ocr_text,
        body.expected_name,
        threshold=body.threshold,
    )
    payload = {
        "success": True,
        "threshold": body.threshold,
        **result.as_dict(),
    }
    return JSONResponse(content=jsonable_encoder(payload))


@router.post("/api/v1/validate-document")
def api_validate_document(body: ValidateDocumentBody) -> JSONResponse:
    if not body.ocr_text.strip():
        raise HTTPException(status_code=400, detail="ocr_text tidak boleh kosong.")

    resolved = resolve_keywords(body.document_type)
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Jenis dokumen tidak dikenal.",
                "document_type": body.document_type,
                "supported": list_supported_document_types(),
            },
        )

    canonical_id, keywords = resolved
    detail = validate_document_ocr(
        body.ocr_text,
        document_type=body.document_type.strip(),
        keywords=keywords,
        expected_name=body.expected_name,
        threshold=body.threshold,
    )
    payload = {
        "success": True,
        "document_profile_id": canonical_id,
        "keywords_from_profile": keywords,
        **detail,
    }
    return JSONResponse(content=jsonable_encoder(payload))
