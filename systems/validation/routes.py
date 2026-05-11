"""HTTP routes untuk validasi nama (fuzzy) hasil OCR vs input."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from systems.validation.document_profiles import list_supported_document_types, resolve_keywords
from systems.validation.fuzzy_compare import compare_ocr_name_to_expected, validate_document_ocr
from systems.observability.last_tuning_log import (
    log_safe_failure,
    summarize_validation_result,
    write_last_tuning_log,
)

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
    aggregate_min_pass_ratio: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimal rata-rata skor fuzzy keyword (0–1) terhadap OCR penuh: "
            "rata-rata best_score/100 tiap keyword profil (non-skip). Default 0.7."
        ),
    )
    identity_min_score: float = Field(
        65.0,
        ge=0.0,
        le=100.0,
        description=(
            "Minimal skor identitas 0–100: nama diekstrak dari OCR vs nama referensi, "
            "rata-rata(token_sort_ratio, WRatio, partial_ratio) pada dua string pendek. "
            "Hanya dipakai jika expected_name tidak kosong. Default 65."
        ),
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
        "validate_document_response": (
            "Respons validate-document menyertakan `explanation` (id): summary, detail_lines, hints, "
            "primary_blockers (DOCUMENT_TYPE | IDENTITY), gates."
        ),
        "validate_document_note": (
            "document_matched = document_type_pass AND (tanpa nama referensi | identity_pass). "
            "Default aggregate_min_pass_ratio=0.7, identity_min_score=65 (opsional di body JSON)."
        ),
        "last_tuning_log": {
            "file": "logs/last_tuning.json",
            "env_override": "LAST_TUNING_LOG_PATH",
        },
    }


@router.get("/")
def validation_page() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "index.html tidak ditemukan.")
    return FileResponse(page)


@router.post("/api/v1/compare-names")
def api_compare_names(body: CompareNamesBody) -> JSONResponse:
    path = "/systems/validation/api/v1/compare-names"
    sub = "validation_compare_names"
    if not body.ocr_text.strip() and not body.expected_name.strip():
        log_safe_failure(
            subsystem=sub,
            method="POST",
            path=path,
            http_status=400,
            detail="ocr_text dan expected_name tidak boleh keduanya kosong.",
        )
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
    write_last_tuning_log(
        {
            "success": True,
            "subsystem": sub,
            "method": "POST",
            "path": path,
            "request": {
                "threshold": body.threshold,
                "ocr_text_chars": len(body.ocr_text),
                "expected_name_chars": len(body.expected_name),
            },
            "result": {
                "matched": result.matched,
                "best_score": result.best_score,
                "scores": result.scores,
            },
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))


@router.post("/api/v1/validate-document")
def api_validate_document(body: ValidateDocumentBody) -> JSONResponse:
    path = "/systems/validation/api/v1/validate-document"
    sub = "validation_document"
    if not body.ocr_text.strip():
        log_safe_failure(
            subsystem=sub, method="POST", path=path, http_status=400, detail="ocr_text kosong."
        )
        raise HTTPException(status_code=400, detail="ocr_text tidak boleh kosong.")

    resolved = resolve_keywords(body.document_type)
    if not resolved:
        detail = {
            "message": "Jenis dokumen tidak dikenal.",
            "document_type": body.document_type,
            "supported": list_supported_document_types(),
        }
        log_safe_failure(
            subsystem=sub, method="POST", path=path, http_status=400, detail=detail
        )
        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    canonical_id, keywords = resolved
    detail = validate_document_ocr(
        body.ocr_text,
        document_type=body.document_type.strip(),
        document_profile_id=canonical_id,
        keywords=keywords,
        expected_name=body.expected_name,
        aggregate_min_pass_ratio=body.aggregate_min_pass_ratio,
        identity_min_score=body.identity_min_score,
    )
    payload = {
        "success": True,
        "document_profile_id": canonical_id,
        "keywords_from_profile": keywords,
        **detail,
    }
    write_last_tuning_log(
        {
            "success": True,
            "subsystem": sub,
            "method": "POST",
            "path": path,
            "request": {
                "document_type": body.document_type.strip(),
                "ocr_text_chars": len(body.ocr_text),
                "expected_name_chars": len(body.expected_name.strip()),
                "aggregate_min_pass_ratio": body.aggregate_min_pass_ratio,
                "identity_min_score": body.identity_min_score,
            },
            "result": summarize_validation_result(payload),
        }
    )
    return JSONResponse(content=jsonable_encoder(payload))
