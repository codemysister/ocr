"""Pydantic models untuk API CV."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CvMatchRequest(BaseModel):
    expected_name: str = Field("", description="Nama referensi untuk dicek")
    education_query: str = Field(
        "",
        description="Opsional: kata kunci tambahan di section pendidikan",
    )
    experience_query: str = Field(
        "",
        description="Opsional: kata kunci tambahan di section pengalaman",
    )
    doc_id: str | None = Field(None, description="CV terindeks (wajib jika tanpa ingest baru)")


class CvMatchResponse(BaseModel):
    matched: bool
    overall_percent: float
    summary: str
    dimensions: dict[str, Any]
    doc_id: str | None = None
