"""Skema & parser document_annotation Mistral OCR untuk dokumen identitas Indonesia."""

from __future__ import annotations

import json
from typing import Any

PLACEHOLDER_HOLDER_NAMES = frozenset(
    {
        "",
        "-",
        "n/a",
        "na",
        "unknown",
        "not specified",
        "not applicable",
        "tidak ada",
        "tidak diketahui",
        "tidak tersedia",
    }
)

DOCUMENT_ANNOTATION_PROMPT = (
    "Extract from this Indonesian identity or tax document: document type label "
    "(e.g. KTP, NPWP, Kartu Keluarga), the full legal name of the document holder "
    "(person only — no labels like 'Wajib Pajak', 'Nama', or 'Halo'), and identity or "
    "tax number (NIK, NPWP, NITKU) if visible."
)

_SCHEMA_DEFINITION: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type_label": {
            "type": "string",
            "description": "Human-readable document type, e.g. KTP, NPWP, Kartu Keluarga",
        },
        "holder_name": {
            "type": "string",
            "description": "Full legal name of the document holder (person only)",
        },
        "identity_number": {
            "type": "string",
            "description": "NIK, NPWP, or other ID/tax number if visible",
        },
    },
    "required": ["holder_name"],
    "additionalProperties": False,
}


def build_document_annotation_format() -> Any:
    from mistralai.client.models import JSONSchema, ResponseFormat

    schema = JSONSchema(
        name="id_document",
        schema_definition=_SCHEMA_DEFINITION,
        strict=True,
    )
    return ResponseFormat(type="json_schema", json_schema=schema)


def parse_document_annotation(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    return raw


def holder_name_from_annotation(ann: dict[str, Any] | None) -> str | None:
    if not ann:
        return None
    name = (ann.get("holder_name") or "").strip()
    if not name or name.casefold() in PLACEHOLDER_HOLDER_NAMES:
        return None
    return name


_DOCUMENT_TYPE_TO_PROFILE: dict[str, str] = {
    "ktp": "ktp",
    "kartu tanda penduduk": "ktp",
    "kartu tandapenduduk": "ktp",
    "e-ktp": "ktp",
    "e ktp": "ktp",
    "identitas": "ktp",
    "npwp": "npwp",
    "nomor pokok wajib pajak": "npwp",
    "kartu keluarga": "kk",
    "kk": "kk",
    "rekening": "rekening",
    "rekening koran": "rekening",
    "rekening tabungan": "rekening",
    "mutasi": "mutasi",
    "mutasi rekening": "mutasi",
    "e-statement": "mutasi",
    "e statement": "mutasi",
    "skck": "skck",
    "surat keterangan catatan kepolisian": "skck",
}


def document_type_profile_from_annotation(ann: dict[str, Any] | None) -> str | None:
    """Petakan document_type_label Mistral ke id profil kanonik (ktp/npwp/kk)."""
    if not ann:
        return None
    label = (ann.get("document_type_label") or "").strip().casefold()
    if not label:
        return None
    spaced = " ".join(label.split())
    if spaced in _DOCUMENT_TYPE_TO_PROFILE:
        return _DOCUMENT_TYPE_TO_PROFILE[spaced]
    compact = spaced.replace(" ", "")
    for key, pid in _DOCUMENT_TYPE_TO_PROFILE.items():
        if compact == key.replace(" ", ""):
            return pid
    if "ktp" in spaced or "tanda penduduk" in spaced:
        return "ktp"
    if "npwp" in spaced or "wajib pajak" in spaced:
        return "npwp"
    if "kartu keluarga" in spaced or spaced == "kk":
        return "kk"
    if "mutasi" in spaced or "e-statement" in spaced or "e statement" in spaced:
        return "mutasi"
    if "rekening" in spaced or "tabungan" in spaced:
        return "rekening"
    if "skck" in spaced or "kepolisian" in spaced or "catatan kepolisian" in spaced:
        return "skck"
    return None
