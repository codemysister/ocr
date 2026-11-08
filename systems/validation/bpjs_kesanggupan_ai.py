"""Validasi surat kesanggupan BPJS Kesehatan via Ornith.

Materai lolos jika AI melihat stempel fisik, atau detektor warna menemukan blob merah.
Label kotak "Materai Rp10.000" tanpa stempel tidak lolos.
"""

from __future__ import annotations

from typing import Any

from systems.validation.bpjs_kesanggupan_stamp import image_has_red_meterai
from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict
from systems.validation.ornith_call import as_bool, decide_identity, run_ornith_json
from systems.validation.user_errors import attach_user_errors

KESANGGUPAN_SYSTEM_PROMPT = """Kamu membaca surat pernyataan kesanggupan menanggung biaya BPJS Kesehatan.
Jangan mengarang.

Kembalikan HANYA satu objek JSON.
- is_surat_kesanggupan (boolean): true hanya jika ini surat pernyataan kesanggupan
  menanggung biaya BPJS Kesehatan (tidak aktif / menanggung pribadi / syarat bekerja).
  false jika kartu KIS, BPJS Ketenagakerjaan, info iuran, atau dokumen lain.
- holder_name (string): nama lengkap peserta / yang bertanda tangan, apa adanya.
- has_physical_stamp (boolean): true hanya jika ada stempel materai fisik
  (tempel merah, keemasan, atau e-meterai bergambar) di area tanda tangan.
  false jika hanya kotak tulisan "Materai Rp10.000" tanpa stempel.
- confidence (number 0-100)
- notes (string): maksimal 12 kata
"""


def bpjs_kesanggupan_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    document_type: str = "bpjs_kesanggupan",
    expected_name: str = "",
    image_bytes: bytes | None = None,
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = "bpjs_kesanggupan"
    is_doc = as_bool(parsed.get("is_surat_kesanggupan"))
    holder = str(parsed.get("holder_name") or "").strip()
    ident = decide_identity(expected_name=expected_name, holder=holder, use_nik=False)
    identity_pass = ident["identity_pass"]
    visual_stamp = False
    if image_bytes:
        try:
            visual_stamp = bool(image_has_red_meterai(image_bytes))
        except Exception:
            visual_stamp = False
    ai_stamp = as_bool(parsed.get("has_physical_stamp"))
    stamp_pass: bool | None = (ai_stamp or visual_stamp) if is_doc else None
    matched = is_doc and identity_pass is not False and stamp_pass is not False
    blockers: list[str] = []
    detail_lines: list[str] = []
    if is_doc:
        detail_lines.append("Gambar dikenali sebagai surat kesanggupan BPJS Kesehatan.")
    else:
        blockers.append("DOCUMENT_TYPE")
        detail_lines.append("Gambar bukan surat kesanggupan BPJS Kesehatan.")
    if ident["name_pass"] is True:
        detail_lines.append(f"Nama cocok: {holder}.")
    elif identity_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append(f"Nama tidak cocok: {holder or 'tidak terbaca'}.")
    if stamp_pass is True:
        detail_lines.append("Materai terdeteksi.")
    elif stamp_pass is False:
        blockers.insert(0, "BPJS_KESANGGUPAN_STAMP")
        detail_lines.insert(0, "Surat kesanggupan BPJS wajib memakai materai merah.")
    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if matched:
        summary = "Surat kesanggupan BPJS valid."
    elif stamp_pass is False:
        summary = "Surat kesanggupan BPJS wajib memakai materai merah."
    else:
        summary = "Surat kesanggupan BPJS tidak valid: " + "; ".join(detail_lines[:2])
    detection = {
        "detected_profile_id": profile if is_doc else None,
        "detected_document_type": profile_label(profile) if is_doc else None,
        "confidence_score": round(confidence, 2),
        "confidence_ratio": round(confidence / 100.0, 4),
        "min_aggregate_ratio": 0.0,
        "candidates": [],
    }
    name_extraction = {
        "candidate_raw": holder or None,
        "candidate_normalized": holder,
        "method": f"ornith_bpjs_kesanggupan_{ident['method']}",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type,
        requested_profile_id=profile,
        detection=detection,
        expected_name=(expected_name or "").strip(),
        identity_pass=identity_pass,
        identity=ident["identity"],
        identity_min_score=65.0,
        document_type_pass=is_doc,
        document_matched=matched,
        name_extraction=name_extraction,
    )
    if stamp_pass is False:
        verdict = dict(verdict)
        verdict["document_matched"] = False
        verdict["summary"] = summary
    return attach_user_errors(
        {
            "valid": matched,
            "document_type": (document_type or "").strip(),
            "document_profile_id": profile,
            "validation_mode": "image",
            "validation_engine": "ornith",
            "is_surat_kesanggupan": is_doc,
            "holder_name": holder,
            "name_pass": ident["name_pass"],
            "name_match_method": ident["method"],
            "stamp_pass": stamp_pass,
            "stamp": {
                "ai_stamp": ai_stamp,
                "visual_stamp": visual_stamp,
                "reason": summary if stamp_pass is False else "Materai terdeteksi.",
            },
            "document_type_pass": is_doc,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": ident["identity"],
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            "bpjs_kesanggupan_ai": {"llm": llm or {}, "extraction": parsed},
            "explanation": {
                "summary": summary,
                "detail_lines": detail_lines,
                "primary_blockers": blockers,
                "hints": [],
                "document_matched": matched,
                "validation_mode": "image",
            },
            "document_type_detection": detection,
            "verdict": verdict,
            "is_own_document": verdict["is_own_document"],
            "document_type_current": verdict["document_type_current"],
            "document_type_current_label": verdict["document_type_current_label"],
        }
    )


def analyze_bpjs_kesanggupan_image(
    image_bytes: bytes,
    *,
    document_type: str = "bpjs_kesanggupan",
    expected_name: str = "",
    **_: Any,
) -> dict[str, Any]:
    lines = [
        "Apakah ini surat kesanggupan BPJS Kesehatan? Ambil nama dan cek stempel materai fisik.",
        "Kotak tulisan Materai tanpa stempel tidak dihitung.",
        "Kembalikan JSON saja.",
    ]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi: {expected_name.strip()}")
    raw = run_ornith_json(
        image_bytes,
        user_prompt="\n".join(lines),
        system_text=KESANGGUPAN_SYSTEM_PROMPT,
        env_prefix="LLM_BPJS_KESANGGUPAN_AI",
    )
    if raw.get("error"):
        return raw
    validation = bpjs_kesanggupan_validation_from_extraction(
        raw["parsed"],
        document_type=document_type,
        expected_name=expected_name,
        image_bytes=image_bytes,
        llm=raw.get("llm"),
    )
    return {
        "error": None,
        "llm": raw.get("llm"),
        "extraction": raw["parsed"],
        "validation": validation,
        "document_matched": validation.get("document_matched"),
        "image_prepare": raw.get("image_prepare"),
        "raw_content": raw.get("raw_content"),
    }
