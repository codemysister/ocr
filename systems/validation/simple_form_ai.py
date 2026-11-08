"""Validasi ijazah, transkrip, dan formulir lamaran via Ornith."""

from __future__ import annotations

from typing import Any

from systems.validation.document_profiles import profile_label
from systems.validation.fuzzy_compare import build_document_verdict
from systems.validation.ornith_call import as_bool, decide_identity, run_ornith_json
from systems.validation.user_errors import attach_user_errors

_FORMS: dict[str, dict[str, str]] = {
    "ijasah": {
        "flag": "is_ijasah",
        "env": "LLM_IJASAH_AI",
        "ok": "Gambar dikenali sebagai ijazah.",
        "bad": "Gambar bukan ijazah.",
        "summary": "Ijazah valid.",
        "ask": "Apakah ini ijazah? Ambil nama pemilik ijazah.",
        "system": (
            "Kamu membaca ijazah pendidikan Indonesia.\n"
            "Jangan mengarang.\n"
            "Kembalikan HANYA satu objek JSON.\n"
            "- is_ijasah (boolean): true hanya jika ini ijazah. "
            "false jika transkrip, surat lamaran, KTP, atau dokumen lain.\n"
            "- holder_name (string): nama pemilik ijazah, apa adanya.\n"
            "- confidence (number 0-100)\n"
            "- notes (string): maksimal 12 kata"
        ),
    },
    "transkrip": {
        "flag": "is_transkrip",
        "env": "LLM_TRANSKRIP_AI",
        "ok": "Gambar dikenali sebagai transkrip nilai.",
        "bad": "Gambar bukan transkrip nilai.",
        "summary": "Transkrip nilai valid.",
        "ask": "Apakah ini transkrip nilai? Ambil nama mahasiswa.",
        "system": (
            "Kamu membaca transkrip nilai.\n"
            "Jangan mengarang.\n"
            "Kembalikan HANYA satu objek JSON.\n"
            "- is_transkrip (boolean): true hanya jika ini transkrip nilai. "
            "false jika ijazah, KTP, atau dokumen lain.\n"
            "- holder_name (string): nama mahasiswa, apa adanya.\n"
            "- confidence (number 0-100)\n"
            "- notes (string): maksimal 12 kata"
        ),
    },
    "formulir_okb": {
        "flag": "is_formulir_okb",
        "env": "LLM_FORMULIR_OKB_AI",
        "ok": "Gambar dikenali sebagai formulir OKB.",
        "bad": "Gambar bukan formulir OKB.",
        "summary": "Formulir OKB valid.",
        "ask": "Apakah ini formulir OKB? Ambil nama yang tercantum.",
        "system": (
            "Kamu membaca formulir OKB / pemeriksaan OKB.\n"
            "Jangan mengarang.\n"
            "Kembalikan HANYA satu objek JSON.\n"
            "- is_formulir_okb (boolean): true hanya jika ini formulir OKB. "
            "false jika surat lamaran, KTP, atau dokumen lain.\n"
            "- holder_name (string): nama orang pada formulir, apa adanya.\n"
            "- confidence (number 0-100)\n"
            "- notes (string): maksimal 12 kata"
        ),
    },
    "formulir_lamaran": {
        "flag": "is_formulir_lamaran",
        "env": "LLM_FORMULIR_LAMARAN_AI",
        "ok": "Gambar dikenali sebagai formulir lamaran pekerjaan.",
        "bad": "Gambar bukan formulir lamaran pekerjaan.",
        "summary": "Formulir lamaran valid.",
        "ask": "Apakah ini formulir lamaran pekerjaan? Ambil nama pelamar.",
        "system": (
            "Kamu membaca formulir lamaran pekerjaan (isian data pribadi), bukan surat lamaran.\n"
            "Jangan mengarang.\n"
            "Kembalikan HANYA satu objek JSON.\n"
            "- is_formulir_lamaran (boolean): true hanya jika ini formulir lamaran. "
            "false jika surat lamaran, CV, KTP, atau dokumen lain.\n"
            "- holder_name (string): nama pelamar, apa adanya.\n"
            "- confidence (number 0-100)\n"
            "- notes (string): maksimal 12 kata"
        ),
    },
    "surat_lamaran": {
        "flag": "is_surat_lamaran",
        "env": "LLM_SURAT_LAMARAN_AI",
        "ok": "Gambar dikenali sebagai surat lamaran.",
        "bad": "Gambar bukan surat lamaran.",
        "summary": "Surat lamaran valid.",
        "ask": "Apakah ini surat lamaran? Ambil nama yang bertanda tangan.",
        "system": (
            "Kamu membaca surat lamaran pekerjaan.\n"
            "Jangan mengarang.\n"
            "Kembalikan HANYA satu objek JSON.\n"
            "- is_surat_lamaran (boolean): true hanya jika ini surat lamaran. "
            "false jika formulir isian, CV, KTP, atau dokumen lain.\n"
            "- holder_name (string): nama pelamar / yang bertanda tangan, apa adanya.\n"
            "- confidence (number 0-100)\n"
            "- notes (string): maksimal 12 kata"
        ),
    },
}


def form_validation_from_extraction(
    parsed: dict[str, Any],
    *,
    profile: str,
    document_type: str = "",
    expected_name: str = "",
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _FORMS[profile]
    is_doc = as_bool(parsed.get(spec["flag"]))
    holder = str(parsed.get("holder_name") or "").strip()
    ident = decide_identity(expected_name=expected_name, holder=holder, use_nik=False)
    identity_pass = ident["identity_pass"]
    matched = is_doc and identity_pass is not False
    blockers: list[str] = []
    detail_lines = [spec["ok"] if is_doc else spec["bad"]]
    if not is_doc:
        blockers.append("DOCUMENT_TYPE")
    if ident["name_pass"] is True:
        detail_lines.append(f"Nama cocok: {holder}.")
    elif identity_pass is False:
        blockers.append("IDENTITY")
        detail_lines.append(f"Nama tidak cocok: {holder or 'tidak terbaca'}.")
    notes = str(parsed.get("notes") or "").strip()
    if notes:
        detail_lines.append(notes)
    try:
        confidence = float(parsed.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    summary = spec["summary"] if matched else profile_label(profile) + " tidak valid: " + "; ".join(detail_lines[:2])
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
        "method": f"ornith_{profile}_{ident['method']}",
    }
    verdict = build_document_verdict(
        requested_document_type=document_type or profile,
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
    return attach_user_errors(
        {
            "valid": matched,
            "document_type": (document_type or profile).strip(),
            "document_profile_id": profile,
            "validation_mode": "image",
            "validation_engine": "ornith",
            spec["flag"]: is_doc,
            "holder_name": holder,
            "name_pass": ident["name_pass"],
            "name_match_method": ident["method"],
            "document_type_pass": is_doc,
            "document_matched": matched,
            "identity_pass": identity_pass,
            "identity": ident["identity"],
            "name_extraction": name_extraction,
            "keywords": [],
            "excluded_keywords": [],
            "exclusion_violated": False,
            f"{profile}_ai": {"llm": llm or {}, "extraction": parsed},
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


def _analyze(image_bytes: bytes, *, profile: str, document_type: str, expected_name: str) -> dict[str, Any]:
    spec = _FORMS[profile]
    lines = [spec["ask"], "Kembalikan JSON saja."]
    if expected_name.strip():
        lines.insert(1, f"Nama referensi: {expected_name.strip()}")
    raw = run_ornith_json(
        image_bytes,
        user_prompt="\n".join(lines),
        system_text=spec["system"],
        env_prefix=spec["env"],
    )
    if raw.get("error"):
        return raw
    validation = form_validation_from_extraction(
        raw["parsed"],
        profile=profile,
        document_type=document_type,
        expected_name=expected_name,
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


def analyze_ijasah_image(image_bytes: bytes, *, document_type: str = "ijasah", expected_name: str = "", **_: Any) -> dict[str, Any]:
    return _analyze(image_bytes, profile="ijasah", document_type=document_type, expected_name=expected_name)


def analyze_transkrip_image(image_bytes: bytes, *, document_type: str = "transkrip", expected_name: str = "", **_: Any) -> dict[str, Any]:
    return _analyze(image_bytes, profile="transkrip", document_type=document_type, expected_name=expected_name)


def analyze_formulir_okb_image(image_bytes: bytes, *, document_type: str = "formulir_okb", expected_name: str = "", **_: Any) -> dict[str, Any]:
    return _analyze(image_bytes, profile="formulir_okb", document_type=document_type, expected_name=expected_name)


def analyze_formulir_lamaran_image(image_bytes: bytes, *, document_type: str = "formulir_lamaran", expected_name: str = "", **_: Any) -> dict[str, Any]:
    return _analyze(image_bytes, profile="formulir_lamaran", document_type=document_type, expected_name=expected_name)


def analyze_surat_lamaran_image(image_bytes: bytes, *, document_type: str = "surat_lamaran", expected_name: str = "", **_: Any) -> dict[str, Any]:
    return _analyze(image_bytes, profile="surat_lamaran", document_type=document_type, expected_name=expected_name)
