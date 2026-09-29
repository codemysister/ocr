"""Pipeline preprocess → OCR → validasi (dipakai HTTP API & benchmark dataset)."""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2

from systems.ocr.fast_runner import run_paddleocr_fast
from systems.ocr.mistral_annotation import parse_document_annotation
from systems.ocr.mistral_runner import run_mistral_ocr
from systems.ocr.remote_client import (
    ocr_remote_enabled,
    run_remote_ocr_fast,
    run_remote_ocr_mistral,
    run_remote_ocr_vl,
)
from systems.ocr.vl_runner import run_paddleocr_vl
from systems.preprocessing.pipeline import (
    decode_image_bytes_bgr,
    encode_image_bytes_for_ocr_direct,
    encode_image_bytes_for_ocr_passthrough,
    encode_image_bytes_for_ocr_remote_safe,
    encode_ktp_name_band_png,
    encode_ktp_status_band_png,
    encode_kk_member_table_png,
    encode_npwp_id_band_png,
    encode_rekening_detail_png,
    preprocess_image_bytes,
)
from systems.validation.document_profiles import (
    is_cv_ingest_profile,
    is_image_only_profile,
    resolve_keywords,
)
from systems.validation.bank_rekening import (
    parse_account_from_filename,
    validate_account_in_ocr,
)
from systems.validation.fuzzy_compare import (
    compare_extracted_identity_scores,
    kk_nik_matches_expected,
    npwp_id_matches_expected,
    parse_nik16_from_filename,
    validate_document_ocr,
)
from systems.validation.name_extraction import extract_holder_name_candidate
from systems.validation.ktp_marital import (
    apply_ktp_helper_marital_gate,
    extract_ktp_marital_status,
    is_helper_job_position,
)
from systems.validation.skck_validity import apply_skck_expiry_gate
from systems.validation.bpjs_kesanggupan_stamp import apply_bpjs_kesanggupan_stamp_gate
from systems.validation.jkn_status import apply_jkn_status_extract
from systems.validation.keterangan_kesehatan import apply_keterangan_kesehatan_gates
from systems.validation.keterangan_kesehatan_ai import (
    analyze_health_letter_image,
    overlay_health_ai_on_validation,
)
from systems.validation.llm_fallback import (
    is_llm_fallback_enabled,
    merge_llm_vision_into_validation,
    validate_document_via_llm_vision,
)
from systems.observability.llm_fallback_log import record_pipeline_llm_fallback
from systems.observability.validation_failure_log import record_from_pipeline_result
from systems.observability.validation_success_log import record_success_from_pipeline_result
from systems.validation.portrait_photo_validate import validate_foto_profile
from systems.validation.user_errors import attach_user_errors

OcrMode = Literal["mistral", "fast", "vl"]


@dataclass
class PipelineTiming:
    preprocess_s: float = 0.0
    ocr_s: float = 0.0
    validation_s: float = 0.0
    total_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess": round(self.preprocess_s, 3),
            "ocr": round(self.ocr_s, 3),
            "validation": round(self.validation_s, 3),
            "total": round(self.total_s, 3),
        }


@dataclass
class PipelineResult:
    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    document_matched: bool | None = None

    @property
    def http_status_hint(self) -> int:
        if self.error_kind == "empty_file":
            return 400
        if self.error_kind == "unknown_document_type":
            return 400
        if self.error_kind == "cv_unavailable":
            return 503
        if self.error_kind == "opencv_unavailable":
            return 503
        if self.error_kind in ("ocr_unavailable", "mistral_unavailable", "health_ai_unavailable"):
            return 503
        if self.error_kind == "empty_ocr":
            return 400
        return 500


def _ocr_text_from_payload(payload: dict) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    markdown = payload.get("markdown")
    if isinstance(markdown, str):
        return markdown.strip()
    return ""


def _run_ocr_inner(
    png_bytes: bytes,
    ocr_mode: OcrMode,
    *,
    full_json: bool = False,
    pp_ocr_tier: str | None = None,
    filename: str = "upload.jpg",
) -> dict:
    if ocr_remote_enabled():
        if ocr_mode == "mistral":
            return run_remote_ocr_mistral(png_bytes, filename=filename)
        if ocr_mode == "vl":
            return run_remote_ocr_vl(png_bytes, include_full_json=full_json, filename=filename)
        return run_remote_ocr_fast(png_bytes, pp_ocr_tier=pp_ocr_tier, filename=filename)
    if ocr_mode == "mistral":
        return run_mistral_ocr(png_bytes)
    if ocr_mode == "vl":
        return run_paddleocr_vl(png_bytes, include_full_json=full_json)
    return run_paddleocr_fast(png_bytes, pp_ocr_tier=pp_ocr_tier)


def _ocr_timing_s(ocr_payload: dict | None, fallback: float) -> float:
    if not ocr_payload:
        return fallback
    timing = ocr_payload.get("timing")
    if isinstance(timing, dict):
        wall = timing.get("total_wall_s")
        if isinstance(wall, (int, float)):
            return float(wall)
    return fallback


def _empty_paddle_validation(
    *,
    canonical_id: str,
    keywords: list[str],
    doc_type: str,
    reason: str,
) -> dict[str, Any]:
    return attach_user_errors(
        {
            "document_type": doc_type,
            "document_profile_id": canonical_id,
            "document_type_pass": False,
            "identity_pass": False,
            "document_matched": False,
            "explanation": {
                "summary": reason,
                "primary_blockers": ["DOCUMENT_TYPE"],
                "detail_lines": [reason],
            },
            "keywords": [],
            "keywords_from_profile": keywords,
        }
    )


def _try_llm_vision_fallback(
    validation_detail: dict[str, Any],
    *,
    image_bytes: bytes,
    image_mime: str,
    canonical_id: str,
    name_ref: str,
    expected_nik: str,
) -> tuple[dict[str, Any], bool]:
    if not is_llm_fallback_enabled():
        return validation_detail, bool(validation_detail.get("document_matched"))

    llm_result, fb_meta = validate_document_via_llm_vision(
        image_bytes,
        image_mime=image_mime,
        document_profile_id=canonical_id,
        expected_name=name_ref,
        expected_nik=expected_nik,
    )

    if llm_result and fb_meta.get("success"):
        merged = merge_llm_vision_into_validation(
            validation_detail,
            llm_result,
            fb_meta,
            expected_name=name_ref,
        )
        return merged, bool(merged.get("document_matched"))

    if fb_meta.get("attempted"):
        out = dict(validation_detail)
        out["llm_fallback"] = fb_meta
        if llm_result:
            out["llm_validation"] = llm_result
        return out, bool(validation_detail.get("document_matched"))

    return validation_detail, bool(validation_detail.get("document_matched"))


def _run_pipeline_bytes_impl(
    raw: bytes,
    *,
    document_type: str,
    expected_name: str = "",
    filename: str = "upload",
    ocr_mode: OcrMode = "fast",
    pp_ocr_tier: str | None = "medium",
    include_preprocessed_image: bool = False,
    enable_preprocess: bool = False,
    skip_passthrough: bool = False,
    full_json: bool = False,
    cv_education_query: str = "",
    cv_experience_query: str = "",
    expected_bank: str = "",
    expected_account: str = "",
    job_position: str = "",
    log_source: str = "pipeline",
) -> PipelineResult:
    """Jalankan pipeline lengkap; tidak raise HTTPException."""
    t_total0 = time.perf_counter()
    timing = PipelineTiming()
    doc_type = document_type.strip()
    name_ref = expected_name.strip()

    if not raw:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="File kosong.",
            error_kind="empty_file",
            timing=timing,
        )
    if not doc_type:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="document_type tidak boleh kosong.",
            error_kind="empty_document_type",
            timing=timing,
        )

    resolved = resolve_keywords(doc_type)
    if not resolved:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="Jenis dokumen tidak dikenal.",
            error_kind="unknown_document_type",
            timing=timing,
        )

    canonical_id, keywords = resolved

    if is_cv_ingest_profile(canonical_id):
        try:
            from systems.cv.ingest.pipeline import ingest_cv_bytes
        except ImportError:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(
                ok=False,
                error="Subsistem CV belum terpasang.",
                error_kind="cv_unavailable",
                timing=timing,
            )

        cv_name = name_ref
        t_cv0 = time.perf_counter()
        try:
            ingest_result = ingest_cv_bytes(
                raw,
                filename=filename,
                expected_name=cv_name,
                education_query=cv_education_query.strip(),
                experience_query=cv_experience_query.strip(),
            )
        except ValueError as e:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="value_error", timing=timing)
        except Exception as e:
            from systems.cv.index.opensearch_client import (
                OpenSearchUnavailableError,
                is_opensearch_connection_error,
            )

            if isinstance(e, OpenSearchUnavailableError) or is_opensearch_connection_error(e):
                timing.total_s = time.perf_counter() - t_total0
                return PipelineResult(
                    ok=False,
                    error=str(e),
                    error_kind="opencv_unavailable",
                    timing=timing,
                )
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="internal", timing=timing)

        timing.validation_s = time.perf_counter() - t_cv0
        cv_match = ingest_result.get("cv_match") or {}
        matched = bool(cv_match.get("matched", True))
        timing.total_s = time.perf_counter() - t_total0

        payload = {
            "success": True,
            "valid": matched,
            "ocr_mode": None,
            "pp_ocr_tier": None,
            "validation_mode": "cv",
            "preprocess": {"skipped": True, "reason": "cv_ingest"},
            "ocr": None,
            "validation": {
                "document_profile_id": canonical_id,
                "keywords_from_profile": keywords,
                "document_matched": matched,
            },
            "cv_ingest": ingest_result,
            "cv_match": cv_match,
            "verdict": {
                "summary": cv_match.get("summary")
                or (
                    f"CV terindeks: {ingest_result.get('chunk_count', 0)} chunk "
                    f"({ingest_result.get('parse_mode', '?')})."
                ),
                "is_own_document": (
                    cv_match.get("dimensions", {}).get("nama", {}).get("pass")
                    if cv_name
                    else None
                ),
                "document_type_current": canonical_id,
                "document_type_current_label": "CV",
            },
            "is_own_document": None,
            "document_type_current": canonical_id,
            "document_type_current_label": "CV",
            "timing": timing.as_dict(),
        }
        return PipelineResult(
            ok=True,
            payload=payload,
            timing=timing,
            document_matched=matched,
        )

    if is_image_only_profile(canonical_id):
        t_pre0 = time.perf_counter()
        try:
            bgr, decode_meta = decode_image_bytes_bgr(raw)
        except Exception as e:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="decode_error", timing=timing)
        timing.preprocess_s = time.perf_counter() - t_pre0

        t_val0 = time.perf_counter()
        validation_detail = attach_user_errors(
            validate_foto_profile(
                bgr,
                document_type=doc_type,
                expected_name=name_ref,
            )
        )
        timing.validation_s = time.perf_counter() - t_val0
        analysis_bgr = validation_detail.pop("_analysis_bgr", None)
        matched = bool(validation_detail.get("document_matched"))

        payload = {
            "success": True,
            "valid": matched,
            "ocr_mode": None,
            "pp_ocr_tier": None,
            "validation_mode": "image",
            "preprocess": {"skipped": True, "reason": "image_only_profile", **decode_meta},
            "ocr": None,
            "validation": {
                "document_profile_id": canonical_id,
                "keywords_from_profile": keywords,
                **validation_detail,
            },
            "verdict": validation_detail.get("verdict"),
            "is_own_document": validation_detail.get("is_own_document"),
            "document_type_current": validation_detail.get("document_type_current"),
            "document_type_current_label": validation_detail.get("document_type_current_label"),
            "error_code": validation_detail.get("error_code"),
            "error_message": validation_detail.get("error_message"),
            "error_cases": validation_detail.get("error_cases") or [],
            "timing": timing.as_dict(),
        }
        if include_preprocessed_image:
            preview_bgr = analysis_bgr if analysis_bgr is not None else bgr
            ok, encoded = cv2.imencode(".png", preview_bgr)
            if ok:
                payload["preprocessed_image"] = {
                    "mime": "image/png",
                    "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }

        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=True,
            payload=payload,
            timing=timing,
            document_matched=matched,
        )

    t_pre0 = time.perf_counter()
    use_direct_upload = skip_passthrough or (ocr_remote_enabled() and not enable_preprocess)
    try:
        if enable_preprocess:
            png_bytes, pre_meta = preprocess_image_bytes(raw, document_profile_id=canonical_id)
        elif use_direct_upload:
            if ocr_remote_enabled():
                png_bytes, pre_meta = encode_image_bytes_for_ocr_remote_safe(raw)
            else:
                png_bytes, pre_meta = encode_image_bytes_for_ocr_direct(raw)
        else:
            png_bytes, pre_meta = encode_image_bytes_for_ocr_passthrough(raw)
    except ValueError as e:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind="preprocess_error", timing=timing)
    timing.preprocess_s = time.perf_counter() - t_pre0

    expected_nik = parse_nik16_from_filename(filename) or ""
    if canonical_id == "keterangan_kesehatan":
        t_ocr0 = time.perf_counter()
        try:
            ai_out = analyze_health_letter_image(
                raw,
                image_mime="image/jpeg",
                expected_name=name_ref,
                expected_nik=expected_nik,
            )
        except ValueError as e:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="ocr_value_error", timing=timing)
        except Exception as e:
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind="health_ai_unavailable", timing=timing)
        timing.ocr_s = time.perf_counter() - t_ocr0
        if ai_out.get("error") or not ai_out.get("extraction"):
            err = str(ai_out.get("error") or "empty_extraction")
            llm_err = (ai_out.get("llm") or {}).get("error")
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(
                ok=False,
                error=f"OCR AI surat kesehatan gagal: {llm_err or err}",
                error_kind="health_ai_unavailable",
                timing=timing,
            )
        ocr_text = str((ai_out.get("validation") or {}).get("synthetic_ocr") or "").strip()
        ocr_payload = {
            "success": True,
            "mode": "ornith_health_ai",
            "model": (ai_out.get("llm") or {}).get("model"),
            "text": ocr_text,
            "markdown": ocr_text,
            "extraction": ai_out.get("extraction"),
            "raw_content": ai_out.get("raw_content"),
            "timing": {"total_wall_s": round(timing.ocr_s, 3)},
        }
        image_mime = pre_meta.get("mime") if isinstance(pre_meta.get("mime"), str) else "image/png"
        expected_acct = (expected_account or "").strip() or parse_account_from_filename(filename) or ""
        t_val0 = time.perf_counter()
        validation_detail = validate_document_ocr(
            ocr_text,
            document_type=doc_type,
            document_profile_id=canonical_id,
            keywords=keywords,
            expected_name=name_ref,
            expected_nik=expected_nik,
            expected_bank=expected_bank,
            expected_account=expected_acct,
            job_position=job_position,
            mistral_annotation=None,
        )
        validation_detail = apply_keterangan_kesehatan_gates(
            apply_jkn_status_extract(
                apply_bpjs_kesanggupan_stamp_gate(
                    apply_skck_expiry_gate(
                        apply_ktp_helper_marital_gate(
                            validation_detail,
                            ocr_text=ocr_text,
                            job_position=job_position,
                            profile_id=canonical_id,
                        ),
                        ocr_text=ocr_text,
                        profile_id=canonical_id,
                    ),
                    ocr_text=ocr_text,
                    profile_id=canonical_id,
                    image=png_bytes,
                ),
                ocr_text=ocr_text,
                profile_id=canonical_id,
                expected_name=name_ref,
            ),
            ocr_text=ocr_text,
            profile_id=canonical_id,
        )
        validation_detail = overlay_health_ai_on_validation(validation_detail, ai_out)
        validation_detail = attach_user_errors(validation_detail)
        matched = bool(validation_detail.get("document_matched"))
        timing.validation_s = time.perf_counter() - t_val0
        timing.total_s = time.perf_counter() - t_total0
        payload = {
            "success": True,
            "ocr_mode": "ornith",
            "ocr_engine": "ornith",
            "pp_ocr_tier": None,
            "preprocess": pre_meta,
            "ocr": ocr_payload,
            "validation": {
                "document_profile_id": canonical_id,
                "keywords_from_profile": keywords,
                **validation_detail,
            },
            "verdict": validation_detail.get("verdict"),
            "is_own_document": validation_detail.get("is_own_document"),
            "document_type_current": validation_detail.get("document_type_current"),
            "document_type_current_label": validation_detail.get("document_type_current_label"),
            "error_code": validation_detail.get("error_code"),
            "error_message": validation_detail.get("error_message"),
            "error_cases": validation_detail.get("error_cases") or [],
            "timing": timing.as_dict(),
        }
        if include_preprocessed_image:
            preview_mime = pre_meta.get("mime") if isinstance(pre_meta.get("mime"), str) else "image/png"
            payload["preprocessed_image"] = {
                "mime": preview_mime,
                "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            }
        return PipelineResult(
            ok=True,
            payload=payload,
            timing=timing,
            document_matched=matched,
        )

    tier_arg = pp_ocr_tier if ocr_mode == "fast" else None
    t_ocr0 = time.perf_counter()
    try:
        ocr_payload = _run_ocr_inner(
            png_bytes,
            ocr_mode,
            full_json=full_json,
            pp_ocr_tier=tier_arg,
            filename=filename,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if ocr_remote_enabled() and ("http 500" in msg or "http 503" in msg):
            try:
                png_bytes, retry_meta = encode_image_bytes_for_ocr_remote_safe(
                    raw, max_long_side=1280
                )
                if isinstance(pre_meta, dict):
                    pre_meta = {**pre_meta, "remote_gpu_retry": True, **retry_meta}
                ocr_payload = _run_ocr_inner(
                    png_bytes,
                    ocr_mode,
                    full_json=full_json,
                    pp_ocr_tier=tier_arg,
                    filename=filename,
                )
            except Exception:
                kind = "ocr_unavailable"
                if "mistral_api_key" in msg or "pasang sdk mistral" in msg:
                    kind = "mistral_unavailable"
                timing.total_s = time.perf_counter() - t_total0
                return PipelineResult(ok=False, error=str(e), error_kind=kind, timing=timing)
        else:
            kind = "ocr_unavailable"
            if "mistral_api_key" in msg or "pasang sdk mistral" in msg:
                kind = "mistral_unavailable"
            timing.total_s = time.perf_counter() - t_total0
            return PipelineResult(ok=False, error=str(e), error_kind=kind, timing=timing)
    except ValueError as e:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind="ocr_value_error", timing=timing)
    except Exception as e:
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(ok=False, error=str(e), error_kind="ocr_error", timing=timing)

    client_ocr_s = time.perf_counter() - t_ocr0
    if ocr_remote_enabled():
        timing.ocr_s = client_ocr_s
        if isinstance(ocr_payload, dict):
            ocr_payload = dict(ocr_payload)
            timing_obj = ocr_payload.get("timing")
            if not isinstance(timing_obj, dict):
                timing_obj = {}
            timing_obj = dict(timing_obj)
            timing_obj["client_wall_s"] = round(client_ocr_s, 3)
            ocr_payload["timing"] = timing_obj
    else:
        timing.ocr_s = _ocr_timing_s(ocr_payload, client_ocr_s)
    ocr_text = _ocr_text_from_payload(ocr_payload)
    if (
        ocr_mode == "fast"
        and canonical_id == "ktp"
        and is_helper_job_position(job_position)
        and extract_ktp_marital_status(ocr_text).get("status") is None
    ):
        t_band0 = time.perf_counter()
        try:
            band_bytes, band_meta = encode_ktp_status_band_png(png_bytes)
        except Exception:
            band_bytes, band_meta = None, {}
        if band_bytes:
            try:
                band_payload = _run_ocr_inner(
                    band_bytes,
                    ocr_mode,
                    full_json=False,
                    pp_ocr_tier=tier_arg,
                    filename="ktp_status_band.png",
                )
                band_text = _ocr_text_from_payload(band_payload)
                if band_text:
                    ocr_text = f"{ocr_text}\n{band_text}".strip()
                    if isinstance(ocr_payload, dict):
                        ocr_payload = dict(ocr_payload)
                        ocr_payload["text"] = ocr_text
                if isinstance(pre_meta, dict):
                    pre_meta["ktp_status_band_ocr"] = True
                    pre_meta["ktp_status_band"] = {
                        k: band_meta.get(k)
                        for k in ("band_box", "band_width", "band_height")
                        if k in band_meta
                    }
            except Exception:
                if isinstance(pre_meta, dict):
                    pre_meta["ktp_status_band_ocr"] = False
        timing.ocr_s = round(timing.ocr_s + (time.perf_counter() - t_band0), 3)

    if (
        ocr_mode == "fast"
        and canonical_id == "ktp"
        and name_ref
    ):
        cand, _ = extract_holder_name_candidate(ocr_text, "ktp")
        ident = compare_extracted_identity_scores(cand or "", name_ref)
        compact_ocr = re.sub(r"[^a-z]", "", (ocr_text or "").casefold())
        compact_exp = re.sub(r"[^a-z]", "", name_ref.casefold())
        name_in_ocr = len(compact_exp) >= 10 and compact_exp in compact_ocr
        if not ident.get("token_gate_pass") and not name_in_ocr:
            t_name0 = time.perf_counter()
            try:
                name_bytes, name_meta = encode_ktp_name_band_png(png_bytes)
            except Exception:
                name_bytes, name_meta = None, {}
            if name_bytes:
                try:
                    name_payload = _run_ocr_inner(
                        name_bytes,
                        ocr_mode,
                        full_json=False,
                        pp_ocr_tier=tier_arg,
                        filename="ktp_name_band.png",
                    )
                    name_text = _ocr_text_from_payload(name_payload)
                    if name_text:
                        ocr_text = f"{ocr_text}\n{name_text}".strip()
                        if isinstance(ocr_payload, dict):
                            ocr_payload = dict(ocr_payload)
                            ocr_payload["text"] = ocr_text
                    if isinstance(pre_meta, dict):
                        pre_meta["ktp_name_band_ocr"] = True
                        pre_meta["ktp_name_band"] = {
                            k: name_meta.get(k)
                            for k in ("band_box", "band_width", "band_height")
                            if k in name_meta
                        }
                except Exception:
                    if isinstance(pre_meta, dict):
                        pre_meta["ktp_name_band_ocr"] = False
            timing.ocr_s = round(timing.ocr_s + (time.perf_counter() - t_name0), 3)

    expected_nik = parse_nik16_from_filename(filename) or ""
    if (
        ocr_mode == "fast"
        and canonical_id in {"npwp", "pemadanan_npwp"}
        and expected_nik
        and not npwp_id_matches_expected(ocr_text, expected_nik)
    ):
            t_npwp0 = time.perf_counter()
            try:
                npwp_bytes, npwp_meta = encode_npwp_id_band_png(png_bytes)
            except Exception:
                npwp_bytes, npwp_meta = None, {}
            if npwp_bytes:
                try:
                    npwp_payload = _run_ocr_inner(
                        npwp_bytes,
                        ocr_mode,
                        full_json=False,
                        pp_ocr_tier=tier_arg,
                        filename="npwp_id_band.png",
                    )
                    npwp_text = _ocr_text_from_payload(npwp_payload)
                    if npwp_text:
                        ocr_text = f"{ocr_text}\n{npwp_text}".strip()
                        if isinstance(ocr_payload, dict):
                            ocr_payload = dict(ocr_payload)
                            ocr_payload["text"] = ocr_text
                    if isinstance(pre_meta, dict):
                        pre_meta["npwp_id_band_ocr"] = True
                        pre_meta["npwp_id_band"] = {
                            k: npwp_meta.get(k)
                            for k in ("band_box", "band_width", "band_height")
                            if k in npwp_meta
                        }
                except Exception:
                    if isinstance(pre_meta, dict):
                        pre_meta["npwp_id_band_ocr"] = False
            timing.ocr_s = round(timing.ocr_s + (time.perf_counter() - t_npwp0), 3)

    if ocr_mode == "fast" and canonical_id == "kk" and (name_ref or expected_nik):
        compact_ocr = re.sub(r"[^a-z0-9]", "", (ocr_text or "").casefold())
        compact_exp = re.sub(r"[^a-z0-9]", "", (name_ref or "").casefold())
        name_in_ocr = len(compact_exp) >= 5 and compact_exp in compact_ocr
        nik_in_ocr = bool(expected_nik) and bool(
            kk_nik_matches_expected(ocr_text, expected_nik)
        )
        if not name_in_ocr and not nik_in_ocr:
            t_kk0 = time.perf_counter()
            try:
                kk_bytes, kk_meta = encode_kk_member_table_png(png_bytes)
            except Exception:
                kk_bytes, kk_meta = None, {}
            if kk_bytes:
                try:
                    kk_payload = _run_ocr_inner(
                        kk_bytes,
                        ocr_mode,
                        full_json=False,
                        pp_ocr_tier=tier_arg,
                        filename="kk_member_table.png",
                    )
                    kk_text = _ocr_text_from_payload(kk_payload)
                    if kk_text:
                        ocr_text = f"{ocr_text}\n{kk_text}".strip()
                        if isinstance(ocr_payload, dict):
                            ocr_payload = dict(ocr_payload)
                            ocr_payload["text"] = ocr_text
                    if isinstance(pre_meta, dict):
                        pre_meta["kk_member_table_ocr"] = True
                        pre_meta["kk_member_table"] = {
                            k: kk_meta.get(k)
                            for k in ("band_boxes", "band_width", "band_height")
                            if k in kk_meta
                        }
                except Exception:
                    if isinstance(pre_meta, dict):
                        pre_meta["kk_member_table_ocr"] = False
            timing.ocr_s = round(timing.ocr_s + (time.perf_counter() - t_kk0), 3)

    expected_acct_early = (
        (expected_account or "").strip() or parse_account_from_filename(filename) or ""
    )
    if ocr_mode == "fast" and canonical_id in {"rekening", "mutasi"} and (
        (canonical_id == "rekening" and name_ref) or expected_acct_early
    ):
        compact_ocr = re.sub(r"[^a-z0-9]", "", (ocr_text or "").casefold())
        compact_exp = re.sub(r"[^a-z0-9]", "", (name_ref or "").casefold())
        name_in_ocr = True
        if canonical_id == "rekening":
            name_in_ocr = len(compact_exp) >= 8 and compact_exp in compact_ocr
            if not name_in_ocr and len(compact_exp) >= 10:
                compact_letters = re.sub(r"[^a-z]", "", compact_ocr)
                for n in range(len(compact_exp), 9, -1):
                    pref = compact_exp[:n]
                    if pref in compact_letters and (n / len(compact_exp)) >= 0.68:
                        name_in_ocr = True
                        break
        acct_info = (
            validate_account_in_ocr(ocr_text, expected_acct_early)
            if expected_acct_early
            else {}
        )
        acct_ok = acct_info.get("account_pass") is True
        need_crop = (canonical_id == "rekening" and not name_in_ocr) or (
            bool(expected_acct_early) and not acct_ok
        )
        if need_crop:
            t_rek0 = time.perf_counter()
            try:
                rek_bytes, rek_meta = encode_rekening_detail_png(png_bytes)
            except Exception:
                rek_bytes, rek_meta = None, {}
            if rek_bytes:
                try:
                    rek_payload = _run_ocr_inner(
                        rek_bytes,
                        ocr_mode,
                        full_json=False,
                        pp_ocr_tier=tier_arg,
                        filename="rekening_detail.png",
                    )
                    rek_text = _ocr_text_from_payload(rek_payload)
                    if rek_text:
                        ocr_text = f"{ocr_text}\n{rek_text}".strip()
                        if isinstance(ocr_payload, dict):
                            ocr_payload = dict(ocr_payload)
                            ocr_payload["text"] = ocr_text
                    if isinstance(pre_meta, dict):
                        pre_meta["rekening_detail_ocr"] = True
                        pre_meta["rekening_detail"] = {
                            k: rek_meta.get(k)
                            for k in ("band_boxes", "band_width", "band_height")
                            if k in rek_meta
                        }
                except Exception:
                    if isinstance(pre_meta, dict):
                        pre_meta["rekening_detail_ocr"] = False
            timing.ocr_s = round(timing.ocr_s + (time.perf_counter() - t_rek0), 3)

    image_mime = pre_meta.get("mime") if isinstance(pre_meta.get("mime"), str) else "image/png"
    expected_acct = (expected_account or "").strip() or parse_account_from_filename(filename) or ""

    mistral_ann: dict | None = None
    if ocr_mode == "mistral":
        mistral_ann = ocr_payload.get("document_annotation_parsed")
        if mistral_ann is None:
            mistral_ann = parse_document_annotation(ocr_payload.get("document_annotation"))

    t_val0 = time.perf_counter()
    if ocr_text:
        validation_detail = validate_document_ocr(
            ocr_text,
            document_type=doc_type,
            document_profile_id=canonical_id,
            keywords=keywords,
            expected_name=name_ref,
            expected_nik=expected_nik,
            expected_bank=expected_bank,
            expected_account=expected_acct,
            job_position=job_position,
            mistral_annotation=mistral_ann,
        )
    else:
        validation_detail = _empty_paddle_validation(
            canonical_id=canonical_id,
            keywords=keywords,
            doc_type=doc_type,
            reason="OCR Paddle tidak menghasilkan teks — mencoba validasi AI vision.",
        )

    matched = bool(validation_detail.get("document_matched"))
    paddle_detail = validation_detail
    if canonical_id != "keterangan_kesehatan" and not matched:
        validation_detail, matched = _try_llm_vision_fallback(
            validation_detail,
            image_bytes=png_bytes,
            image_mime=image_mime,
            canonical_id=canonical_id,
            name_ref=name_ref,
            expected_nik=expected_nik,
        )
        record_pipeline_llm_fallback(
            paddle_detail=paddle_detail,
            final_detail=validation_detail,
            filename=filename,
            document_type=doc_type,
            document_profile_id=canonical_id,
            expected_name=name_ref,
            expected_nik=expected_nik,
            ocr_mode=ocr_mode,
            ocr_text=ocr_text,
            source=log_source,
        )
    validation_detail = apply_keterangan_kesehatan_gates(
        apply_jkn_status_extract(
            apply_bpjs_kesanggupan_stamp_gate(
                apply_skck_expiry_gate(
                    apply_ktp_helper_marital_gate(
                        validation_detail,
                        ocr_text=ocr_text,
                        job_position=job_position,
                        profile_id=canonical_id,
                    ),
                    ocr_text=ocr_text,
                    profile_id=canonical_id,
                ),
                ocr_text=ocr_text,
                profile_id=canonical_id,
                image=png_bytes,
            ),
            ocr_text=ocr_text,
            profile_id=canonical_id,
            expected_name=name_ref,
        ),
        ocr_text=ocr_text,
        profile_id=canonical_id,
    )
    validation_detail = attach_user_errors(validation_detail)
    matched = bool(validation_detail.get("document_matched"))
    timing.validation_s = time.perf_counter() - t_val0

    if not ocr_text and not matched and not validation_detail.get("llm_fallback", {}).get("attempted"):
        timing.total_s = time.perf_counter() - t_total0
        return PipelineResult(
            ok=False,
            error="OCR tidak menghasilkan teks.",
            error_kind="empty_ocr",
            timing=timing,
        )

    timing.total_s = time.perf_counter() - t_total0

    payload = {
        "success": True,
        "ocr_mode": ocr_mode,
        "pp_ocr_tier": tier_arg if ocr_mode == "fast" else None,
        "preprocess": pre_meta,
        "ocr": ocr_payload,
        "validation": {
            "document_profile_id": canonical_id,
            "keywords_from_profile": keywords,
            **validation_detail,
        },
        "verdict": validation_detail.get("verdict"),
        "is_own_document": validation_detail.get("is_own_document"),
        "document_type_current": validation_detail.get("document_type_current"),
        "document_type_current_label": validation_detail.get("document_type_current_label"),
        "error_code": validation_detail.get("error_code"),
        "error_message": validation_detail.get("error_message"),
        "error_cases": validation_detail.get("error_cases") or [],
        "timing": timing.as_dict(),
    }
    if include_preprocessed_image:
        preview_mime = pre_meta.get("mime") if isinstance(pre_meta.get("mime"), str) else "image/png"
        payload["preprocessed_image"] = {
            "mime": preview_mime,
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
        }

    return PipelineResult(
        ok=True,
        payload=payload,
        timing=timing,
        document_matched=matched,
    )


def run_pipeline_bytes(
    raw: bytes,
    *,
    document_type: str,
    expected_name: str = "",
    filename: str = "upload",
    ocr_mode: OcrMode = "fast",
    pp_ocr_tier: str | None = "medium",
    include_preprocessed_image: bool = False,
    enable_preprocess: bool = False,
    skip_passthrough: bool = False,
    full_json: bool = False,
    cv_education_query: str = "",
    cv_experience_query: str = "",
    expected_bank: str = "",
    expected_account: str = "",
    job_position: str = "",
    log_source: str = "pipeline",
) -> PipelineResult:
    result = _run_pipeline_bytes_impl(
        raw,
        document_type=document_type,
        expected_name=expected_name,
        filename=filename,
        ocr_mode=ocr_mode,
        pp_ocr_tier=pp_ocr_tier,
        include_preprocessed_image=include_preprocessed_image,
        enable_preprocess=enable_preprocess,
        skip_passthrough=skip_passthrough,
        full_json=full_json,
        cv_education_query=cv_education_query,
        cv_experience_query=cv_experience_query,
        expected_bank=expected_bank,
        expected_account=expected_account,
        job_position=job_position,
        log_source=log_source,
    )
    try:
        record_from_pipeline_result(
            result,
            filename=filename,
            document_type=document_type,
            expected_name=expected_name,
            expected_bank=expected_bank,
            expected_account=expected_account,
            job_position=job_position,
            source=log_source,
        )
        record_success_from_pipeline_result(
            result,
            filename=filename,
            document_type=document_type,
            expected_name=expected_name,
            expected_bank=expected_bank,
            expected_account=expected_account,
            job_position=job_position,
            source=log_source,
        )
    except Exception:
        logging.getLogger(__name__).exception("Gagal mencatat validation failure/success log")
    return result
