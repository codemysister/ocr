"""Pesan error tetap per dokumen + case — untuk tampilan frontend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from systems.validation.document_profiles import profile_label

# code → pesan singkat (satu kalimat, siap tampil di UI).
ERROR_CATALOG: dict[str, str] = {
    # Pipeline
    "pipeline.empty_file": "File kosong. Unggah gambar dokumen.",
    "pipeline.unknown_document_type": "Jenis dokumen tidak dikenali.",
    "pipeline.empty_document_type": "Jenis dokumen wajib diisi.",
    "pipeline.unreadable_image": "Gambar tidak bisa dibaca. Gunakan JPEG, PNG, atau WebP.",
    "pipeline.empty_ocr": "Teks pada dokumen tidak terbaca. Foto ulang dengan lebih jelas.",
    "pipeline.ocr_unavailable": "Layanan OCR sedang tidak tersedia. Coba lagi nanti.",
    "pipeline.health_ai_unavailable": "OCR AI surat kesehatan tidak merespons. Coba lagi nanti.",
    "pipeline.portrait_ai_unavailable": "AI foto profil tidak merespons. Coba lagi nanti.",
    "pipeline.npwp_ai_unavailable": "AI NPWP tidak merespons. Coba lagi nanti.",
    "pipeline.pemadanan_npwp_ai_unavailable": "AI pemadanan NPWP tidak merespons. Coba lagi nanti.",
    "pipeline.rekening_ai_unavailable": "AI rekening tidak merespons. Coba lagi nanti.",
    "pipeline.bpjs_ai_unavailable": "AI BPJS Kesehatan tidak merespons. Coba lagi nanti.",
    "pipeline.iuran_ai_unavailable": "AI Info Iuran tidak merespons. Coba lagi nanti.",
    "pipeline.vaksinasi_1_ai_unavailable": "AI vaksinasi dosis 1 tidak merespons. Coba lagi nanti.",
    "pipeline.vaksinasi_2_ai_unavailable": "AI vaksinasi dosis 2 tidak merespons. Coba lagi nanti.",
    "pipeline.vaksinasi_3_ai_unavailable": "AI vaksinasi booster tidak merespons. Coba lagi nanti.",
    "pipeline.ocr_error": "Gagal membaca teks dokumen. Coba unggah ulang.",
    "pipeline.internal": "Terjadi kesalahan sistem saat memproses dokumen.",
    "pipeline.missing_expected_bank": "Bank referensi wajib diisi (mandiri atau mas).",
    "pipeline.missing_expected_account": "Nomor rekening referensi wajib diisi.",
    "pipeline.missing_expected_identity": "Isi nama dan/atau nomor rekening referensi.",
    "pipeline.missing_ocr_text": "Teks OCR kosong.",
    "generic.validation_failed": "Dokumen tidak lolos validasi.",
    # KTP
    "ktp.wrong_document": "Dokumen ini bukan KTP yang valid.",
    "ktp.name_mismatch": "Nama pada KTP tidak cocok.",
    "ktp.helper_not_single": "Untuk posisi Helper, status perkawinan di KTP harus BELUM KAWIN.",
    # NPWP
    "npwp.wrong_document": "Dokumen ini bukan NPWP yang valid.",
    "npwp.name_mismatch": "Nama pada NPWP tidak cocok.",
    "npwp.nik_mismatch": "NIK/nomor NPWP 16 digit tidak cocok.",
    "pemadanan_npwp.wrong_document": "Dokumen ini bukan bukti pemadanan NPWP yang valid.",
    "pemadanan_npwp.name_mismatch": "Nama pada bukti pemadanan NPWP tidak cocok.",
    "pemadanan_npwp.nik_mismatch": "NIK/nomor NPWP 16 digit tidak cocok.",
    # KK
    "kk.wrong_document": "Dokumen ini bukan Kartu Keluarga yang valid.",
    "kk.identity_mismatch": "Nama atau NIK anggota pada Kartu Keluarga tidak cocok.",
    # Rekening
    "rekening.wrong_document": "Dokumen ini bukan bukti rekening tabungan yang valid.",
    "rekening.wrong_type_statement": "Ini tampak e-statement/mutasi, bukan halaman rekening tabungan.",
    "rekening.wrong_bank": "Bank pada rekening tidak sesuai (harus Mandiri atau Bank MAS sesuai data).",
    "rekening.name_mismatch": "Nama pemilik rekening tidak cocok.",
    "rekening.account_mismatch": "Nomor rekening tidak terlihat atau tidak cocok.",
    # Mutasi
    "mutasi.wrong_document": "Dokumen ini bukan mutasi/e-statement yang valid.",
    "mutasi.wrong_bank": "Bank pada mutasi tidak sesuai (harus Mandiri atau Bank MAS sesuai data).",
    "mutasi.account_mismatch": "Nomor rekening pada mutasi tidak sesuai.",
    # SKCK
    "skck.wrong_document": "Dokumen ini bukan SKCK yang valid.",
    "skck.name_mismatch": "Nama pada SKCK tidak cocok.",
    "skck.expiry_missing": "Tanggal kadaluarsa SKCK (Sampai dengan) tidak terbaca.",
    "skck.expired": "SKCK sudah kadaluarsa.",
    "skck.not_yet_valid": "SKCK belum berlaku pada tanggal hari ini.",
    # BPJS / JKN / iuran / vaksin
    "bpjs.wrong_document": "Dokumen ini bukan kartu BPJS Kesehatan (KIS) yang valid.",
    "bpjs.name_mismatch": "Nama pada kartu BPJS Kesehatan tidak cocok.",
    "bpjs_tk.wrong_document": "Dokumen ini bukan kartu BPJS Ketenagakerjaan yang valid.",
    "bpjs_tk.name_mismatch": "Nama pada kartu BPJS Ketenagakerjaan tidak cocok.",
    "bpjs_kesanggupan.wrong_document": "Dokumen ini bukan surat kesanggupan BPJS yang valid.",
    "bpjs_kesanggupan.name_mismatch": "Nama pada surat kesanggupan BPJS tidak cocok.",
    "bpjs_kesanggupan.stamp_missing": "Surat kesanggupan BPJS wajib memakai materai merah.",
    "jkn.wrong_document": "Dokumen ini bukan layar Info Peserta JKN yang valid.",
    "jkn.name_mismatch": "Nama peserta JKN tidak cocok.",
    "iuran.wrong_document": "Dokumen ini bukan layar Info Iuran JKN yang valid.",
    "iuran.name_mismatch": "Nama pada Info Iuran tidak cocok.",
    "vaksinasi_1.wrong_document": "Dokumen ini bukan sertifikat vaksinasi dosis 1 yang valid.",
    "vaksinasi_1.name_mismatch": "Nama pada sertifikat vaksinasi dosis 1 tidak cocok.",
    "vaksinasi_2.wrong_document": "Dokumen ini bukan sertifikat vaksinasi dosis 2 yang valid.",
    "vaksinasi_2.name_mismatch": "Nama pada sertifikat vaksinasi dosis 2 tidak cocok.",
    "vaksinasi_3.wrong_document": "Dokumen ini bukan sertifikat vaksinasi booster yang valid.",
    "vaksinasi_3.name_mismatch": "Nama pada sertifikat vaksinasi booster tidak cocok.",
    # Pendidikan / lamaran
    "ijasah.wrong_document": "Dokumen ini bukan ijazah yang valid.",
    "ijasah.name_mismatch": "Nama pada ijazah tidak cocok.",
    "transkrip.wrong_document": "Dokumen ini bukan transkrip nilai yang valid.",
    "transkrip.name_mismatch": "Nama pada transkrip tidak cocok.",
    "formulir_okb.wrong_document": "Dokumen ini bukan Formulir OKB yang valid.",
    "formulir_okb.name_mismatch": "Nama pada Formulir OKB tidak cocok.",
    "formulir_lamaran.wrong_document": "Dokumen ini bukan formulir lamaran yang valid.",
    "formulir_lamaran.name_mismatch": "Nama pada formulir lamaran tidak cocok.",
    "surat_lamaran.wrong_document": "Dokumen ini bukan surat lamaran yang valid.",
    "surat_lamaran.name_mismatch": "Nama pada surat lamaran tidak cocok.",
    "keterangan_kesehatan.wrong_document": "Dokumen ini bukan surat keterangan kesehatan yang valid.",
    "keterangan_kesehatan.wrong_issuer": "Surat harus dari rumah sakit, klinik, atau puskesmas, bukan bidan.",
    "keterangan_kesehatan.name_mismatch": "Nama pada surat keterangan kesehatan tidak cocok.",
    "keterangan_kesehatan.not_healthy": "Surat tidak menyatakan bahwa yang bersangkutan dalam kondisi sehat.",
    "keterangan_kesehatan.date_missing": "Tanggal cek kesehatan tidak terbaca.",
    "keterangan_kesehatan.expired": "Surat keterangan kesehatan sudah kadaluarsa.",
    # Foto profil
    "foto_profile.not_profile_photo": "Ini bukan foto profil / pas foto.",
    "foto_profile.no_face": "Wajah tidak terdeteksi pada foto profil.",
    "foto_profile.multiple_faces": "Foto profil harus menampilkan satu wajah saja.",
    "foto_profile.face_size": "Ukuran wajah pada foto profil tidak sesuai (terlalu kecil atau terlalu besar).",
    "foto_profile.not_blue": "Latar foto profil harus biru solid seperti pas foto.",
    "foto_profile.detector_unavailable": "Deteksi wajah tidak tersedia di server. Coba lagi nanti.",
    # CV
    "cv.match_failed": "CV tidak memenuhi kriteria pencocokan.",
}

PIPELINE_KIND_TO_CODE: dict[str, str] = {
    "empty_file": "pipeline.empty_file",
    "empty_document_type": "pipeline.empty_document_type",
    "unknown_document_type": "pipeline.unknown_document_type",
    "preprocess_error": "pipeline.unreadable_image",
    "decode_error": "pipeline.unreadable_image",
    "empty_ocr": "pipeline.empty_ocr",
    "ocr_unavailable": "pipeline.ocr_unavailable",
    "health_ai_unavailable": "pipeline.health_ai_unavailable",
    "portrait_ai_unavailable": "pipeline.portrait_ai_unavailable",
    "npwp_ai_unavailable": "pipeline.npwp_ai_unavailable",
    "pemadanan_npwp_ai_unavailable": "pipeline.pemadanan_npwp_ai_unavailable",
    "rekening_ai_unavailable": "pipeline.rekening_ai_unavailable",
    "bpjs_ai_unavailable": "pipeline.bpjs_ai_unavailable",
    "iuran_ai_unavailable": "pipeline.iuran_ai_unavailable",
    "vaksinasi_1_ai_unavailable": "pipeline.vaksinasi_1_ai_unavailable",
    "vaksinasi_2_ai_unavailable": "pipeline.vaksinasi_2_ai_unavailable",
    "vaksinasi_3_ai_unavailable": "pipeline.vaksinasi_3_ai_unavailable",
    "mistral_unavailable": "pipeline.ocr_unavailable",
    "ocr_error": "pipeline.ocr_error",
    "ocr_value_error": "pipeline.ocr_error",
    "cv_unavailable": "pipeline.internal",
    "opencv_unavailable": "pipeline.internal",
    "internal": "pipeline.internal",
    "value_error": "pipeline.internal",
}


def message_for_code(code: str) -> str:
    return ERROR_CATALOG.get(code) or ERROR_CATALOG["generic.validation_failed"]


def _case(code: str) -> dict[str, str]:
    return {"code": code, "message": message_for_code(code)}


def frontend_error_usage() -> dict[str, Any]:
    """Kontrak baca field di frontend — ikut di OpenAPI / error-catalog."""
    return {
        "rule": (
            "Tampilkan error_message ke user. Jangan compose explanation.summary. "
            "Jika error_cases berisi lebih dari 1 item, tampilkan semua message."
        ),
        "valid_document": {
            "http_status": 200,
            "document_matched": True,
            "valid": True,
            "error_code": None,
            "error_message": None,
            "error_cases": [],
        },
        "invalid_document": {
            "http_status": 200,
            "document_matched": False,
            "valid": False,
            "error_code": "ktp.name_mismatch",
            "error_message": "Nama pada KTP tidak cocok.",
            "error_cases": [
                {"code": "ktp.name_mismatch", "message": "Nama pada KTP tidak cocok."}
            ],
            "read_from": [
                "POST /api/v1/pipeline → root.error_code / root.error_message / root.error_cases "
                "(sama di validation.*)",
                "POST /systems/validation/api/v1/validate-document → root",
                "POST /systems/validation/api/v1/validate-foto-profile → root",
            ],
        },
        "transport_error": {
            "http_status": "400 | 503",
            "body": {
                "detail": {
                    "error_code": "pipeline.empty_file",
                    "error_message": "File kosong. Unggah gambar dokumen.",
                    "error_cases": [
                        {
                            "code": "pipeline.empty_file",
                            "message": "File kosong. Unggah gambar dokumen.",
                        }
                    ],
                }
            },
            "read_from": "response.detail.error_message (FastAPI HTTPException)",
        },
        "health_ai_transport_error": {
            "http_status": 503,
            "when": "POST /api/v1/pipeline + document_type=keterangan_kesehatan, Ornith gagal",
            "body": {
                "detail": {
                    "error_code": "pipeline.health_ai_unavailable",
                    "error_message": "OCR AI surat kesehatan tidak merespons. Coba lagi nanti.",
                    "error_cases": [
                        {
                            "code": "pipeline.health_ai_unavailable",
                            "message": "OCR AI surat kesehatan tidak merespons. Coba lagi nanti.",
                        }
                    ],
                    "code": "HEALTH_AI_UNAVAILABLE",
                    "engine": "ornith",
                }
            },
        },
        "health_letter_ai_body_error": {
            "http_status": 200,
            "when": "POST /api/v1/health-letter-ai — baca body.error, bukan HTTP status",
            "examples": {
                "empty_content": "Model tidak mengisi content (thinking/token)",
                "invalid_json": "Jawaban bukan JSON valid",
            },
        },
        "typescript": (
            "type UserErrorCase = { code: string; message: string };\n"
            "type ValidationErrors = {\n"
            "  error_code: string | null;\n"
            "  error_message: string | null;\n"
            "  error_cases: UserErrorCase[];\n"
            "};"
        ),
    }


def error_catalog_for_api() -> dict[str, Any]:
    """Daftar semua kode + pesan, dikelompokkan per dokumen untuk frontend."""
    by_doc: dict[str, list[dict[str, str]]] = {}
    for code, message in ERROR_CATALOG.items():
        doc = code.split(".", 1)[0]
        by_doc.setdefault(doc, []).append({"code": code, "message": message})
    return {
        "catalog": ERROR_CATALOG,
        "by_document": by_doc,
        "usage": frontend_error_usage(),
    }


def user_error_http_detail(code: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Body `detail` untuk HTTPException (400/503) — field sama dengan validasi gagal."""
    case = _case(code)
    detail: dict[str, Any] = {
        "error_code": case["code"],
        "error_message": case["message"],
        "error_cases": [case],
    }
    if extra:
        for key, val in extra.items():
            if key not in detail:
                detail[key] = val
    return detail


def pipeline_http_detail(
    error_kind: str | None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pe = pipeline_user_error(error_kind)
    return user_error_http_detail(pe["code"], extra=extra)


def pipeline_user_error(error_kind: str | None) -> dict[str, str]:
    code = PIPELINE_KIND_TO_CODE.get((error_kind or "").strip(), "pipeline.internal")
    return _case(code)


def attach_user_errors(validation: dict[str, Any]) -> dict[str, Any]:
    """Isi error_code, error_message, error_cases pada hasil validasi."""
    out = dict(validation)
    if out.get("document_matched") is True:
        out["error_code"] = None
        out["error_message"] = None
        out["error_cases"] = []
        return out

    cases = infer_error_cases(out)
    if not cases:
        cases = [_case("generic.validation_failed")]
    out["error_cases"] = cases
    out["error_code"] = cases[0]["code"]
    out["error_message"] = cases[0]["message"]
    return out


def infer_error_cases(validation: dict[str, Any]) -> list[dict[str, str]]:
    profile = str(validation.get("document_profile_id") or "").strip().casefold()
    blockers = list((validation.get("explanation") or {}).get("primary_blockers") or [])
    type_failed = (
        "DOCUMENT_TYPE" in blockers or validation.get("document_type_pass") is False
    )
    cases: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(code: str) -> None:
        if code in seen:
            return
        seen.add(code)
        cases.append(_case(code))

    portrait_ai = profile == "foto_profile" and validation.get("validation_engine") == "ornith"
    npwp_ai = profile == "npwp" and validation.get("validation_engine") == "ornith"
    pemadanan_ai = profile == "pemadanan_npwp" and validation.get("validation_engine") == "ornith"
    if portrait_ai:
        if validation.get("is_profile_photo") is False:
            add("foto_profile.not_profile_photo")
        if validation.get("is_blue_background") is False:
            add("foto_profile.not_blue")
    if npwp_ai:
        if validation.get("is_npwp") is False:
            add("npwp.wrong_document")
        else:
            if validation.get("nik_pass") is False:
                add("npwp.nik_mismatch")
            if validation.get("name_pass") is False and validation.get("nik_pass") is not True:
                add("npwp.name_mismatch")
    if pemadanan_ai:
        if validation.get("is_pemadanan_npwp") is False:
            add("pemadanan_npwp.wrong_document")
        else:
            if validation.get("nik_pass") is False:
                add("pemadanan_npwp.nik_mismatch")
            if validation.get("name_pass") is False and validation.get("nik_pass") is not True:
                add("pemadanan_npwp.name_mismatch")

    # 1) Gate kebijakan (meski jenis dokumen sudah lolos).
    if "BPJS_KESANGGUPAN_STAMP" in blockers or (
        profile == "bpjs_kesanggupan" and validation.get("stamp_pass") is False
    ):
        add("bpjs_kesanggupan.stamp_missing")

    if "SKCK_EXPIRY" in blockers or (
        profile == "skck" and validation.get("skck_expiry_pass") is False
    ):
        reason = str((validation.get("skck_expiry") or {}).get("reason") or "")
        until = validation.get("skck_valid_until")
        if "tidak terbaca" in reason or not until:
            add("skck.expiry_missing")
        elif "belum berlaku" in reason:
            add("skck.not_yet_valid")
        else:
            add("skck.expired")

    if "HEALTH_ISSUER" in blockers or (
        profile == "keterangan_kesehatan" and validation.get("issuer_pass") is False
    ):
        add("keterangan_kesehatan.wrong_issuer")

    if "HEALTH_STATUS" in blockers or (
        profile == "keterangan_kesehatan" and validation.get("health_status_pass") is False
    ):
        add("keterangan_kesehatan.not_healthy")

    if "HEALTH_CHECK_EXPIRY" in blockers or (
        profile == "keterangan_kesehatan" and validation.get("health_expiry_pass") is False
    ):
        reason = str((validation.get("health_check") or {}).get("reason") or "").casefold()
        if "tidak terbaca" in reason:
            add("keterangan_kesehatan.date_missing")
        else:
            add("keterangan_kesehatan.expired")

    if "HELPER_MARITAL" in blockers or (
        profile == "ktp" and validation.get("marital_pass") is False
    ):
        add("ktp.helper_not_single")

    # 2) Jenis dokumen / exclusion / foto.
    if not portrait_ai and "FACE_DETECTOR" in blockers:
        add("foto_profile.detector_unavailable")
    if not portrait_ai and "FACE" in blockers:
        img = validation.get("image_validation") or {}
        count = img.get("face_count")
        if count == 0 or count is None:
            add("foto_profile.no_face")
        elif isinstance(count, int) and count > 1:
            add("foto_profile.multiple_faces")
        else:
            add("foto_profile.face_size")
    if not portrait_ai and "BLUE_BACKGROUND" in blockers:
        add("foto_profile.not_blue")

    if validation.get("exclusion_violated") is True:
        if profile == "rekening":
            add("rekening.wrong_type_statement")
        elif profile:
            add(f"{profile}.wrong_document")
    elif type_failed and not portrait_ai and not npwp_ai and not pemadanan_ai:
        if profile == "foto_profile":
            if not any(c["code"].startswith("foto_profile.") for c in cases):
                add("foto_profile.not_blue")
        elif profile == "cv":
            add("cv.match_failed")
        elif profile:
            add(f"{profile}.wrong_document")

    # 3) Isi dokumen (hanya jika jenis sudah lolos).
    if not type_failed:
        if "BANK" in blockers or validation.get("bank_pass") is False:
            if profile == "mutasi":
                add("mutasi.wrong_bank")
            elif profile == "rekening":
                add("rekening.wrong_bank")

        if profile == "rekening":
            if validation.get("identity_pass") is False:
                add("rekening.name_mismatch")
            if validation.get("account_pass") is False or "ACCOUNT" in blockers:
                add("rekening.account_mismatch")
            elif "IDENTITY_AND_ACCOUNT" in blockers:
                if validation.get("identity_pass") is not True:
                    add("rekening.name_mismatch")
                if validation.get("account_pass") is not True:
                    add("rekening.account_mismatch")

        if profile == "mutasi" and (
            validation.get("account_pass") is False or "ACCOUNT" in blockers
        ):
            add("mutasi.account_mismatch")

        if (
            not npwp_ai
            and not pemadanan_ai
            and ("IDENTITY" in blockers or validation.get("identity_pass") is False)
        ):
            ident = validation.get("identity") or {}
            if profile in {"npwp", "pemadanan_npwp"} or ident.get("npwp_nik_identity"):
                add(f"{profile or 'npwp'}.nik_mismatch")
            elif profile == "kk" or ident.get("kk_or_identity"):
                add("kk.identity_mismatch")
            elif profile and profile not in {"rekening", "mutasi"}:
                add(f"{profile}.name_mismatch")

    if profile == "cv" and validation.get("document_matched") is False and not cases:
        add("cv.match_failed")

    filtered: list[dict[str, str]] = []
    for item in cases:
        code = item["code"]
        if code in ERROR_CATALOG:
            filtered.append(item)
        elif code.endswith(".name_mismatch"):
            lbl = profile_label(profile) or "dokumen"
            filtered.append({"code": code, "message": f"Nama pada {lbl} tidak cocok."})
        elif code.endswith(".wrong_document"):
            lbl = profile_label(profile) or "dokumen"
            filtered.append({"code": code, "message": f"Dokumen ini bukan {lbl} yang valid."})
        elif code.endswith(".nik_mismatch"):
            filtered.append({"code": code, "message": "NIK/nomor NPWP 16 digit tidak cocok."})
    return filtered


class UserErrorCase(BaseModel):
    """Satu case gagal — tampilkan `message` di UI."""

    code: str = Field(..., examples=["ktp.name_mismatch"], description="Kode tetap, tidak berubah antar rilis.")
    message: str = Field(
        ...,
        examples=["Nama pada KTP tidak cocok."],
        description="Kalimat siap tampil di frontend.",
    )


class FrontendErrorFields(BaseModel):
    """Field error di respons HTTP 200 (dokumen tidak lolos). Juga ada di `validation`."""

    error_code: str | None = Field(
        None,
        examples=["ktp.name_mismatch"],
        description="Kode case utama. null jika document_matched=true.",
    )
    error_message: str | None = Field(
        None,
        examples=["Nama pada KTP tidak cocok."],
        description="Pesan utama untuk UI. null jika lolos.",
    )
    error_cases: list[UserErrorCase] = Field(
        default_factory=list,
        description="Semua case gagal. Panjang > 1 jika misalnya nama dan norek sama-sama gagal.",
    )


class HttpUserErrorDetail(BaseModel):
    """Isi `detail` pada HTTP 400/503 (FastAPI HTTPException)."""

    error_code: str = Field(..., examples=["pipeline.empty_file"])
    error_message: str = Field(..., examples=["File kosong. Unggah gambar dokumen."])
    error_cases: list[UserErrorCase]


class HttpUserErrorBody(BaseModel):
    detail: HttpUserErrorDetail


class PipelineFrontendResponse(FrontendErrorFields):
    """Subset field yang dibaca frontend. Respons asli juga berisi ocr, validation, verdict, timing."""

    success: bool = True
    document_matched: bool | None = Field(
        None,
        description="true = lolos. false = tampilkan error_message. HTTP tetap 200.",
    )
    valid: bool | None = Field(None, description="Alias document_matched (foto/CV).")


class ErrorCatalogResponse(BaseModel):
    success: bool = True
    catalog: dict[str, str] = Field(..., description="Map error_code → error_message.")
    by_document: dict[str, list[UserErrorCase]] = Field(
        ...,
        description="Katalog dikelompokkan per prefix dokumen (ktp, rekening, skck, …).",
    )
    usage: dict[str, Any] = Field(
        default_factory=frontend_error_usage,
        description="Cara baca field di frontend, termasuk contoh TypeScript.",
    )
