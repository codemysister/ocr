"""Tes parser status perkawinan KTP (tanpa OCR)."""

from systems.validation.ktp_marital import extract_ktp_marital_status
from systems.validation.fuzzy_compare import validate_document_ocr


def _st(text: str) -> str | None:
    return extract_ktp_marital_status(text).get("status")


def test_clean_labeled_belum_kawin() -> None:
    assert _st("Agama : ISLAM Status Perkawinan : BELUM KAWIN Pekerjaan : PELAJAR") == "belum_kawin"


def test_ocr_extra_r_on_perkawinan() -> None:
    assert _st("Agama ISLAM Status Perkawinanr BELUM KAWIN Pekerjaan PELAJAR/MAHASISWA") == "belum_kawin"


def test_glued_perkawinarc_belum() -> None:
    assert _st("Status PerkawinarcBELUM KAWIN Pekerjaan PELAJARAMAHASISWA") == "belum_kawin"


def test_compact_and_typos() -> None:
    assert _st("status perkawinan belunkawin") == "belum_kawin"
    assert _st("BLM KAWIN") == "belum_kawin"
    assert _st("belom kawin") == "belum_kawin"


def test_kawin_not_from_label_word() -> None:
    assert _st("Status Perkawinan Pekerjaan PELAJAR Kewarganegaraan WNI") is None
    assert _st("Status Perkawinan : KAWIN Pekerjaan KARYAWAN SWASTA") == "kawin"


def test_cerai() -> None:
    assert _st("Status Perkawinan : CERAI HIDUP") == "cerai"


def test_ocr_elwin_fragment_is_belum_kawin() -> None:
    assert _st("Kecamatan AI SLAM gama ELWIN ua ekraan PELAJARMAHASISWA") == "belum_kawin"


def test_glued_status_perkawinan_belum() -> None:
    assert _st("Status Perkawinanc BELUM KAWIN Pekerjaan BELUM/TIDAKBEKERJA") == "belum_kawin"


def test_aisyah_passthrough_garbage_not_false_kawin() -> None:
    garbage = """
    PROVINSIJAWABARAT KABUPATENBEKASI NIK 3275025605070009
    Nama AISYAHNURHAFIDZAH Agama ISLAM EA ESI kan E
    Kewarganegaraan: WNI Berlaku Hingga: SEUMUR HIDUP
    """
    assert _st(garbage) is None


def test_ktp_compact_name_in_ocr_recovers_wrong_candidate() -> None:
    ocr = """
    PROVINSI JAWA BARAT KABUPATEN BEKASI
    NIK 3275030506070007
    Nama ANDIENI AZRIL
    Tempat/Tgl Lahir BEKASI, 05-06-2007
    Jenis Kelamin LAKI-LAKI
    Agama ISLAM Status Perkawinan BELUM KAWIN
    Kewarganegaraan WNI
    xxxrezanurhaikalyyy
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="ktp",
        expected_name="Reza Nur Haikal",
    )
    assert out["document_matched"] is True
    assert out["identity"]["ocr_compact_name_fallback"] is True


def test_ktp_compact_name_does_not_pass_shared_token_only() -> None:
    ocr = """
    NIK 3275045610060008 Nama RAHMALIA ANGGUN AGUSTIN
    Provinsi JAWA BARAT Kabupaten BEKASI Agama ISLAM
    Status Perkawinan BELUM KAWIN Kewarganegaraan WNI
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="ktp",
        expected_name="Anggun Ramadhani",
    )
    assert out["document_matched"] is False
    assert not out.get("identity", {}).get("ocr_compact_name_fallback")


def test_npwp_last_name_and_nik_suffix_recover_ocr_typo() -> None:
    ocr = """
    npwp. KANTOR PELAYANAN PAJAK PRATAMA JAKARTA KOJA
    172 0458 0906 0002
    JNGA SABRIYA
    TANGGAL TERDAFTAR 16/06/2026
    djp
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="npwp",
        expected_name="Bunga Sabriya",
        expected_nik="3172045809060002",
    )
    assert out["document_matched"] is True
    assert out["identity_pass"] is True
    assert out["identity"]["npwp_nik_identity"] is True


def test_npwp_wrong_name_still_passes_when_nik_matches() -> None:
    ocr = """
    npwp. KANTOR PELAYANAN PAJAK PRATAMA CIAWI
    3201 1755 1203 0002
    ORANG LAIN
    TANGGAL TERDAFTAR 01/12/2023
    djp
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="npwp",
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert out["identity_pass"] is True
    assert out["document_matched"] is True


def test_npwp_right_name_fails_when_nik_mismatch() -> None:
    ocr = """
    npwp. KANTOR PELAYANAN PAJAK PRATAMA CIAWI
    1111 2222 3333 4444
    SRI PUTRI
    TANGGAL TERDAFTAR 01/12/2023
    djp
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="npwp",
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert out["identity_pass"] is False
    assert out["document_matched"] is False


def test_npwp_nik_match_passes_even_if_ocr_looks_like_ktp() -> None:
    ocr = """
    PROVINSI JAWA BARAT KOTA DEPOK
    NIK : 3276025406070011
    Nama : INDAH AMELIA PUTRI
    Tempat/Tgl Lahir DEPOK Agama ISLAM
    Status Perkawinan BELUM KAWIN RT/RW 001/008
    Kel/Desa LEUWINANGGUNG Kecamatan TAPOS
    Kewarganegaraan WNI Berlaku Hingga SEUMUR HIDUP
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="npwp",
        expected_name="Indah amelia putri",
        expected_nik="3276025406070011",
    )
    assert out["identity_pass"] is True
    assert out["document_type_pass"] is True
    assert out["document_matched"] is True


def test_npwp_new_card_keywords_and_name() -> None:
    ocr = """
    npwp. KANTOR PELAYANAN PAJAK PRATAMA CIAWI
    3201 1755 1203 0002
    SRI PUTRI
    TANGGAL TERDAFTAR 01/12/2023
    djp
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="npwp",
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert out["document_matched"] is True
    assert out["document_type_pass"] is True
    assert out["identity_pass"] is True


def test_npwp_name_plus_tax_id_recovers_weak_keywords() -> None:
    ocr = """
    dMd
    32011755 1203 0002
    SRI PUTRI
    KH ABDUL HAMID NO.14 RT. 001 RW. 007, CIBITUNG KULON
    """
    out = validate_document_ocr(
        ocr,
        document_profile_id="npwp",
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert out["identity_pass"] is True
    assert out["document_matched"] is True


_KK_OCR = """
Kartu Keluarga No. 3173030710200004
Kepala Keluarga BUDI SANTOSO
NIK
1 BUDI SANTOSO 3216061012920024
2 SRI PUTRI 3201175512030002
"""


def test_kk_name_or_nik() -> None:
    ok = validate_document_ocr(
        _KK_OCR,
        document_type="kk",
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert ok["identity"]["kk_name_pass"] is True
    assert ok["identity"]["kk_nik_pass"] is True
    assert ok["identity_pass"] is True
    assert ok["document_matched"] is True

    name_only = validate_document_ocr(
        _KK_OCR,
        document_type="kk",
        expected_name="Sri Putri",
        expected_nik="3276025406070011",
    )
    assert name_only["identity"]["kk_name_pass"] is True
    assert name_only["identity"]["kk_nik_pass"] is False
    assert name_only["identity_pass"] is True
    assert name_only["document_matched"] is True

    nik_only = validate_document_ocr(
        _KK_OCR,
        document_type="kk",
        expected_name="Orang Lain Saja",
        expected_nik="3201175512030002",
    )
    assert nik_only["identity"]["kk_nik_pass"] is True
    assert nik_only["identity"]["kk_name_pass"] is False
    assert nik_only["identity_pass"] is True
    assert nik_only["document_matched"] is True

    neither = validate_document_ocr(
        _KK_OCR,
        document_type="kk",
        expected_name="Orang Lain Saja",
        expected_nik="3276025406070011",
    )
    assert neither["identity_pass"] is False
    assert neither["document_matched"] is False


def test_kk_nik_ocr_confusable_and_one_digit() -> None:
    ocr = """
    Kartu Keluarga No. 3215052110120098
    Kepala Keluarga AMIN
    NIK
    1 AMIN 3215051707730007
    2 MAWAR 32150544090600C4
    3 MUHAMAD EKA GALIH PERMANA 3215251708010003
    """
    mawar = validate_document_ocr(
        ocr,
        document_type="kk",
        expected_name="Bukan Nama",
        expected_nik="3215054409060004",
    )
    assert mawar["identity"]["kk_nik_pass"] is True
    assert mawar["identity_pass"] is True

    eka = validate_document_ocr(
        ocr,
        document_type="kk",
        expected_name="Bukan Nama",
        expected_nik="3215261708010003",
    )
    assert eka["identity"]["kk_nik_pass"] is True
    assert eka["identity_pass"] is True


_LIVIN_OK = """
Detail Rekening & Pengaturan
1330035980275
Tabungan NOW IDR
Sebagai Rekening Utama
Daftar Kartu
Visa Gold
4616994837398959
MELI SAPUTRI
"""

_LIVIN_NO_NAME = """
Detail Rekening & Pengaturan
1140034558968
Tabungan NOW IDR
Sebagai Rekening
Tampilkan di Intip Saldo
Terima Push Notification
"""


def test_rekening_requires_name_and_visible_account() -> None:
    ok = validate_document_ocr(
        _LIVIN_OK,
        document_type="rekening",
        expected_name="Meli Saputri",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert ok["account_pass"] is True
    assert ok["identity_pass"] is True
    assert ok["identity_or_account_pass"] is True
    assert ok["document_matched"] is True

    no_name = validate_document_ocr(
        _LIVIN_NO_NAME,
        document_type="rekening",
        expected_name="Meli Saputri",
        expected_bank="mandiri",
        expected_account="1140034558968",
    )
    assert no_name["account_pass"] is True
    assert no_name["identity_pass"] is False
    assert no_name["identity_or_account_pass"] is False
    assert no_name["document_matched"] is False

    visa_not_account = validate_document_ocr(
        _LIVIN_OK,
        document_type="rekening",
        expected_name="Meli Saputri",
        expected_bank="mandiri",
        expected_account="4616994837398",
    )
    assert visa_not_account["account_pass"] is False
    assert visa_not_account["document_matched"] is False

    leading_zero = validate_document_ocr(
        "Detail Rekening\nTabungan NOW IDR\n700013443119\nNADYA AMELIA SALSABILAH\n",
        document_type="rekening",
        expected_name="Nadya Amelia Salsabilah",
        expected_bank="mandiri",
        expected_account="0700013443119",
    )
    assert leading_zero["account_pass"] is True
    assert leading_zero["document_matched"] is True

    truncated = validate_document_ocr(
        "Livin\nDetail Rekening\n1670011368247\nNURMA ZAHROTU...\nTabungan NOW\n",
        document_type="rekening",
        expected_name="NURMA ZAHROTUNNISA",
        expected_bank="mandiri",
        expected_account="1670011368247",
    )
    assert truncated["identity_pass"] is True
    assert truncated["account_pass"] is True
    assert truncated["document_matched"] is True

    placeholder = validate_document_ocr(
        "Livin\nDetail Rekening\n1670011368858\nMAYA NOVA RIYANTI\nTabungan NOW\n",
        document_type="rekening",
        expected_name="Maya Nova Riyanti",
        expected_bank="mandiri",
        expected_account="0000000000000",
    )
    assert placeholder["identity_pass"] is True
    assert placeholder["account_pass"] is True
    assert placeholder["document_matched"] is True

    mutasi_ok = validate_document_ocr(
        "Tabungan NOW IDR\n1330035980275\nTransaksi\ne-Statement\nSetor Tunai\n",
        document_type="mutasi",
        expected_name="Meli Saputri",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert mutasi_ok["account_pass"] is True
    assert mutasi_ok["bank_pass"] is True
    assert mutasi_ok["document_matched"] is True

    mutasi_wrong = validate_document_ocr(
        "Tabungan NOW IDR\n1330035980275\nTransaksi\ne-Statement\n",
        document_type="mutasi",
        expected_bank="mandiri",
        expected_account="1560027756842",
    )
    assert mutasi_wrong["account_pass"] is False
    assert mutasi_wrong["document_matched"] is False

    nadya = validate_document_ocr(
        "Tabungan Payroll\n700013443119\nTransaksi\ne-Statement\n",
        document_type="mutasi",
        expected_bank="mandiri",
        expected_account="0700013443119",
    )
    assert nadya["account_pass"] is True
    assert nadya["bank_pass"] is True
    assert nadya["document_matched"] is True

    farah = validate_document_ocr(
        "Tabungan Payroll\n1360034647690\ne-Statement\n"
        "Transfer dari BANK MAS\nEFRIYANA D 1004489264\n",
        document_type="mutasi",
        expected_bank="mandiri",
        expected_account="1360034647690",
    )
    assert farah["account_pass"] is True
    assert farah["bank_pass"] is True
    assert farah["document_matched"] is True

    fitri = validate_document_ocr(
        "Tabungan NOW IDR\n1730021695987\nTransaksi\ne-Statement\n",
        document_type="mutasi",
        expected_bank="mandiri",
        expected_account="1730321695987",
    )
    assert fitri["account_pass"] is True
    assert fitri["bank_pass"] is True
    assert fitri["document_matched"] is True


def test_skck_expiry_must_not_be_past() -> None:
    from datetime import date

    from systems.validation.skck_validity import evaluate_skck_expiry_gate
    from systems.validation.fuzzy_compare import validate_document_ocr

    sample = (
        "SURAT KETERANGAN CATATAN KEPOLISIAN SKCK\n"
        "Nama NUR HOLIFAH\n"
        "Berlaku untuk Valid from 20 MEI 2026\n"
        "Sampai dengan Until 20 NOVEMBER 2026\n"
    )
    today = date(2026, 9, 24)
    gate = evaluate_skck_expiry_gate(sample, profile_id="skck", today=today)
    assert gate and gate["expiry_pass"] is True
    assert gate["valid_until"] == "2026-11-20"

    expired = evaluate_skck_expiry_gate(sample, profile_id="skck", today=date(2026, 11, 21))
    assert expired and expired["expiry_pass"] is False

    image = (
        "Untuk keperluan MELAMAR PEKERJAAN\n"
        "Berlaku dari tanggal : 04 DESEMBER 2025\n"
        "Valid from\n"
        "Sampai dengan : 04 JUNI 2026\n"
        "Until\n"
    )
    image_gate = evaluate_skck_expiry_gate(image, profile_id="skck", today=date(2026, 10, 27))
    assert image_gate and image_gate["expiry_pass"] is False
    assert image_gate["valid_from"] == "2025-12-04"
    assert image_gate["valid_until"] == "2026-06-04"

    ok = validate_document_ocr(
        sample,
        document_type="skck",
        expected_name="NUR HOLIFAH",
    )
    # live today; only assert expiry fields exist after gate
    assert ok.get("skck_expiry_pass") is not None

    expired_doc = validate_document_ocr(
        "SKCK Kepolisian\nNama NUR HOLIFAH\nSampai dengan 20 MEI 2024\n",
        document_type="skck",
        expected_name="NUR HOLIFAH",
    )
    assert expired_doc["skck_expiry_pass"] is False
    assert expired_doc["document_matched"] is False
    assert expired_doc["error_code"] == "skck.expired"
    assert expired_doc["error_message"] == "SKCK sudah kadaluarsa."


def test_skck_ai_expiry_and_name() -> None:
    from datetime import date

    from systems.validation.skck_ai import skck_validation_from_extraction
    from systems.validation.user_errors import attach_user_errors

    today = date(2026, 10, 15)

    def codes(parsed, name="Nur Holifah"):
        detail = attach_user_errors(
            skck_validation_from_extraction(parsed, expected_name=name, today=today)
        )
        return detail["document_matched"], detail.get("error_code")

    ok, code = codes(
        {
            "is_skck": True,
            "holder_name": "NUR HOLIFAH",
            "valid_from": "2026-05-20",
            "valid_until": "2026-11-20",
            "notes": "SKCK",
        }
    )
    assert ok is True and code is None

    ok, code = codes(
        {
            "is_skck": True,
            "holder_name": "NUR HOLIFAH",
            "valid_from": "2026-05-20",
            "valid_until": "2026-05-20",
        }
    )
    assert ok is False and code == "skck.expired"

    ok, code = codes(
        {"is_skck": True, "holder_name": "NUR HOLIFAH", "valid_from": "2026-05-20", "valid_until": ""}
    )
    assert ok is False and code == "skck.expiry_missing"

    ok, code = codes(
        {
            "is_skck": True,
            "holder_name": "NUR HOLIFAH",
            "valid_from": "2026-12-01",
            "valid_until": "2027-06-01",
        }
    )
    assert ok is False and code == "skck.not_yet_valid"

    ok, code = codes({"is_skck": False, "holder_name": "NUR HOLIFAH", "notes": "KTP"})
    assert ok is False and code == "skck.wrong_document"

    ok, code = codes(
        {
            "is_skck": True,
            "holder_name": "BUDI SANTOSO",
            "valid_from": "2026-05-20",
            "valid_until": "2026-11-20",
        }
    )
    assert ok is False and code == "skck.name_mismatch"

    ok, code = codes(
        {
            "is_skck": True,
            "holder_name": "ANGGUN RAMADHANI",
            "valid_from": "29 MEI 2026",
            "valid_until": "29 NOVEMBER 2026",
        },
        name="Anggun Ramadhani",
    )
    assert ok is True and code is None

    ok, code = codes(
        {
            "is_skck": True,
            "holder_name": "ANGGUN RAMADHANI",
            "valid_from": "2026-12-04",
            "valid_until": "2026-12-04",
            "valid_from_text": "04 DESEMBER 2025",
            "valid_until_text": "04 JUNI 2026",
            "notes": "Berlaku dari 04 DESEMBER 2025. Sampai dengan 04 JUNI 2026.",
        },
        name="Anggun Ramadhani",
    )
    assert ok is False and code == "skck.expired"

    swapped = skck_validation_from_extraction(
        {
            "is_skck": True,
            "holder_name": "RINDA SARI",
            "valid_from": "2026-04-16",
            "valid_from_text": "16 APRIL 2027",
            "valid_until": "2026-10-16",
            "valid_until_text": "16 OKTOBER 2026",
            "notes": "Valid dari 16 April 2027 sampai 16 Oktober 2026",
        },
        expected_name="Rinda Sari",
        today=date(2026, 11, 11),
    )
    assert swapped["skck_valid_from"] == "2026-10-16"
    assert swapped["skck_valid_until"] == "2027-04-16"
    assert swapped["skck_expiry_pass"] is True
    assert swapped["document_matched"] is True


_KESANGGUPAN_BODY = """
SURAT PERNYATAAN KESANGGUPAN
MENANGGUNG BIAYA BPJS KESEHATAN
Nama Lengkap : Evi Nurajiah
NIK Karyawan :
No. KTP : 3204037110071002
No. BPJS Kesehatan : 0000492530415
Jenis Kelamin : Perempuan
BPJS kesehatan/KIS Saya tidak aktif dan bersedia menanggung secara pribadi
mengalihkan kepesertaan menjadi peserta mandiri
pembayaran iuran BPJS Kesehatan melalui virtual account setelah 14 hari
sebagai syarat bekerja di PT Prakarsa Alam Segar
Hormat saya
( Evi Nurajiah )
"""


def test_bpjs_kesanggupan_requires_meterai() -> None:
    from systems.validation.bpjs_kesanggupan_stamp import ocr_has_meterai
    from systems.validation.fuzzy_compare import validate_document_ocr

    assert ocr_has_meterai("METERAI TEMPEL") is True
    assert ocr_has_meterai("e-Meterai") is True
    assert ocr_has_meterai("Materai Rp10000,-") is False
    assert ocr_has_meterai(_KESANGGUPAN_BODY) is False

    missing = validate_document_ocr(
        _KESANGGUPAN_BODY,
        document_type="bpjs_kesanggupan",
        expected_name="Evi Nurajiah",
    )
    assert missing["document_type_pass"] is True
    assert missing["stamp_pass"] is False
    assert missing["document_matched"] is False
    assert missing["error_code"] == "bpjs_kesanggupan.stamp_missing"
    assert missing["error_message"] == "Surat kesanggupan BPJS wajib memakai materai merah."

    stamped = validate_document_ocr(
        _KESANGGUPAN_BODY + "\nMETERAI TEMPEL\n",
        document_type="bpjs_kesanggupan",
        expected_name="Evi Nurajiah",
    )
    assert stamped["stamp_pass"] is True
    assert stamped["document_matched"] is True
    assert stamped.get("error_code") is None


def test_bpjs_kesanggupan_red_meterai_on_dataset_images() -> None:
    from pathlib import Path

    from systems.validation.bpjs_kesanggupan_stamp import image_has_red_meterai, ocr_has_meterai

    root = Path("dataset/surat kesanggupan bpjs")
    stamped = root / "surat kesanggupan bpjs_ADANAN HASBI AHMAD NASUTION_3216072612060008.jpg"
    empty_box = root / "surat kesanggupan bpjs_SITI AMARSHA ROSAFIA_1771054403050001.jpg"
    no_stamp = root / "surat kesanggupan bpjs_Puput Hendrawati_3216216509070006.jpg"
    mawar = root / "surat kesanggupan bpjs_Mawar_3215054409060004.jpg"
    nadia = root / "surat kesanggupan bpjs_Nadia Afritiarani_3175086301040002.jpg"
    if not stamped.is_file():
        return
    assert image_has_red_meterai(stamped.read_bytes()) is True
    assert image_has_red_meterai(empty_box.read_bytes()) is False
    assert image_has_red_meterai(no_stamp.read_bytes()) is False
    assert image_has_red_meterai(mawar.read_bytes()) is False
    assert image_has_red_meterai(nadia.read_bytes()) is False
    from systems.preprocessing.pipeline import encode_image_bytes_for_ocr_remote_safe

    mawar_pipeline, _ = encode_image_bytes_for_ocr_remote_safe(mawar.read_bytes())
    assert image_has_red_meterai(mawar_pipeline) is False
    stamped_pipeline, _ = encode_image_bytes_for_ocr_remote_safe(stamped.read_bytes())
    assert image_has_red_meterai(stamped_pipeline) is True
    assert ocr_has_meterai("Materai Rp10000,-") is False


def test_keterangan_kesehatan_sehat_and_expiry() -> None:
    from datetime import date

    from systems.validation.fuzzy_compare import validate_document_ocr
    from systems.validation.keterangan_kesehatan import evaluate_keterangan_kesehatan_gates

    letter = (
        "SURAT KETERANGAN KESEHATAN\n"
        "Dokter: dr. A\n"
        "Nama: CITRA AMELIA\n"
        "Berdasarkan hasil pemeriksaan, dinyatakan SEHAT.\n"
        "Bekasi, 15 Juni 2026\n"
    )
    gate_ok = evaluate_keterangan_kesehatan_gates(
        letter, profile_id="keterangan_kesehatan", today=date(2026, 6, 20)
    )
    assert gate_ok and gate_ok["health_status_pass"] is True
    assert gate_ok["health_expiry_pass"] is True
    assert gate_ok["exam_date"] == "2026-06-15"

    gate_expired = evaluate_keterangan_kesehatan_gates(
        letter, profile_id="keterangan_kesehatan", today=date(2026, 9, 27)
    )
    assert gate_expired and gate_expired["health_expiry_pass"] is True
    assert gate_expired["valid_until"] is None

    pass_doc = validate_document_ocr(
        letter,
        document_type="keterangan_kesehatan",
        expected_name="CITRA AMELIA",
        today=date(2026, 6, 20),
    )
    assert pass_doc["health_status_pass"] is True
    assert pass_doc["health_expiry_pass"] is True
    assert pass_doc["document_matched"] is True

    unfit = validate_document_ocr(
        "SURAT KETERANGAN KESEHATAN\nDokter: dr. A\nNama: CITRA AMELIA\n"
        "Dinyatakan tidak sehat.\nBekasi, 15 Juni 2026\n",
        document_type="keterangan_kesehatan",
        expected_name="CITRA AMELIA",
        today=date(2026, 6, 20),
    )
    assert unfit["health_status_pass"] is False
    assert unfit["document_matched"] is False
    assert unfit["error_code"] == "keterangan_kesehatan.not_healthy"

    missing_date = validate_document_ocr(
        "SURAT KETERANGAN KESEHATAN\nDokter: dr. A\nNama: ARIF SETIAWAN\n"
        "Sehat Jasmani dan Rohani\n",
        document_type="keterangan_kesehatan",
        expected_name="ARIF SETIAWAN",
        today=date(2026, 6, 20),
    )
    assert missing_date["health_status_pass"] is True
    assert missing_date["health_expiry_pass"] is False
    assert missing_date["error_code"] == "keterangan_kesehatan.date_missing"

    from systems.validation.keterangan_kesehatan_ai import _marked_healthy, _trailing_patient_name

    mora = {
        "document_type_label": "Surat Keterangan Dokter",
        "is_health_certificate": True,
        "holder_name": "Safinatunnisa",
        "health_conclusion": "tidak_jelas",
        "health_conclusion_text": (
            "menyatakan telah memeriksa kesehatan secara umum dari: Safinatunnisa"
        ),
    }
    assert _trailing_patient_name("Safinatunnisa", "Syahwa Safinatunnisa") is True
    assert _trailing_patient_name("Sari", "Rinda Sari") is False
    assert _trailing_patient_name("Budi Santoso", "Syahwa Safinatunnisa") is False
    assert _marked_healthy(mora, False) is True
    assert _marked_healthy({"health_conclusion": "tidak_sehat", "health_conclusion_text": "tidak sehat"}, False) is False

    expired = validate_document_ocr(
        "SURAT KETERANGAN KESEHATAN\nNama: CITRA AMELIA\ndinyatakan SEHAT.\n"
        "berlaku sampai dengan tanggal 01 Juni 2026\nBekasi, 01 Maret 2026\n",
        document_type="keterangan_kesehatan",
        expected_name="CITRA AMELIA",
        today=date(2026, 9, 27),
    )
    assert expired["health_expiry_pass"] is False
    assert expired["document_matched"] is False
    assert expired["error_code"] == "keterangan_kesehatan.expired"

    form = validate_document_ocr(
        "SURAT KETERANGAN DOKTER\nNama: WULAN PUTRI\n"
        "Kesimpulan : Sehat / Tidak Sehat\nCilamaya, 19/6/26\nDokter\n",
        document_type="keterangan_kesehatan",
        expected_name="Wulan Putri",
        today=date(2026, 6, 20),
    )
    assert form["health_status_pass"] is True
    assert form["identity_pass"] is True
    assert form["health_expiry_pass"] is True

    ocr_name = validate_document_ocr(
        "SURAT KETERANGAN SEHAT\nDokter: dr. A\n2a14an\nAviek\nRaomathan\nNama\n"
        "keadaan sehat.\n13-6-.2026\n",
        document_type="keterangan_kesehatan",
        expected_name="Zaidan Arief Ramadhan",
        today=date(2026, 6, 20),
    )
    assert ocr_name["identity_pass"] is True
    assert ocr_name["health_status_pass"] is True

    gina = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN KESEHATAN\nNama: GINA SRI WAHYUNI\n"
        "dinyatakan SEHAT.\nSurat keterangan ini berlaku selama 3 (tiga) bulan "
        "yaitu sampai dengan tanggal 02 September 2026\n"
        "Darangdan, 02 Juni 2026\nDokter pemeriksa\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert gina and gina["exam_date"] == "2026-06-02"
    assert gina["valid_until"] == "2026-09-02"
    assert gina["health_expiry_pass"] is False

    ayu = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN KESEHATAN\nNama: AYU SAFITRI\ndinyatakan SEHAT.\n"
        "KOTA BEKASI, 15 Juni 2026\nDokter\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert ayu and ayu["exam_date"] == "2026-06-15"
    assert ayu["valid_until"] is None
    assert ayu["health_expiry_pass"] is True

    amandah = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN DOKTER\nNama: AMANDAH NOVIYANI\n"
        "Hasil Pemeriksaan : Seha/Tidak Sehat\nBekasi, 12/06\n20.26\nDokter\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert amandah and amandah["health_status_pass"] is True
    assert amandah["exam_date"] == "2026-06-12"
    assert amandah["valid_until"] is None
    assert amandah["health_expiry_pass"] is True

    amandah_ocr = evaluate_keterangan_kesehatan_gates(
        "Hasil Pemeriksaan : Seha/Tidak Sehat\n12/06\nBekasi,\n20.26\nDokter Pemeriksa\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert amandah_ocr and amandah_ocr["exam_date"] == "2026-06-12"

    aiga = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN SEHAT\nNama: AIGA TARA NASUCHA\n"
        "keadaan Sehat / Tidak Schat\n22funi.\n26.\nBekasi,\n.20..\nDokter Pemeriksa\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert aiga and aiga["exam_date"] == "2026-06-22"

    cipto = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN DOKTER\nNama: Cipto Prayoga\nS E H A T\n"
        "10 Suni 2026\nBerlaku : Maksimal 3 Bulan\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert cipto and cipto["exam_date"] == "2026-06-10"
    assert cipto["valid_until"] == "2026-09-10"

    achmad = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN SEHAT\nNama: Achmad\nSehat/ Tidak Sehat\n"
        "30-06-2a6\nCikarang,\nDokter/Pemeriksa\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert achmad and achmad["exam_date"] == "2026-06-30"

    alfiah = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN DOKTER\nNama: ALFIAH ZAHRA\ndinyatakan SEHAT\n"
        "Bekasi, 11 Juni 2026\nYang memeriksa,\n"
        "Tgl.Cetak. 11 Juni 2026, 20.01\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert alfiah and alfiah["exam_date"] == "2026-06-11"
    assert alfiah["valid_until"] is None
    assert alfiah["health_expiry_pass"] is True

    abdullah = evaluate_keterangan_kesehatan_gates(
        "SURAT KETERANGAN SEHAT\nNama: ABDULLAH FARIQ PURNAMA\n"
        "Pada pemeriksaan tanggal 14 Juni 2026 dinyatakan berbadan SEHAT / TIDAK-SEHAT\n"
        "Karawang,14 Juni 2026\n",
        profile_id="keterangan_kesehatan",
        today=date(2026, 9, 27),
    )
    assert abdullah and abdullah["exam_date"] == "2026-06-14"
    assert abdullah["valid_until"] is None
    assert abdullah["health_expiry_pass"] is True


def test_format_id_date_for_benchmark() -> None:
    from systems.validation.skck_validity import format_id_date

    assert format_id_date("2026-05-06") == "6 Mei 2026"
    assert format_id_date("2026-06-15") == "15 Juni 2026"
    assert format_id_date(None) is None


def test_health_letter_ai_synthetic_ocr_and_url() -> None:
    from datetime import date

    from systems.validation.keterangan_kesehatan import evaluate_keterangan_kesehatan_gates
    from systems.validation.keterangan_kesehatan_ai import (
        _synthetic_ocr,
        assert_allowed_llm_url,
        health_letter_user_prompt,
    )

    blob = _synthetic_ocr(
        {
            "holder_name": "Taliyah",
            "health_conclusion": "sehat",
            "city": "Cidempet",
            "issued_date": "2026-05-06",
            "issued_date_raw": "Cidempet, 06 Mei 2026",
            "doctor_name": "dr. Aziz",
            "clinic_name": "Puskesmas Cidempet",
        }
    )
    gate = evaluate_keterangan_kesehatan_gates(
        blob, profile_id="keterangan_kesehatan", today=date(2026, 6, 20)
    )
    assert gate and gate["exam_date"] == "2026-05-06"
    assert gate["health_status_pass"] is True
    assert "Taliyah" in health_letter_user_prompt(expected_name="Taliyah")
    assert assert_allowed_llm_url("http://172.21.13.26:8080/v1").endswith("/v1")
    try:
        assert_allowed_llm_url("http://example.com/v1")
        raise AssertionError("public url must be rejected")
    except ValueError:
        pass

    from systems.validation.llm_fallback import _assistant_text

    empty_msg = {"role": "assistant", "content": "", "reasoning_content": '{"holder_name":"Aiga"}'}
    assert '"holder_name"' in _assistant_text(empty_msg)
    parts_msg = {"content": [{"type": "text", "text": '{"ok":true}'}]}
    assert _assistant_text(parts_msg) == '{"ok":true}'


def test_pipeline_health_letter_skips_paddle_uses_ornith() -> None:
    from app import pipeline_runner as pr
    from systems.validation.keterangan_kesehatan_ai import analyze_health_letter_image as real_ai

    calls = {"ocr_inner": 0}

    def fake_ai(image_bytes, **kwargs):
        return {
            "error": None,
            "extraction": {
                "is_health_certificate": True,
                "holder_name": "ANGGUN RAMADHANI",
                "health_conclusion": "sehat",
                "issued_date": "2026-06-04",
                "issued_date_raw": "Bekasi, 04 Juni 2026",
                "document_type_label": "Surat Keterangan Sehat",
            },
            "validation": {
                "issued_date": "2026-06-04",
                "health_exam_date": "2026-06-04",
                "document_matched": True,
                "document_type_pass": True,
                "issuer_pass": True,
                "issuer_type": "klinik",
                "health_status_pass": True,
                "health_expiry_pass": True,
                "identity_pass": True,
            },
            "llm": {"model": "ornith-1.5-9b", "base_url": "http://172.21.13.26:8080/v1"},
            "raw_content": "{}",
        }

    real_ocr = pr._run_ocr_inner

    def counting_ocr(*args, **kwargs):
        calls["ocr_inner"] += 1
        return real_ocr(*args, **kwargs)

    pr.analyze_health_letter_image = fake_ai
    pr._run_ocr_inner = counting_ocr
    try:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = pr.run_pipeline_bytes(
            png,
            document_type="keterangan_kesehatan",
            expected_name="ANGGUN RAMADHANI",
            filename="keterangan kesehatan_ANGGUN RAMADHANI_3275045610060008.jpg",
        )
        assert result.ok is True
        assert result.document_matched is True
        assert calls["ocr_inner"] == 0
        assert result.payload["ocr_engine"] == "ornith"
        assert result.payload["ocr"]["mode"] == "ornith_health_ai"
        assert result.payload["validation"]["health_exam_date"] == "2026-06-04"
    finally:
        pr.analyze_health_letter_image = real_ai
        pr._run_ocr_inner = real_ocr



def test_health_letter_issuer_rejects_bidan() -> None:
    from systems.validation.keterangan_kesehatan_ai import _issuer_type, overlay_health_ai_on_validation
    from systems.validation.user_errors import infer_error_cases

    assert _issuer_type({"clinic_name": "RSUD Bekasi", "issuer_type": "lain"}) == "rumah_sakit"
    assert _issuer_type({"clinic_name": "Klinik Sehat", "issuer_type": "lain"}) == "klinik"
    assert _issuer_type({"clinic_name": "Puskesmas Sukmajaya"}) == "puskesmas"
    assert _issuer_type({"clinic_name": "Klinik Bidan Melati", "issuer_type": "klinik"}) == "bidan"

    overlaid = overlay_health_ai_on_validation(
        {
            "document_profile_id": "keterangan_kesehatan",
            "document_type_pass": True,
            "document_matched": True,
            "health_status_pass": True,
            "health_expiry_pass": True,
            "identity_pass": True,
            "explanation": {"primary_blockers": []},
        },
        {
            "extraction": {"is_health_certificate": True, "issued_date": "2026-06-04"},
            "validation": {"issuer_type": "bidan", "issuer_pass": False, "issued_date": "2026-06-04"},
        },
    )
    assert overlaid["document_matched"] is False
    codes = [item["code"] for item in infer_error_cases(overlaid)]
    assert "keterangan_kesehatan.wrong_issuer" in codes


def test_keterangan_kesehatan_reads_ocr_dates() -> None:
    from systems.validation.keterangan_kesehatan import extract_health_check_dates

    visus = extract_health_check_dates(
        "Catatan: Visus ODS : 6/6\nBekasi, 11 Juni 2026\nYang memeriksa"
    )
    assert visus["exam_date"] == "2026-06-11"

    taliyah = extract_health_check_dates(
        "Cidempet 06 WE! ..202\nDokter UPTD Puskesmas Cidempet"
    )
    assert taliyah["exam_date"] == "2026-05-06"

    tita = extract_health_check_dates(
        "Bekasi, 15 / 6 / 20 26\nDokter yang memeriksa"
    )
    assert tita["exam_date"] == "2026-06-15"

    spaced = extract_health_check_dates("15 16 20 26 Bekasi, Dokter pemeriksa")
    assert spaced["exam_date"] == "2026-06-15"

    glued = extract_health_check_dates("Pegagan, 221052026 mmHg")
    assert glued["exam_date"] == "2026-05-22"

    slash = extract_health_check_dates("15 / 06 12026 Bekasi")
    assert slash["exam_date"] == "2026-06-15"

    taliyah_scattered = extract_health_check_dates(
        "Demikian surat keterangan ini kami buat.\n"
        "Catatan :\nWE!\n..202\nCidempet 06\n:55 cm\n"
        "okter UPTD Puskesmas Cidempet\n"
    )
    assert taliyah_scattered["exam_date"] == "2026-05-06"

    tita_scattered = extract_health_check_dates(
        "Pemeriksaan Khusus\n154\nTinggi Badan\ncm\n15\n16\n"
        "dr. Elira\n6.9 kg\nBerat Badan\n20\nPenglihatan\n"
        "Dokter yang memeriksa\n"
    )
    assert tita_scattered["exam_date"] == "2026-06-15"

    adinda_glued = extract_health_check_dates(
        "Berat Badan\nks\n1516\n20\n26\nPenglihatan\nDokter yang memeriksa"
    )
    assert adinda_glued["exam_date"] == "2026-06-15"

    putri_106 = extract_health_check_dates("15\n106 1 2026\nBekasi,\nDokter Pemeriksa")
    assert putri_106["exam_date"] == "2026-06-15"

    repaired_day = extract_health_check_dates("Tangerang, 66 / 11 / 2026")
    assert repaired_day["exam_date"] == "2026-11-06"

    not_iso_fragment = extract_health_check_dates(
        "berlaku selama 1 bulan yaitu sampai dengan tanggal 2026-07-11\n"
        "Tangerang, 06 / 11 / 2026"
    )
    assert not_iso_fragment["exam_date"] == "2026-11-06"

    signature = extract_health_check_dates("Tambun Bekasi, 09/08/2026")
    assert signature["exam_date"] == "2026-08-09"

    jahra_roman = extract_health_check_dates(
        "NO. SK5-03.\n15\nVI...\n1....2006\nKesimpulan SEHAT\nKUSUHABEDIK2026"
    )
    assert jahra_roman["exam_date"] == "2026-06-15"


def test_user_error_messages_per_case() -> None:
    rekening_name = validate_document_ocr(
        _LIVIN_NO_NAME,
        document_type="rekening",
        expected_name="Meli Saputri",
        expected_bank="mandiri",
        expected_account="1140034558968",
    )
    assert rekening_name["error_code"] == "rekening.name_mismatch"
    assert rekening_name["error_message"] == "Nama pemilik rekening tidak cocok."

    rekening_acct = validate_document_ocr(
        _LIVIN_OK,
        document_type="rekening",
        expected_name="Meli Saputri",
        expected_bank="mandiri",
        expected_account="4616994837398",
    )
    assert rekening_acct["error_code"] == "rekening.account_mismatch"

    ktp_name = validate_document_ocr(
        "NIK 3275045610060008 Nama RAHMALIA ANGGUN AGUSTIN Provinsi JAWA BARAT "
        "Status Perkawinan BELUM KAWIN Kewarganegaraan WNI",
        document_profile_id="ktp",
        expected_name="Reza Nur Haikal",
    )
    assert ktp_name["document_matched"] is False
    assert ktp_name["error_code"] == "ktp.name_mismatch"

    helper = validate_document_ocr(
        "PROVINSI JAWA BARAT NIK 3275030506070007 Nama ANDIENI AZRIL "
        "Status Perkawinan KAWIN Kewarganegaraan WNI",
        document_profile_id="ktp",
        expected_name="ANDIENI AZRIL",
        job_position="Helper",
    )
    assert helper["error_code"] == "ktp.helper_not_single"

    npwp_nik = validate_document_ocr(
        "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA DIREKTORAT JENDERAL PAJAK "
        "NPWP 10.0.1.5-012.000 NIK 3275015803060008 NAMA BUDI SANTOSO",
        document_profile_id="npwp",
        expected_name="Adinda Putri Khalista",
        expected_nik="1111111111111111",
    )
    assert npwp_nik["error_code"] == "npwp.nik_mismatch"

    mutasi = validate_document_ocr(
        "e-Statement Livin Mandiri\nNomor rekening 1330035980275\n",
        document_type="mutasi",
        expected_bank="mandiri",
        expected_account="1140034558968",
    )
    assert mutasi["error_code"] == "mutasi.account_mismatch"


_JKN_FAMILY = """
Info Peserta
Jenis Tampilan
Kepesertaan Terdaftar
HANIM
TIDAK AKTIF (NON AKTIF DIAKHIR BULAN)
0001768748027
Peserta ( PBPU dan BP Pemerintah Daerah )
Lahir 03-04-1972
Faskes 1 WANJAYA
SATI
(AKTIF)
0001768748038
Istri ( PBPU dan BP Pemerintah Daerah )
Lahir 05-07-1963
Faskes 1 WANJAYA
NUR HOLIFAH
TIDAK AKTIF (NON AKTIF DIAKHIR BULAN)
0001768748049
Anak ( PBPU dan BP Pemerintah Daerah )
Lahir 26-11-2006
Faskes 1 WANJAYA
"""


def test_jkn_extracts_aktif_status_without_blocking() -> None:
    aktif = validate_document_ocr(
        _JKN_FAMILY,
        document_type="jkn",
        expected_name="SATI",
    )
    assert aktif["document_type_pass"] is True
    assert aktif["identity_pass"] is True
    assert aktif["jkn_status_label"] == "aktif"
    assert aktif["jkn_status_display"] == "AKTIF"
    assert aktif["document_matched"] is True

    inactive = validate_document_ocr(
        _JKN_FAMILY,
        document_type="jkn",
        expected_name="NUR HOLIFAH",
    )
    assert inactive["identity_pass"] is True
    assert inactive["jkn_status_label"] == "tidak_aktif"
    assert inactive["jkn_status_display"] == "TIDAK AKTIF"
    assert inactive["document_matched"] is True
    assert inactive.get("error_code") is None

    hanim = validate_document_ocr(
        _JKN_FAMILY,
        document_type="jkn",
        expected_name="HANIM",
    )
    assert hanim["jkn_status_label"] == "tidak_aktif"
    assert hanim["document_matched"] is True


_JKN_WRAPPED_NAME = """
Info Peserta
Jenis Tampilan
Kepesertaan Terdaftar
SYAWQINA NAHJAN
32750363
FARAHIYAH ZIADEH
Kelas 3
(AKTIF)
0001632175536
Peserta (PBi Jaminan Kesehatan)
Lahir
23-05-2005
Faskes 1
Klinik Tiara Bunda
3275036305050001
"""

_JKN_PANDAWA_CHAT = """
Pandawa
PANDAWA
BPJS Kesehatan
3215135612050008
Nama Peserta: NEXXXXX RAXXXXXXX
Jenis Peserta: PBPU DAN BP
PEMERINTAH DAERAH
Status Peserta: AKTIF
Nama Peserta: IRXXX ATXXX PEXXXXX
Jenis Peserta: PBPU DAN BP
PEMERINTAH DAERAH
Status Peserta: TIDAK AKTIF - MENINGGAL
Nama Peserta: SIXX JUXXXXXX
Jenis Peserta: PBPU DAN BP
PEMERINTAH DAERAH
Status Peserta: AKTIF
Nama Peserta: GIXX MUXXXXX RIXXX
Jenis Peserta: PBPU DAN BP
PEMERINTAH DAERAH
Status Peserta: AKTIF
"""

_JKN_OCR_GLUED = """
Info Peserta
Jenis Tampilan
Kepesertaan Terdaftar
MUMUN PUSPITASARI
Kelas 3
(AKTIF)
ITIKHOLISAH
Kelas 3
AKTIF)
Faskes 1
KLINIK MEKAR MULIA
3216096103070004
"""

_JKN_NIK_ONLY = """
(AKTIF)
0001520363777
Peserta (PBPU dan BP
Pemerintah Daerah)
Lahir
02-11-2005
Faskes 1
KEBON PEDES
3202344211050002
"""

_JKN_NIK_FRAGMENT = """
Jenis Tampilan
Kepesertaan Terdaftar
SHARON REGINA
SANTOSO
Kelas 3
(AKTIF)
Anak (Pekerja Bukan Penerima Upah)
Lahir
25-10-2004
Faskes 1
Klinik Abdul Radjak
3022510040011
SANTOSO
Kelas 3
(AKTIF)
"""


def test_jkn_wrapped_and_pandawa_and_nik_fallback() -> None:
    wrapped = validate_document_ocr(
        _JKN_WRAPPED_NAME,
        document_type="jkn",
        expected_name="Syawqina Nahjan Farahiyah Ziadeh",
        expected_nik="3275036305050001",
    )
    assert wrapped["identity_pass"] is True
    assert wrapped["document_matched"] is True
    assert wrapped["jkn_status_label"] == "aktif"

    chat = validate_document_ocr(
        _JKN_PANDAWA_CHAT,
        document_type="jkn",
        expected_name="Siti jubaedah",
        expected_nik="3215135612050008",
    )
    assert chat["document_type_pass"] is True
    assert chat["identity_pass"] is True
    assert chat["document_matched"] is True
    assert chat["jkn_status_label"] == "aktif"
    assert chat.get("error_code") is None

    glued = validate_document_ocr(
        _JKN_OCR_GLUED,
        document_type="jkn",
        expected_name="Siti Kholisah",
        expected_nik="3216096103070004",
    )
    assert glued["identity_pass"] is True
    assert glued["document_matched"] is True

    nik_only = validate_document_ocr(
        _JKN_NIK_ONLY,
        document_type="jkn",
        expected_name="Selfia Nur Ramadani",
        expected_nik="3202344211050002",
    )
    assert nik_only["identity_pass"] is True
    assert nik_only["document_matched"] is True

    frag = validate_document_ocr(
        _JKN_NIK_FRAGMENT,
        document_type="jkn",
        expected_name="Shane Ricardo Santoso",
        expected_nik="3173022510040011",
    )
    assert frag["identity_pass"] is True
    assert frag["document_matched"] is True

    wilda = validate_document_ocr(
        """
Jenis Tampilan
Kepesertaan Terdaftar
PARIDA NASUTION
Kelas 3
(AKTIF)
Peserta (PBPU dan BP Pemerintah Daerah)
Lahir
26-06-1981
Faskes 1
SAYUR MATINGGI
ASRIN TARIHORAN
Kelas 3
(AKTIF)
Faskes 1
SAYUR MATINGGI
WILDA RISMA
Kelas 3
TARIHORAN
(AKTIF)
Lahir
03-07-2005
Faskes 1
SAYUR MATINGGI
""",
        document_type="jkn",
        expected_name="WILDA RISMA TARIHORAN",
        expected_nik="1203214307050001",
    )
    assert wilda["identity_pass"] is True
    assert wilda["document_matched"] is True

    rafi = validate_document_ocr(
        """
Kepesertaan Terdaftar
RUSMIYATI
Kelas 3
(AKTIF)
Peserta(Pekerja Bukan Penerima Upah)
Lahir
06-03-1970
Faskes 1
Klinik Trimedika
MUHAMMAD RAFI
Kelas 3
RAMADHAN
(AKTIF)
Anak (Pekerja Bukan Penerima Upah)
Lahir
02-10-2007
Faskes 1
Klinik Trimedika
""",
        document_type="jkn",
        expected_name="MUHAMMAD RAFI RAMADHAN",
        expected_nik="3175070210070003",
    )
    assert rafi["identity_pass"] is True
    assert rafi["document_matched"] is True

    rio = validate_document_ocr(
        """
Kelas 1
(AKTIF)
Peserta (Pekerja Penerima Upah)
Lahir
17-06-1996
Faskes1
KLINIKSRITINA2
""",
        document_type="jkn",
        expected_name="RIO BASTIAR",
        expected_nik="3216081706960003",
    )
    assert rio["identity_pass"] is True
    assert rio["document_matched"] is True

    anwar = validate_document_ocr(
        """
Nomor Kartu Keluarga
AGUSTINAOLIVIA
Kelas 3
PUTRI
(AKTIF)
Peserta (PBPU dan BP Pemerintah Daerah)
Lahir
16-08-2002
Faskes 1
Kelas 3
(AKTIF)
Anak(PBPU dan BP Pemerintah Daerah)
Lahir
04-10-2006
Puskesmas Makasar
""",
        document_type="jkn",
        expected_name="Muhammad Anwar",
        expected_nik="3175040410060006",
    )
    assert anwar["identity_pass"] is True
    assert anwar["document_matched"] is True

    april = validate_document_ocr(
        """
Info Peserta
AP***** SU************
Kelas 3
(AKTIF)
Anak (PBI (APBN))
Lahir
21--**_****
Faskes 1
Klinik PMI Purbalingga
""",
        document_type="jkn",
        expected_name="Aprilia Sulistyowati",
        expected_nik="3303056104040002",
    )
    assert april["identity_pass"] is True
    assert april["document_matched"] is True

    wrong_mask = validate_document_ocr(
        _JKN_PANDAWA_CHAT,
        document_type="jkn",
        expected_name="Budi Santoso",
        expected_nik="9999999999999999",
    )
    assert wrong_mask["identity_pass"] is False


def test_bpjs_kis_nik_truncated_name_and_keyword_recovery() -> None:
    tiara = validate_document_ocr(
        """
Kartu Indonesia Sehat
Nomor Kartu
0000472154512
Nama
TIARA
Alamat
BLOK KEDUNG RW 01
Tanggal Lahir
: 01 Januari 2007
NIK
Faskes Tingkat
:LOHBENER
Syarat dan Ketentuan:
Kartu Peserta harap dibawa ketika berobat.
""",
        document_type="bpjs",
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert tiara["identity_pass"] is True
    assert tiara["document_matched"] is True

    shella = validate_document_ocr(
        """
www.bpjs-kesehatan.go.id
BPJS Kesehatan Care Center 165
3. Apabila ada perubahan atau kehilangan Kartu, segera lapor ke kantor BPJS Kesehatan setempat.
Syarat dan Ketentuan:
Kel.Penggilingan I
Faskes
3175066010000006
NIK
Kartu Indonesia Sehat
Kartu Indonesia Sehat
""",
        document_type="bpjs",
        expected_name="Shella Octhaviani",
        expected_nik="3175066010000006",
    )
    assert shella["identity_pass"] is True
    assert shella["document_matched"] is True

    fazriah = validate_document_ocr(
        """
Kartu Indonesia Sehat
Nomor Kartu
0001011791801
Nama
Alamat
KP.BABAKAN 6/2 SATRIAJAYA, TAMBUN UTARA
Tanggal Lahir
08-05-2001
NIK
3216054805010004
Faskes Tingkat I
SRIAMUR
Syarat dan ketentuan
BPJS Kesehatan Care Center 165
""",
        document_type="bpjs",
        expected_name="NUR FAZRIAH FARADILA",
        expected_nik="3216054805010004",
    )
    assert fazriah["identity_pass"] is True
    assert fazriah["document_matched"] is True

    sefia = validate_document_ocr(
        """
Faskes
NIK
SEIA INGSIH
Nama
""",
        document_type="bpjs",
        expected_name="Sefia Ningsih",
        expected_nik="3212024709040001",
    )
    assert sefia["identity_pass"] is True
    assert sefia["document_type_pass"] is True
    assert sefia["document_matched"] is True

    wilda = validate_document_ocr(
        """
SAYVOI
120214070501
pda
BPJS Keschatar
WILDA RISMA TARHORAN
0003370376005
Kartu
Kartu Indonesia Sehat
Kartu Indonesia Sehat
""",
        document_type="bpjs",
        expected_name="WILDA RISMA TARIHORAN",
        expected_nik="1203214307050001",
    )
    assert wilda["exclusion_violated"] is False
    assert wilda["document_type_pass"] is True
    assert wilda["identity_pass"] is True
    assert wilda["document_matched"] is True

    pandu_kis = validate_document_ocr(
        """
Kartu Indonesia Sehat
Nomor Kartu
0003526475106
Nama
PANDU AJI PUTRA PRATAMA
Alamat
UJUNG HARAPAN 1/17
Tanggal Lahir
20-01-2006
NIK
3216022001060010
Faskes Tingkat I
BAHAGIA
Syarat dan Ketentuan:
BPJS Kesehatan Care Center 165
""",
        document_type="jkn",
        expected_name="Pandu aji Putra pratama",
        expected_nik="3216022001060010",
    )
    assert pandu_kis["exclusion_violated"] is False
    assert pandu_kis["identity_pass"] is True
    assert pandu_kis["document_type_pass"] is True
    assert pandu_kis["document_matched"] is True

    fheni_kis = validate_document_ocr(
        """
Syarat dan Ketentuan:
NIK
Alamat
FHENI SALSA OCTAVIANI
Nama
Nomor Kartu
0000509690924
BPJSKesehatan
Kartu Indonesia Sehat
Kartu Indonesia Sehat
""",
        document_type="jkn",
        expected_name="FHENI SALSA OCTAVIANI",
        expected_nik="3275065210990007",
    )
    assert fheni_kis["exclusion_violated"] is False
    assert fheni_kis["identity_pass"] is True
    assert fheni_kis["document_type_pass"] is True
    assert fheni_kis["document_matched"] is True


def test_truncated_jpeg_is_repaired_for_ocr() -> None:
    from pathlib import Path

    from systems.preprocessing.pipeline import encode_image_bytes_for_ocr_remote_safe

    p = Path("dataset/rekening bank/rekening bank_Vina Andriyani_1380024894169.jpg")
    if not p.is_file():
        return
    data = p.read_bytes()
    out, meta = encode_image_bytes_for_ocr_remote_safe(data)
    assert meta.get("truncated_repaired") is True
    assert out[:2] == b"\xff\xd8"
    assert out[-2:] == b"\xff\xd9"
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(out, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[0] >= 100 and img.shape[1] >= 100


def test_rekening_ai_accepts_partial_and_masked_name() -> None:
    from systems.validation.rekening_ai import rekening_validation_from_extraction

    base = {
        "is_savings_account": True,
        "is_statement": False,
        "bank": "mandiri",
        "account_number": "1330035980275",
        "confidence": 90,
    }
    partial = rekening_validation_from_extraction(
        {**base, "holder_name": "Deva"},
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert partial["name_pass"] is True
    assert partial["name_match_method"] == "partial"
    assert partial["document_matched"] is True

    masked = rekening_validation_from_extraction(
        {**base, "holder_name": "Dxxa"},
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert masked["name_match_method"] == "masked"
    assert masked["document_matched"] is True

    other = rekening_validation_from_extraction(
        {**base, "holder_name": "Budi"},
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert other["document_matched"] is False
    assert other["error_code"] == "rekening.name_mismatch"

    apostrophe = rekening_validation_from_extraction(
        {**base, "holder_name": "NAYLA SAIDAH"},
        expected_name="Nayla Sa'idah",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert apostrophe["name_pass"] is True
    assert apostrophe["document_matched"] is True

    passbook = rekening_validation_from_extraction(
        {
            **base,
            "is_front_page": False,
            "is_passbook": True,
            "holder_name": "AULIA FEBRIYZA",
            "account_number": "1560028468471",
            "notes": "Buku tabungan kertas dengan stempel cabang",
        },
        expected_name="Aulia Febriyza",
        expected_bank="mandiri",
        expected_account="1560028468471",
    )
    assert passbook["document_matched"] is False
    assert passbook["error_code"] == "rekening.wrong_document"

    glued = rekening_validation_from_extraction(
        {**base, "holder_name": "SYAHRUL ARRIZAL MAARIF"},
        expected_name="Syahrul Arrizal Ma'arif",
        expected_bank="mas",
        expected_account="1005992474",
    )
    assert glued["name_pass"] is True

    glued_words = rekening_validation_from_extraction(
        {**base, "holder_name": "Nayla andhrafitriyani", "account_number": "1560027773102"},
        expected_name="Nayla andhra fitriyani",
        expected_bank="mandiri",
        expected_account="1560027773102",
    )
    assert glued_words["document_matched"] is True

    missing_digits = rekening_validation_from_extraction(
        {**base, "holder_name": "Nadia Afritiarani", "account_number": "166000778061"},
        expected_name="Nadia Afritiarani",
        expected_bank="mandiri",
        expected_account="1660007778061",
    )
    assert missing_digits["account_pass"] is True
    assert missing_digits["document_matched"] is True

    two_missing = rekening_validation_from_extraction(
        {**base, "holder_name": "NAYLA SAIDAH", "account_number": "16700132666"},
        expected_name="Nayla Sa'idah",
        expected_bank="mandiri",
        expected_account="1670011326666",
    )
    assert two_missing["account_pass"] is True
    assert two_missing["document_matched"] is True

    hidden_name = rekening_validation_from_extraction(
        {**base, "holder_name": "Tabungan NOW IDR", "account_number": "1660007606288"},
        expected_name="abdullah fariq purnama",
        expected_bank="mandiri",
        expected_account="1660007606288",
    )
    assert hidden_name["document_matched"] is False
    assert hidden_name["error_code"] == "rekening.name_mismatch"

    no_owner = rekening_validation_from_extraction(
        {
            **base,
            "is_front_page": True,
            "holder_name": "",
            "account_number": "1730022558424",
            "notes": "Detail Rekening tanpa nama pemilik",
        },
        expected_name="Siti Aminah",
        expected_bank="mandiri",
        expected_account="1730022558424",
    )
    assert no_owner["document_matched"] is False
    assert no_owner["error_code"] == "rekening.name_mismatch"

    odd_length = rekening_validation_from_extraction(
        {**base, "holder_name": "ASHA FAUZIA RAHMA", "account_number": "60014403566"},
        expected_name="Asha Fauzia Rahma",
        expected_bank="mandiri",
        expected_account="1850003928840",
    )
    assert odd_length["document_matched"] is True
    assert odd_length["account_pass"] is False

    different_account = rekening_validation_from_extraction(
        {**base, "holder_name": "ASHA FAUZIA RAHMA", "account_number": "1670011380341"},
        expected_name="Asha Fauzia Rahma",
        expected_bank="mandiri",
        expected_account="1850003928840",
    )
    assert different_account["document_matched"] is False
    assert different_account["error_code"] == "rekening.account_mismatch"

    mas_home = rekening_validation_from_extraction(
        {
            **base,
            "is_savings_account": False,
            "bank": "mas",
            "holder_name": "Shane Ricardo Santoso",
            "account_number": "1005852068",
        },
        expected_name="Shane Ricardo Santoso",
        expected_bank="mas",
        expected_account="1005852068",
    )
    assert mas_home["document_matched"] is True

    statement = rekening_validation_from_extraction(
        {**base, "is_savings_account": False, "is_statement": True, "holder_name": "Deva Apriana"},
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert statement["error_code"] == "rekening.wrong_type_statement"

    transaction_home = rekening_validation_from_extraction(
        {
            **base,
            "is_front_page": True,
            "is_transaction_home": True,
            "holder_name": "Deva Apriana",
            "account_number": "1620011962644",
            "notes": "Tabungan NOW IDR. Saldo Rp 0. Transaksi. e-Statement. Transfer Rupiah.",
        },
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1620011962644",
    )
    assert transaction_home["document_matched"] is False
    assert transaction_home["error_code"] == "rekening.wrong_document"

    transaction_notes = rekening_validation_from_extraction(
        {
            **base,
            "is_front_page": True,
            "holder_name": "Deva Apriana",
            "account_number": "1620011962644",
            "notes": "Belum ada transaksi pada bulan ini. Top-up. Bayar/VA.",
        },
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1620011962644",
    )
    assert transaction_notes["document_matched"] is False
    assert transaction_notes["error_code"] == "rekening.wrong_document"

    product_list = rekening_validation_from_extraction(
        {
            **base,
            "is_front_page": True,
            "is_product_list": True,
            "holder_name": "Erika Rizqi Nurhayati",
            "account_number": "1560028474734",
            "notes": "Produk Anda. Tabungan & Giro Perorangan. Tabungan Emas. Lihat SPT.",
        },
        expected_name="Erika Rizqi Nurhayati",
        expected_bank="mandiri",
        expected_account="1560028474734",
    )
    assert product_list["document_matched"] is False
    assert product_list["error_code"] == "rekening.wrong_document"

    opening = rekening_validation_from_extraction(
        {
            **base,
            "is_front_page": True,
            "is_account_opening": True,
            "holder_name": "Erika Rizqi Nurhayati",
            "account_number": "1560028475350",
            "notes": "Yeay, Pembukaan Rekening Berhasil. Nomor Rekening. Cara Setor Dana. Lanjut ke Beranda.",
        },
        expected_name="Erika Rizqi Nurhayati",
        expected_bank="mandiri",
        expected_account="1560028475350",
    )
    assert opening["document_matched"] is False
    assert opening["error_code"] == "rekening.wrong_document"

    detail_card = rekening_validation_from_extraction(
        {
            **base,
            "is_savings_account": False,
            "is_front_page": False,
            "holder_name": "MEI NUR ADLUN",
            "account_number": "1760007000324",
            "notes": "Tabungan NOW IDR. Lihat Detail. Sebagai Rekening Utama. Daftar Kartu. GPN Silver.",
        },
        expected_name="Mei Nur Adlun",
        expected_bank="mandiri",
        expected_account="1760007000324",
    )
    assert detail_card["document_matched"] is True
    assert detail_card["error_code"] is None

    detail_settings = rekening_validation_from_extraction(
        {
            **base,
            "is_savings_account": False,
            "is_account_opening": False,
            "is_front_page": False,
            "is_transaction_home": False,
            "is_product_list": False,
            "is_passbook": False,
            "is_statement": False,
            "bank": "mandiri",
            "holder_name": "Mathew HF Sitanggang",
            "account_number": "1330020587671",
            "confidence": 90,
            "notes": "Halaman Detail Rekening Mandiri dengan nomor rekening dan nama pemilik.",
        },
        expected_name="Mathew HF Sitanggang",
        expected_bank="mandiri",
        expected_account="1330020587671",
    )
    assert detail_settings["is_savings_account"] is True
    assert detail_settings["is_front_page"] is True
    assert detail_settings["document_matched"] is True
    assert detail_settings["error_code"] is None


def test_npwp_ai_format_lama_flag() -> None:
    from systems.validation.npwp_ai import npwp_validation_from_extraction

    old = npwp_validation_from_extraction(
        {
            "is_npwp": True,
            "holder_name": "Bunga Sabriya",
            "npwp_number": "09.123.456.7-890.000",
            "confidence": 90,
        },
        expected_name="Bunga Sabriya",
        expected_nik="3172045809060002",
    )
    assert old["format_lama"] is True
    assert old["is_nik_format"] is False
    assert old["document_matched"] is True
    assert old["error_code"] is None

    new = npwp_validation_from_extraction(
        {
            "is_npwp": True,
            "holder_name": "Orang Lain",
            "npwp_number": "31.720458.0906.0002",
        },
        expected_name="Bunga Sabriya",
        expected_nik="3172045809060002",
    )
    assert new["format_lama"] is False
    assert new["document_matched"] is True

    mismatch = npwp_validation_from_extraction(
        {
            "is_npwp": True,
            "holder_name": "Sri Putri",
            "npwp_number": "1111222233334444",
        },
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert mismatch["format_lama"] is False
    assert mismatch["document_matched"] is False
    assert mismatch["error_code"] == "npwp.nik_mismatch"

    wrong_name = npwp_validation_from_extraction(
        {
            "is_npwp": True,
            "holder_name": "Orang Lain",
            "npwp_number": "09.123.456.7-890.000",
        },
        expected_name="Bunga Sabriya",
    )
    assert wrong_name["format_lama"] is True
    assert wrong_name["error_code"] == "npwp.name_mismatch"

    not_npwp = npwp_validation_from_extraction(
        {"is_npwp": False, "npwp_number": "", "holder_name": ""}
    )
    assert not_npwp["error_code"] == "npwp.wrong_document"


def test_pemadanan_npwp_ai_identity() -> None:
    from systems.validation.bpjs_ai import parse_bpjs_name_from_filename
    from systems.validation.pemadanan_npwp_ai import pemadanan_validation_from_extraction

    ok = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "holder_name": "ADANAN HASBI AHMAD NASUTION",
            "npwp_number": "3216072612060008",
        },
        expected_name="ADANAN HASBI AHMAD NASUTION",
        expected_nik="3216072612060008",
    )
    assert ok["document_matched"] is True
    assert ok["name_match_method"] == "full"

    truncated = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "holder_name": "ADANAN HASBI",
            "npwp_number": "32.160726.1206.0008",
        },
        expected_name="ADANAN HASBI AHMAD NASUTION",
        expected_nik="3216072612060008",
    )
    assert truncated["document_matched"] is True

    nik_saves = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "holder_name": "ORANG LAIN",
            "npwp_number": "3216072612060008",
        },
        expected_name="ADANAN HASBI AHMAD NASUTION",
        expected_nik="3216072612060008",
    )
    assert nik_saves["document_matched"] is True
    assert nik_saves["name_match_method"] == "nik"

    mismatch = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "holder_name": "ORANG LAIN",
            "npwp_number": "1111222233334444",
        },
        expected_name="ADANAN HASBI AHMAD NASUTION",
        expected_nik="3216072612060008",
    )
    assert mismatch["document_matched"] is False
    codes = [c["code"] for c in mismatch["error_cases"]]
    assert "pemadanan_npwp.nik_mismatch" in codes
    assert "pemadanan_npwp.name_mismatch" in codes

    card = pemadanan_validation_from_extraction(
        {"is_pemadanan_npwp": False, "holder_name": "ADANAN", "npwp_number": "", "notes": "Ini KTP."}
    )
    assert card["error_code"] == "pemadanan_npwp.wrong_document"

    new_card = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "document_kind": "kartu_npwp",
            "holder_name": "BUNGA SABRIYA",
            "npwp_number": "3172045809060002",
            "notes": "Kartu NPWP 16 digit.",
        },
        expected_name="Bunga Sabriya",
        expected_nik="3172045809060002",
    )
    assert new_card["document_matched"] is True

    bare_nik = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": False,
            "holder_name": "",
            "npwp_number": "3172045809060002",
            "notes": "Hanya ada NIK 16 digit.",
        },
        expected_name="Bunga Sabriya",
        expected_nik="3172045809060002",
    )
    assert bare_nik["document_matched"] is False
    assert bare_nik["error_code"] == "pemadanan_npwp.wrong_document"

    old_card = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "holder_name": "WAHYU BAGASKORO",
            "npwp_number": "10.999.045.7-523.000",
        },
        expected_name="Wahyu Bagaskoro",
        expected_nik="3305022108060001",
    )
    assert old_card["error_code"] == "pemadanan_npwp.wrong_document"

    aktivasi = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": False,
            "holder_name": "NELA NAILA NURHIKMAH",
            "npwp_number": "",
            "nik": "3214066412070001",
            "notes": "Surat keterangan aktivasi NIK.",
        },
        expected_name="NELA NAILA NURHIKMAH",
        expected_nik="3214066412070001",
    )
    assert aktivasi["document_matched"] is True
    assert aktivasi["nik"] == "3214066412070001"

    login = pemadanan_validation_from_extraction(
        {
            "is_pemadanan_npwp": True,
            "holder_name": "",
            "npwp_number": "3215045608040002",
            "notes": "Selamat Datang. ID Pengguna. Kata Sandi. Masuk. Coretax DJP.",
        },
        expected_name="Siti Aminah",
        expected_nik="3215045608040002",
    )
    assert login["document_matched"] is False
    assert login["error_code"] == "pemadanan_npwp.wrong_document"

    name = parse_bpjs_name_from_filename(
        "pemadanan npwp_M. CHAIRIL HAMZAH_3213080506990001.jpg"
    )
    assert name == "M. CHAIRIL HAMZAH"


def test_bpjs_ai_accepts_loose_model_json() -> None:
    from systems.validation.bpjs_ai import bpjs_validation_from_extraction
    from systems.validation.keterangan_kesehatan_ai import _parse_health_json

    samples = {
        "tita": """{
- "is_bpjs_kesehatan": true,
- "holder_name": "TITA DWILESTARI",
- "nik": "3215236104010004",
- "card_number": "0003635883775",
- "confidence": 98,
- "notes": "Kartu Indonesia Sehat."
}""",
        "syahrul": """JSON
{
is_bpjs_kesehatan: true,
holder_name: "SYAHRUL, ARRIZAL MA'ARIF",
nik: "3302172305320",
card_number: "0002698191145",
confidence: 98,
notes: "Kartu Indonesia Sehat."
}""",
        "lia": """Is_bpjs_kesehatan: true
Holder_name: "LIA APRIANI"
Nik: "321601570650001"
Card_number: "0000492438971"
Confidence: 99
Notes: "Kartu Indonesia Sehat."
""",
        "dyah": """is_bpjs_kesehatan: true
holder_name: DYAH AYU WULANDARI
nik: 3216076208960001
card_number: 0001771359884
confidence: 99
notes: Ini kartu Indonesia Sehat / BPJS Kesehatan; nama dan NIK terbaca jelas.
""",
    }
    tita = bpjs_validation_from_extraction(
        _parse_health_json(samples["tita"]),
        expected_name="Tita Dwi Lestari",
        expected_nik="3215236104010004",
    )
    assert tita["holder_name"] == "TITA DWILESTARI"
    assert tita["document_matched"] is True

    syahrul = bpjs_validation_from_extraction(
        _parse_health_json(samples["syahrul"]),
        expected_name="Syahrul Arrizal Ma'arif",
        expected_nik="3302172305020002",
    )
    assert syahrul["name_pass"] is True
    assert syahrul["document_matched"] is True

    lia = bpjs_validation_from_extraction(
        _parse_health_json(samples["lia"]),
        expected_name="Lia Apriani",
        expected_nik="3216015706050001",
    )
    assert lia["holder_name"] == "LIA APRIANI"
    assert lia["document_matched"] is True

    dyah = bpjs_validation_from_extraction(
        _parse_health_json(samples["dyah"]),
        expected_name="Dyah ayu wulandari",
        expected_nik="3216076208960001",
    )
    assert dyah["nik"] == "3216076208960001"
    assert dyah["document_matched"] is True


def test_vaksinasi_3_ai_booster_and_name() -> None:
    from systems.validation.vaksinasi_3_ai import vaksinasi_3_validation_from_extraction

    booster = vaksinasi_3_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_3": True,
            "holder_name": "TIARA PUTRI",
            "dose_label": "VAKSIN BOOSTER",
            "nik": "3212017010070005",
        },
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert booster["document_matched"] is True

    primer = vaksinasi_3_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_3": True,
            "holder_name": "YOLANDA INESTASYA",
            "dose_label": "VAKSIN PRIMER 2",
            "notes": "Ini vaksin primer 2, bukan booster.",
        },
        expected_name="Yolanda Inestasya Saragih",
    )
    assert primer["is_dose_3"] is False
    assert primer["error_code"] == "vaksinasi_3.wrong_document"

    printed = vaksinasi_3_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_3": False,
            "holder_name": "TIARA",
            "dose_label": "",
            "notes": "Kartu vaksin booster, bukan dosis kedua.",
        },
        expected_name="Tiara Putri",
    )
    assert printed["is_dose_3"] is True
    assert printed["name_pass"] is True
    assert printed["document_matched"] is True


def test_vaksinasi_2_ai_dose_and_name() -> None:
    from systems.validation.bpjs_ai import parse_bpjs_name_from_filename
    from systems.validation.fuzzy_compare import parse_nik16_from_filename
    from systems.validation.vaksinasi_2_ai import vaksinasi_2_validation_from_extraction

    both = vaksinasi_2_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_2": True,
            "holder_name": "AAP HABATUL FADLI",
            "nik": "3212042904060005",
            "dose_label": "tahap 2",
        },
        expected_name="Aap habatul fadli",
        expected_nik="3212042904060005",
    )
    assert both["document_matched"] is True

    only_first = vaksinasi_2_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_2": False,
            "holder_name": "AAP HABATUL FADLI",
            "dose_label": "tahap 1",
        },
        expected_name="Aap habatul fadli",
    )
    assert only_first["error_code"] == "vaksinasi_2.wrong_document"

    booster = vaksinasi_2_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_2": False,
            "holder_name": "AAP HABATUL FADLI",
            "dose_label": "booster",
        },
        expected_name="Aap habatul fadli",
    )
    assert booster["error_code"] == "vaksinasi_2.wrong_document"

    primer = vaksinasi_2_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_2": False,
            "holder_name": "FAHRU RAMADHAN",
            "dose_label": "vaksin primer 2",
        },
        expected_name="FAHRU RAMADHAN",
    )
    assert primer["is_dose_2"] is True
    assert primer["document_matched"] is True

    denied_but_printed = vaksinasi_2_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_2": False,
            "holder_name": "MOETIARA SARI",
            "dose_label": "",
            "notes": "Kartu vaksin primer 2, bukan dosis kedua.",
        },
        expected_name="Moetiara Sari",
    )
    assert denied_but_printed["is_dose_2"] is True
    assert denied_but_printed["document_matched"] is True

    yolanda = vaksinasi_2_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_2": False,
            "holder_name": "YOLANDA INESTASYA",
            "dose_label": "vaksin primer 2",
        },
        expected_name="Yolanda Inestasya Saragih",
    )
    assert yolanda["name_pass"] is True
    assert yolanda["document_matched"] is True

    dotted = "vaksinasi 2_M. CHAIRIL HAMZAH_3213080506990001.jpg"
    assert parse_nik16_from_filename(dotted) == "3213080506990001"
    assert parse_bpjs_name_from_filename(dotted) == "M. CHAIRIL HAMZAH"


def test_vaksinasi_1_ai_dose_and_name() -> None:
    from systems.validation.vaksinasi_1_ai import vaksinasi_1_validation_from_extraction

    both_doses = vaksinasi_1_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_1": True,
            "holder_name": "AAP HABATUL FADLI",
            "nik": "3212042904060005",
            "dose_label": "tahap 1",
            "confidence": 95,
        },
        expected_name="Aap habatul fadli",
        expected_nik="3212042904060005",
    )
    assert both_doses["document_matched"] is True
    assert both_doses["error_code"] is None

    truncated = vaksinasi_1_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_1": True,
            "holder_name": "TIARA",
            "nik": "",
        },
        expected_name="Tiara Putri",
    )
    assert truncated["name_pass"] is True
    assert truncated["document_matched"] is True

    nik_only = vaksinasi_1_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "dose_label": "tahap 1",
            "holder_name": "",
            "nik": "3212042904060005",
        },
        expected_name="Aap habatul fadli",
        expected_nik="3212042904060005",
    )
    assert nik_only["name_match_method"] == "nik"
    assert nik_only["document_matched"] is True

    dose2 = vaksinasi_1_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_1": False,
            "holder_name": "AAP HABATUL FADLI",
            "dose_label": "tahap 2",
        },
        expected_name="Aap habatul fadli",
    )
    assert dose2["error_code"] == "vaksinasi_1.wrong_document"

    other = vaksinasi_1_validation_from_extraction(
        {
            "is_covid_vaccine": True,
            "is_dose_1": True,
            "holder_name": "BUDI",
            "nik": "1111222233334444",
        },
        expected_name="Aap habatul fadli",
        expected_nik="3212042904060005",
    )
    assert other["error_code"] == "vaksinasi_1.name_mismatch"


def test_iuran_ai_modal_name_and_nik() -> None:
    from systems.validation.iuran_ai import iuran_validation_from_extraction

    modal = iuran_validation_from_extraction(
        {
            "is_info_iuran": True,
            "holder_name": "",
            "name_displayed": False,
            "modal_tanpa_tagihan": True,
            "nik": "",
            "confidence": 90,
        },
        expected_name="Aiga Tara Nasucha",
        expected_nik="3216085009070001",
    )
    assert modal["document_matched"] is True
    assert modal["name_match_method"] == "modal"

    named = iuran_validation_from_extraction(
        {
            "is_info_iuran": True,
            "holder_name": "ANGGUN RAMADHANI",
            "name_displayed": True,
            "modal_tanpa_tagihan": False,
            "nik": "3275045610060008",
            "confidence": 90,
        },
        expected_name="ANGGUN RAMADHANI",
        expected_nik="3275045610060008",
    )
    assert named["document_matched"] is True

    truncated = iuran_validation_from_extraction(
        {
            "is_info_iuran": True,
            "holder_name": "SITI",
            "name_displayed": True,
            "modal_tanpa_tagihan": False,
            "nik": "3212017010070005",
            "confidence": 90,
        },
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert truncated["document_matched"] is True
    assert truncated["name_match_method"] == "nik"

    other = iuran_validation_from_extraction(
        {
            "is_info_iuran": True,
            "holder_name": "Budi Santoso",
            "name_displayed": True,
            "modal_tanpa_tagihan": False,
            "nik": "",
            "confidence": 80,
        },
        expected_name="Anggun Ramadhani",
    )
    assert other["document_matched"] is False
    assert other["error_code"] == "iuran.name_mismatch"

    peserta = iuran_validation_from_extraction(
        {
            "is_info_iuran": False,
            "holder_name": "Anggun Ramadhani",
            "name_displayed": True,
            "modal_tanpa_tagihan": False,
            "notes": "Ini layar Info Peserta.",
            "confidence": 80,
        },
        expected_name="Anggun Ramadhani",
    )
    assert peserta["error_code"] == "iuran.wrong_document"

    riwayat = iuran_validation_from_extraction(
        {
            "is_info_iuran": False,
            "holder_name": "",
            "name_displayed": False,
            "modal_tanpa_tagihan": False,
            "notes": "Info Riwayat Pembayaran. Peserta tidak memiliki riwayat pembayaran iuran.",
            "confidence": 80,
        },
        expected_name="Muhammad Anwar",
        expected_nik="3175040410060006",
    )
    assert riwayat["document_matched"] is True
    assert riwayat["name_match_method"] == "modal"

    riwayat_lunas = iuran_validation_from_extraction(
        {
            "is_info_iuran": True,
            "is_payment_history": True,
            "holder_name": "DEVLINA CHRISTIN SINTOMARITO",
            "name_displayed": True,
            "has_tagihan": True,
            "notes": "Info Riwayat Pembayaran. Pembayaran Iuran. Channel FINNET BTN. Tgl Bayar. Status Lunas.",
        },
        expected_name="Devlina Christin Sintomarito",
    )
    assert riwayat_lunas["document_matched"] is False
    assert riwayat_lunas["error_code"] == "iuran.wrong_document"

    pandawa = iuran_validation_from_extraction(
        {
            "is_info_iuran": False,
            "holder_name": "RIXXX HAXXXX MAXXXX",
            "name_displayed": True,
            "modal_tanpa_tagihan": False,
            "notes": "Chat Pandawa Nama Peserta Tagihan Status Pembayaran Lunas.",
            "confidence": 80,
        },
        expected_name="Risky Haikal Mahtum",
    )
    assert pandawa["document_matched"] is True
    assert pandawa["name_match_method"] == "masked"

    keluarga = iuran_validation_from_extraction(
        {
            "is_info_iuran": True,
            "is_info_peserta": True,
            "has_tagihan": False,
            "holder_name": "MARYATI",
            "name_displayed": True,
            "modal_tanpa_tagihan": False,
            "notes": "Info Peserta. Kepesertaan Terdaftar. Faskes 1.",
            "confidence": 90,
        },
        expected_name="MARYATI",
    )
    assert keluarga["document_matched"] is False
    assert keluarga["error_code"] == "iuran.wrong_document"


def test_bpjs_ai_truncated_name_and_nik() -> None:
    from systems.validation.bpjs_ai import (
        bpjs_validation_from_extraction,
        parse_bpjs_name_from_filename,
    )

    truncated = bpjs_validation_from_extraction(
        {
            "is_bpjs_kesehatan": True,
            "holder_name": "TIARA",
            "nik": "3212017010070005",
            "confidence": 90,
        },
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert truncated["name_pass"] is True
    assert truncated["document_matched"] is True
    assert truncated["error_code"] is None

    nik_only = bpjs_validation_from_extraction(
        {
            "is_bpjs_kesehatan": True,
            "holder_name": "",
            "nik": "3175066010000006",
        },
        expected_name="Shella Octhaviani",
        expected_nik="3175066010000006",
    )
    assert nik_only["name_match_method"] == "nik"
    assert nik_only["document_matched"] is True

    wrong = bpjs_validation_from_extraction(
        {
            "is_bpjs_kesehatan": True,
            "holder_name": "BUDI",
            "nik": "1111222233334444",
        },
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert wrong["document_matched"] is False
    assert wrong["error_code"] == "bpjs.name_mismatch"

    not_kis = bpjs_validation_from_extraction(
        {"is_bpjs_kesehatan": False, "holder_name": "Tiara", "nik": "3212017010070005"},
        expected_name="Tiara Putri",
    )
    assert not_kis["error_code"] == "bpjs.wrong_document"

    assert (
        parse_bpjs_name_from_filename(
            "bpjs kesehatan_ANGGUN RAMADHANI_3275045610060008.jpg"
        )
        == "ANGGUN RAMADHANI"
    )


def test_bpjs_and_jkn_reject_different_card_numbers() -> None:
    from systems.validation.bpjs_ai import bpjs_validation_from_extraction
    from systems.validation.jkn_ai import jkn_validation_from_extraction

    mismatch = {
        "is_bpjs_kesehatan": True,
        "holder_name": "NITA",
        "nik": "3201066604990001",
        "card_number": "0000999699491",
        "info_peserta_card_number": "0002885506964",
        "notes": "KIS dan Info Peserta nomornya beda.",
    }
    bpjs = bpjs_validation_from_extraction(mismatch, expected_name="Anita")
    assert bpjs["document_matched"] is False
    assert bpjs["card_numbers_match"] is False
    assert bpjs["error_code"] == "bpjs.card_mismatch"

    same = bpjs_validation_from_extraction(
        {
            "is_bpjs_kesehatan": True,
            "holder_name": "ANITA",
            "nik": "3201066604990001",
            "card_number": "0002885506964",
            "info_peserta_card_number": "0002885506964",
        },
        expected_name="Anita",
    )
    assert same["document_matched"] is True
    assert same["error_code"] is None

    jkn = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [
                {
                    "name": "ANITA",
                    "nik": "",
                    "card_number": "0002885506964",
                    "status": "AKTIF",
                }
            ],
            "bpjs_card_number": "0000999699491",
            "bpjs_nik": "3201066604990001",
            "notes": "Kepesertaan Terdaftar. Kartu Indonesia Sehat ikut terlihat.",
        },
        expected_name="Anita",
    )
    assert jkn["document_matched"] is False
    assert jkn["error_code"] == "jkn.card_mismatch"

    family = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [
                {"name": "ANITA", "nik": "", "card_number": "0002885506964", "status": "AKTIF"},
                {"name": "BUDI", "nik": "", "card_number": "0001111111111", "status": "AKTIF"},
            ],
        },
        expected_name="Anita",
    )
    assert family["document_matched"] is True
    assert family["error_code"] is None

    kis_payload = bpjs_validation_from_extraction(
        {
            "is_bpjs_kesehatan": True,
            "holder_name": "NITA",
            "card_number": "0000999699491",
            "nik": "3201066604990001",
        },
        expected_name="Nita",
        expected_card_number="0002885506964",
    )
    assert kis_payload["document_matched"] is False
    assert kis_payload["error_code"] == "bpjs.card_mismatch"

    kis_same_payload = bpjs_validation_from_extraction(
        {
            "is_bpjs_kesehatan": True,
            "holder_name": "ANITA",
            "card_number": "0002885506964",
        },
        expected_name="Anita",
        expected_card_number="0002885506964",
    )
    assert kis_same_payload["document_matched"] is True

    jkn_payload = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [
                {"name": "ANITA", "nik": "", "card_number": "0002885506964", "status": "AKTIF"}
            ],
        },
        expected_name="Anita",
        expected_card_number="0000999699491",
    )
    assert jkn_payload["document_matched"] is False
    assert jkn_payload["error_code"] == "jkn.card_mismatch"

    jkn_family_payload = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [
                {"name": "ANITA", "nik": "", "card_number": "0002885506964", "status": "AKTIF"},
                {"name": "BUDI", "nik": "", "card_number": "0001111111111", "status": "AKTIF"},
            ],
        },
        expected_name="Anita",
        expected_card_number="0002885506964",
    )
    assert jkn_family_payload["document_matched"] is True

    from app.pipeline_runner import run_pipeline_bytes

    missing = run_pipeline_bytes(b"xx", document_type="bpjs", expected_name="Anita")
    assert missing.ok is False
    assert missing.error_kind == "missing_expected_card_number"
    assert missing.http_status_hint == 400
    missing_jkn = run_pipeline_bytes(b"xx", document_type="jkn", expected_card_number="123")
    assert missing_jkn.error_kind == "missing_expected_card_number"


def test_ktp_ai_name_nik_and_helper() -> None:
    from systems.validation.ktp_ai import ktp_validation_from_extraction

    truncated = ktp_validation_from_extraction(
        {
            "is_ktp": True,
            "holder_name": "TIARA",
            "nik": "3212017010070005",
            "marital_status": "BELUM KAWIN",
            "confidence": 90,
        },
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert truncated["name_pass"] is True
    assert truncated["document_matched"] is True
    assert truncated["error_code"] is None
    assert truncated["marital_status"]["status"] == "belum_kawin"

    nik_only = ktp_validation_from_extraction(
        {"is_ktp": True, "holder_name": "", "nik": "3175066010000006", "marital_status": "KAWIN"},
        expected_name="Shella Octhaviani",
        expected_nik="3175066010000006",
    )
    assert nik_only["name_match_method"] == "nik"
    assert nik_only["document_matched"] is True

    wrong = ktp_validation_from_extraction(
        {"is_ktp": True, "holder_name": "BUDI", "nik": "1111222233334444"},
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert wrong["error_code"] == "ktp.name_mismatch"

    not_ktp = ktp_validation_from_extraction(
        {"is_ktp": False, "holder_name": "Tiara", "nik": "3212017010070005"},
        expected_name="Tiara Putri",
    )
    assert not_ktp["error_code"] == "ktp.wrong_document"

    helper_ok = ktp_validation_from_extraction(
        {
            "is_ktp": True,
            "holder_name": "Tiara Putri",
            "nik": "3212017010070005",
            "marital_status": "BELUM KAWIN",
        },
        expected_name="Tiara Putri",
        job_position="Helper",
    )
    assert helper_ok["marital_pass"] is True
    assert helper_ok["document_matched"] is True

    helper_kawin = ktp_validation_from_extraction(
        {
            "is_ktp": True,
            "holder_name": "Tiara Putri",
            "marital_status": "KAWIN",
        },
        expected_name="Tiara Putri",
        job_position="helper",
    )
    assert helper_kawin["marital_pass"] is False
    assert helper_kawin["error_code"] == "ktp.helper_not_single"

    helper_blank = ktp_validation_from_extraction(
        {"is_ktp": True, "holder_name": "Tiara Putri", "marital_status": ""},
        expected_name="Tiara Putri",
        job_position="Helper",
    )
    assert helper_blank["error_code"] == "ktp.helper_not_single"


def test_remaining_profiles_use_ornith() -> None:
    from systems.validation.bpjs_kesanggupan_ai import bpjs_kesanggupan_validation_from_extraction
    from systems.validation.bpjs_tk_ai import bpjs_tk_validation_from_extraction
    from systems.validation.jkn_ai import jkn_validation_from_extraction
    from systems.validation.simple_form_ai import form_validation_from_extraction

    ijasah = form_validation_from_extraction(
        {"is_ijasah": True, "holder_name": "SITI AMINAH"},
        profile="ijasah",
        expected_name="Siti Aminah",
    )
    assert ijasah["document_matched"] is True
    assert ijasah["error_code"] is None

    wrong_name = form_validation_from_extraction(
        {"is_ijasah": True, "holder_name": "BUDI"},
        profile="ijasah",
        expected_name="Siti Aminah",
    )
    assert wrong_name["error_code"] == "ijasah.name_mismatch"

    not_diploma = form_validation_from_extraction(
        {"is_ijasah": False, "holder_name": "SITI AMINAH"},
        profile="ijasah",
        expected_name="Siti Aminah",
    )
    assert not_diploma["error_code"] == "ijasah.wrong_document"

    spelled_with_z = form_validation_from_extraction(
        {"is_ijazah": True, "holder_name": "Mathew Hff Sitanggang", "confidence": 99},
        profile="ijasah",
        expected_name="Mathew HF Sitanggang",
    )
    assert spelled_with_z["is_ijasah"] is True
    assert spelled_with_z["document_matched"] is True
    assert spelled_with_z["error_code"] is None

    tk = bpjs_tk_validation_from_extraction(
        {"is_bpjs_tk": True, "holder_name": "", "nik": "3212017010070005"},
        expected_name="Tiara Putri",
        expected_nik="3212017010070005",
    )
    assert tk["name_match_method"] == "nik"
    assert tk["document_matched"] is True

    not_tk = bpjs_tk_validation_from_extraction(
        {"is_bpjs_tk": False, "holder_name": "Tiara"},
        expected_name="Tiara Putri",
    )
    assert not_tk["error_code"] == "bpjs_tk.wrong_document"

    stamped = bpjs_kesanggupan_validation_from_extraction(
        {"is_surat_kesanggupan": True, "holder_name": "EGY SUBAGJA", "has_physical_stamp": True},
        expected_name="Egy Subagja",
    )
    assert stamped["stamp_pass"] is True
    assert stamped["document_matched"] is True

    no_stamp = bpjs_kesanggupan_validation_from_extraction(
        {"is_surat_kesanggupan": True, "holder_name": "EGY SUBAGJA", "has_physical_stamp": False},
        expected_name="Egy Subagja",
    )
    assert no_stamp["error_code"] == "bpjs_kesanggupan.stamp_missing"

    jkn = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [{"name": "BAGUS", "nik": "3275010101010001", "status": "TIDAK AKTIF"}],
        },
        expected_name="Bagus Junda",
    )
    assert jkn["document_matched"] is True
    assert jkn["jkn_status_label"] == "tidak_aktif"
    assert jkn["error_code"] is None

    iuran_bukan_peserta = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "is_info_iuran": True,
            "has_tagihan": True,
            "members": [{"name": "CECEP RUKMANA", "nik": "", "status": ""}],
            "notes": "Info Iuran. Sisa Saldo. Tagihan. Total Tagihan.",
        },
        expected_name="Cecep Rukmana",
    )
    assert iuran_bukan_peserta["document_matched"] is False
    assert iuran_bukan_peserta["error_code"] == "jkn.wrong_document"

    not_jkn = jkn_validation_from_extraction(
        {"is_info_peserta": False, "members": []},
        expected_name="Bagus",
    )
    assert not_jkn["error_code"] == "jkn.wrong_document"

    empty_peserta = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [],
            "notes": "Info Peserta. Jenis Tampilan. Kepesertaan terdaftar. Ubah.",
        },
        expected_name="Anita",
        expected_card_number="0002885506964",
    )
    assert empty_peserta["document_matched"] is False
    assert empty_peserta["error_code"] == "jkn.wrong_document"

    chrome_only = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "members": [{"name": "Kepesertaan terdaftar", "nik": "", "card_number": "", "status": ""}],
        },
        expected_name="Anita",
    )
    assert chrome_only["document_matched"] is False
    assert chrome_only["error_code"] == "jkn.wrong_document"

    kis_card = jkn_validation_from_extraction(
        {
            "is_info_peserta": True,
            "is_kis_card": True,
            "members": [
                {
                    "name": "NADILA FIRYNO SARAS",
                    "nik": "3215034806060002",
                    "card_number": "0002804344711",
                    "status": "",
                }
            ],
            "notes": "Kartu Indonesia Sehat. Nomor Kartu. Faskes CAMPEL.",
        },
        expected_name="Nadila Firyno Saras",
        expected_card_number="0002804344711",
    )
    assert kis_card["document_matched"] is False
    assert kis_card["error_code"] == "jkn.wrong_document"


def test_pipeline_ijasah_skips_paddle_uses_ornith() -> None:
    from app import pipeline_runner as pr
    from systems.validation.simple_form_ai import analyze_ijasah_image as real_ai

    calls = {"ocr_inner": 0}

    def fake_ai(image_bytes, **kwargs):
        from systems.validation.simple_form_ai import form_validation_from_extraction

        parsed = {"is_ijasah": True, "holder_name": "SITI AMINAH"}
        validation = form_validation_from_extraction(
            parsed,
            profile="ijasah",
            expected_name=kwargs.get("expected_name") or "",
        )
        return {
            "error": None,
            "extraction": parsed,
            "validation": validation,
            "llm": {"model": "ornith-1.5-9b", "latency_s": 0.1},
            "raw_content": "{}",
        }

    real_ocr = pr._run_ocr_inner

    def counting_ocr(*args, **kwargs):
        calls["ocr_inner"] += 1
        return real_ocr(*args, **kwargs)

    pr.analyze_ijasah_image = fake_ai
    pr._run_ocr_inner = counting_ocr
    try:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = pr.run_pipeline_bytes(
            png,
            document_type="ijasah",
            expected_name="Siti Aminah",
            ocr_mode="fast",
        )
        assert result.ok is True, result.error
        assert result.document_matched is True
        assert calls["ocr_inner"] == 0
        assert result.payload["ocr"]["mode"] == "ornith_ijasah_ai"
    finally:
        pr.analyze_ijasah_image = real_ai
        pr._run_ocr_inner = real_ocr


def test_mutasi_ai_bank_and_account() -> None:
    from systems.validation.mutasi_ai import mutasi_validation_from_extraction

    ok = mutasi_validation_from_extraction(
        {"is_mutation": True, "bank": "mandiri", "account_number": "1330035980275"},
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert ok["bank_pass"] is True
    assert ok["account_pass"] is True
    assert ok["identity_pass"] is None
    assert ok["document_matched"] is True
    assert ok["error_code"] is None

    wrong_acct = mutasi_validation_from_extraction(
        {"is_mutation": True, "bank": "mandiri", "account_number": "1330035980275"},
        expected_bank="mandiri",
        expected_account="1560027756842",
    )
    assert wrong_acct["account_pass"] is False
    assert wrong_acct["error_code"] == "mutasi.account_mismatch"

    leading_zero = mutasi_validation_from_extraction(
        {"is_mutation": True, "bank": "mandiri", "account_number": "700013443119"},
        expected_bank="mandiri",
        expected_account="0700013443119",
    )
    assert leading_zero["account_pass"] is True
    assert leading_zero["document_matched"] is True

    one_digit = mutasi_validation_from_extraction(
        {"is_mutation": True, "bank": "", "account_number": "1730021695987"},
        expected_bank="mandiri",
        expected_account="1730321695987",
    )
    assert one_digit["bank_pass"] is True
    assert one_digit["document_matched"] is True

    counterparty = mutasi_validation_from_extraction(
        {"is_mutation": True, "bank": "mas", "account_number": "1360034647690"},
        expected_bank="mandiri",
        expected_account="1360034647690",
    )
    assert counterparty["detected_bank"] == "mandiri"
    assert counterparty["document_matched"] is True

    wrong_bank = mutasi_validation_from_extraction(
        {"is_mutation": True, "bank": "mandiri", "account_number": "1004489264"},
        expected_bank="mas",
        expected_account="1200000001",
    )
    assert wrong_bank["error_code"] == "mutasi.wrong_bank"

    detail_page = mutasi_validation_from_extraction(
        {"is_mutation": False, "bank": "mandiri", "account_number": "1330035980275"},
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert detail_page["error_code"] == "mutasi.wrong_document"


def test_pipeline_mutasi_skips_paddle_uses_ornith() -> None:
    from app import pipeline_runner as pr
    from systems.validation.mutasi_ai import analyze_mutasi_image as real_ai

    calls = {"ocr_inner": 0}

    def fake_ai(image_bytes, **kwargs):
        from systems.validation.mutasi_ai import mutasi_validation_from_extraction

        parsed = {
            "is_mutation": True,
            "bank": "mandiri",
            "account_number": "1330035980275",
        }
        validation = mutasi_validation_from_extraction(
            parsed,
            expected_bank=kwargs.get("expected_bank") or "",
            expected_account=kwargs.get("expected_account") or "",
        )
        return {
            "error": None,
            "extraction": parsed,
            "validation": validation,
            "llm": {"model": "ornith-1.5-9b", "latency_s": 0.1},
            "raw_content": "{}",
        }

    real_ocr = pr._run_ocr_inner

    def counting_ocr(*args, **kwargs):
        calls["ocr_inner"] += 1
        return real_ocr(*args, **kwargs)

    pr.analyze_mutasi_image = fake_ai
    pr._run_ocr_inner = counting_ocr
    try:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = pr.run_pipeline_bytes(
            png,
            document_type="mutasi",
            expected_name="Meli Saputri",
            expected_bank="mandiri",
            expected_account="1330035980275",
            filename="mutasi rekening_Meli Saputri_1330035980275.jpg",
            ocr_mode="fast",
        )
        assert result.ok is True, result.error
        assert result.document_matched is True
        assert calls["ocr_inner"] == 0
        assert result.payload["ocr_engine"] == "ornith"
        assert result.payload["ocr"]["mode"] == "ornith_mutasi_ai"
    finally:
        pr.analyze_mutasi_image = real_ai
        pr._run_ocr_inner = real_ocr


def test_kk_ai_name_or_nik() -> None:
    from systems.validation.kk_ai import kk_validation_from_extraction

    members = [
        {"name": "BUDI SANTOSO", "nik": "3216061012920024"},
        {"name": "SRI PUTRI", "nik": "3201175512030002"},
    ]
    both = kk_validation_from_extraction(
        {
            "is_kartu_keluarga": True,
            "family_card_number": "3173030710200004",
            "members": members,
        },
        expected_name="Sri Putri",
        expected_nik="3201175512030002",
    )
    assert both["identity"]["kk_name_pass"] is True
    assert both["identity"]["kk_nik_pass"] is True
    assert both["document_matched"] is True
    assert both["error_code"] is None
    assert both["holder_name"] == "SRI PUTRI"

    name_only = kk_validation_from_extraction(
        {"is_kartu_keluarga": True, "members": members},
        expected_name="Sri Putri",
        expected_nik="3276025406070011",
    )
    assert name_only["identity"]["kk_name_pass"] is True
    assert name_only["identity"]["kk_nik_pass"] is False
    assert name_only["document_matched"] is True

    nik_only = kk_validation_from_extraction(
        {"is_kartu_keluarga": True, "members": members},
        expected_name="Orang Lain Saja",
        expected_nik="3201175512030002",
    )
    assert nik_only["name_match_method"] == "nik"
    assert nik_only["document_matched"] is True

    one_digit = kk_validation_from_extraction(
        {
            "is_kartu_keluarga": True,
            "family_card_number": "3215052110120098",
            "members": [{"name": "MAWAR", "nik": "3215054409060004"}],
        },
        expected_name="Bukan Nama",
        expected_nik="3215054409060014",
    )
    assert one_digit["identity"]["kk_nik_pass"] is True
    assert one_digit["document_matched"] is True

    family_no_only = kk_validation_from_extraction(
        {
            "is_kartu_keluarga": True,
            "family_card_number": "3173030710200004",
            "members": members,
        },
        expected_name="Orang Lain Saja",
        expected_nik="3173030710200004",
    )
    assert family_no_only["error_code"] == "kk.identity_mismatch"

    neither = kk_validation_from_extraction(
        {"is_kartu_keluarga": True, "members": members},
        expected_name="Orang Lain Saja",
        expected_nik="3276025406070011",
    )
    assert neither["error_code"] == "kk.identity_mismatch"

    not_kk = kk_validation_from_extraction(
        {"is_kartu_keluarga": False, "members": members},
        expected_name="Sri Putri",
    )
    assert not_kk["error_code"] == "kk.wrong_document"

    truncated = kk_validation_from_extraction(
        {"is_kartu_keluarga": True, "members": [{"name": "TIARA", "nik": ""}]},
        expected_name="Tiara Putri",
    )
    assert truncated["document_matched"] is True


def test_pipeline_kk_skips_paddle_uses_ornith() -> None:
    from app import pipeline_runner as pr
    from systems.validation.kk_ai import analyze_kk_image as real_ai

    calls = {"ocr_inner": 0}

    def fake_ai(image_bytes, **kwargs):
        from systems.validation.kk_ai import kk_validation_from_extraction

        parsed = {
            "is_kartu_keluarga": True,
            "family_card_number": "3173030710200004",
            "members": [{"name": "SRI PUTRI", "nik": "3201175512030002"}],
        }
        validation = kk_validation_from_extraction(
            parsed,
            expected_name=kwargs.get("expected_name") or "",
            expected_nik=kwargs.get("expected_nik") or "",
        )
        return {
            "error": None,
            "extraction": parsed,
            "validation": validation,
            "llm": {"model": "ornith-1.5-9b", "latency_s": 0.1},
            "raw_content": "{}",
        }

    real_ocr = pr._run_ocr_inner

    def counting_ocr(*args, **kwargs):
        calls["ocr_inner"] += 1
        return real_ocr(*args, **kwargs)

    pr.analyze_kk_image = fake_ai
    pr._run_ocr_inner = counting_ocr
    try:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = pr.run_pipeline_bytes(
            png,
            document_type="kk",
            expected_name="Sri Putri",
            filename="kk_SRI PUTRI_3201175512030002.jpg",
            ocr_mode="fast",
        )
        assert result.ok is True, result.error
        assert result.document_matched is True
        assert calls["ocr_inner"] == 0
        assert result.payload["ocr_engine"] == "ornith"
        assert result.payload["ocr"]["mode"] == "ornith_kk_ai"
        assert result.payload["validation"]["holder_name"] == "SRI PUTRI"
    finally:
        pr.analyze_kk_image = real_ai
        pr._run_ocr_inner = real_ocr


def test_pipeline_ktp_skips_paddle_uses_ornith() -> None:
    from app import pipeline_runner as pr
    from systems.validation.ktp_ai import analyze_ktp_image as real_ai

    calls = {"ocr_inner": 0}

    def fake_ai(image_bytes, **kwargs):
        from systems.validation.ktp_ai import ktp_validation_from_extraction

        parsed = {
            "is_ktp": True,
            "holder_name": "ANGGUN RAMADHANI",
            "nik": "3275045610060008",
            "marital_status": "BELUM KAWIN",
        }
        validation = ktp_validation_from_extraction(
            parsed,
            expected_name=kwargs.get("expected_name") or "",
            expected_nik=kwargs.get("expected_nik") or "",
            job_position=kwargs.get("job_position") or "",
        )
        return {
            "error": None,
            "extraction": parsed,
            "validation": validation,
            "llm": {"model": "ornith-1.5-9b", "base_url": "http://172.21.13.26:8080/v1", "latency_s": 0.1},
            "raw_content": "{}",
        }

    real_ocr = pr._run_ocr_inner

    def counting_ocr(*args, **kwargs):
        calls["ocr_inner"] += 1
        return real_ocr(*args, **kwargs)

    pr.analyze_ktp_image = fake_ai
    pr._run_ocr_inner = counting_ocr
    try:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = pr.run_pipeline_bytes(
            png,
            document_type="ktp",
            expected_name="ANGGUN RAMADHANI",
            filename="ktp_ANGGUN RAMADHANI_3275045610060008.jpg",
            ocr_mode="fast",
        )
        assert result.ok is True
        assert result.document_matched is True
        assert calls["ocr_inner"] == 0
        assert result.payload["ocr_engine"] == "ornith"
        assert result.payload["ocr"]["mode"] == "ornith_ktp_ai"
        assert result.payload["validation"]["holder_name"] == "ANGGUN RAMADHANI"
    finally:
        pr.analyze_ktp_image = real_ai
        pr._run_ocr_inner = real_ocr


def test_portrait_ai_gates_profile_and_blue_only() -> None:
    from systems.validation.foto_profile_ai import portrait_validation_from_extraction

    ok = portrait_validation_from_extraction(
        {"is_profile_photo": True, "is_blue_background": True, "confidence": 90}
    )
    assert ok["document_matched"] is True
    assert ok["error_code"] is None

    red = portrait_validation_from_extraction(
        {"is_profile_photo": True, "is_blue_background": False}
    )
    assert red["document_matched"] is False
    assert red["error_code"] == "foto_profile.not_blue"

    wall = portrait_validation_from_extraction(
        {
            "is_profile_photo": True,
            "is_blue_background": True,
            "is_wall": True,
            "notes": "Biru yang terlihat adalah warna dinding.",
            "confidence": 90,
        }
    )
    assert wall["document_matched"] is False
    assert wall["error_code"] == "foto_profile.not_blue"

    doc = portrait_validation_from_extraction(
        {"is_profile_photo": False, "is_blue_background": True}
    )
    assert doc["error_code"] == "foto_profile.not_profile_photo"

    both = portrait_validation_from_extraction(
        {"is_profile_photo": False, "is_blue_background": False}
    )
    codes = [c["code"] for c in both["error_cases"]]
    assert codes == ["foto_profile.not_profile_photo", "foto_profile.not_blue"]


if __name__ == "__main__":
    test_clean_labeled_belum_kawin()
    test_ocr_extra_r_on_perkawinan()
    test_glued_perkawinarc_belum()
    test_compact_and_typos()
    test_kawin_not_from_label_word()
    test_cerai()
    test_ocr_elwin_fragment_is_belum_kawin()
    test_glued_status_perkawinan_belum()
    test_aisyah_passthrough_garbage_not_false_kawin()
    test_ktp_compact_name_in_ocr_recovers_wrong_candidate()
    test_ktp_compact_name_does_not_pass_shared_token_only()
    test_npwp_last_name_and_nik_suffix_recover_ocr_typo()
    test_npwp_wrong_name_still_passes_when_nik_matches()
    test_npwp_right_name_fails_when_nik_mismatch()
    test_npwp_nik_match_passes_even_if_ocr_looks_like_ktp()
    test_npwp_new_card_keywords_and_name()
    test_npwp_name_plus_tax_id_recovers_weak_keywords()
    test_kk_name_or_nik()
    test_kk_nik_ocr_confusable_and_one_digit()
    test_rekening_requires_name_and_visible_account()
    test_skck_expiry_must_not_be_past()
    test_skck_ai_expiry_and_name()
    test_bpjs_kesanggupan_requires_meterai()
    test_bpjs_kesanggupan_red_meterai_on_dataset_images()
    test_keterangan_kesehatan_sehat_and_expiry()
    test_health_letter_issuer_rejects_bidan()
    test_keterangan_kesehatan_reads_ocr_dates()
    test_user_error_messages_per_case()
    test_jkn_extracts_aktif_status_without_blocking()
    test_jkn_wrapped_and_pandawa_and_nik_fallback()
    test_bpjs_kis_nik_truncated_name_and_keyword_recovery()
    test_truncated_jpeg_is_repaired_for_ocr()
    test_rekening_ai_accepts_partial_and_masked_name()
    test_bpjs_ai_accepts_loose_model_json()
    test_vaksinasi_3_ai_booster_and_name()
    test_vaksinasi_2_ai_dose_and_name()
    test_vaksinasi_1_ai_dose_and_name()
    test_bpjs_ai_truncated_name_and_nik()
    test_bpjs_and_jkn_reject_different_card_numbers()
    test_npwp_ai_format_lama_flag()
    test_iuran_ai_modal_name_and_nik()
    test_pemadanan_npwp_ai_identity()
    test_remaining_profiles_use_ornith()
    test_pipeline_ijasah_skips_paddle_uses_ornith()
    test_mutasi_ai_bank_and_account()
    test_pipeline_mutasi_skips_paddle_uses_ornith()
    test_kk_ai_name_or_nik()
    test_pipeline_kk_skips_paddle_uses_ornith()
    test_ktp_ai_name_nik_and_helper()
    test_pipeline_ktp_skips_paddle_uses_ornith()
    test_portrait_ai_gates_profile_and_blue_only()
    print("ok")
