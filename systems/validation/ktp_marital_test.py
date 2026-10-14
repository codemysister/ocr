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
                "synthetic_ocr": (
                    "SURAT KETERANGAN KESEHATAN\nDokter: dr. A\n"
                    "Nama: ANGGUN RAMADHANI\n"
                    "Berdasarkan hasil pemeriksaan, dinyatakan SEHAT.\n"
                    "Bekasi, 04 Juni 2026\nDokter pemeriksa\n"
                ),
                "health_status_pass": True,
                "health_expiry_pass": True,
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

    glued = rekening_validation_from_extraction(
        {**base, "holder_name": "SYAHRUL ARRIZAL MAARIF"},
        expected_name="Syahrul Arrizal Ma'arif",
        expected_bank="mas",
        expected_account="1005992474",
    )
    assert glued["name_pass"] is True

    statement = rekening_validation_from_extraction(
        {**base, "is_savings_account": False, "is_statement": True, "holder_name": "Deva Apriana"},
        expected_name="Deva Apriana",
        expected_bank="mandiri",
        expected_account="1330035980275",
    )
    assert statement["error_code"] == "rekening.wrong_type_statement"


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
            "is_pemadanan_npwp": False,
            "holder_name": "BUNGA SABRIYA",
            "npwp_number": "3172045809060002",
            "notes": "Kartu NPWP format lama 15 digit, bukan hasil pemadanan.",
        },
        expected_name="Bunga Sabriya",
        expected_nik="3172045809060002",
    )
    assert new_card["document_matched"] is True

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
    test_bpjs_kesanggupan_requires_meterai()
    test_bpjs_kesanggupan_red_meterai_on_dataset_images()
    test_keterangan_kesehatan_sehat_and_expiry()
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
    test_npwp_ai_format_lama_flag()
    test_pemadanan_npwp_ai_identity()
    test_portrait_ai_gates_profile_and_blue_only()
    print("ok")
