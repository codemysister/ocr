# Checklist validasi per jenis dokumen

Ringkasan **apa saja yang dicek** saat `POST /api/v1/pipeline` (atau subsistem terkait).  
Detail teknis & curl: [API.md](./API.md) · kode error UI: `GET /systems/validation/api/v1/error-catalog`

**Legenda**

| Simbol | Arti |
|--------|------|
| **Wajib** | Field request harus diisi |
| **Dicek** | Gate validasi; gagal → `document_matched: false` + `error_code` |
| **Opsional** | Hanya dicek jika field referensi diisi |
| **Info** | Diekstrak untuk tampilan/QA, tidak memblokir lolos |

Ambang umum OCR: keyword profil rata-rata **≥ 70%** (fuzzy); nama/identitas **≥ 65** (skor gabungan).

---

## `ktp` — Kartu Tanda Penduduk

**Metode:** OCR (PP-OCRv6 default)

**Input referensi**

| Field | Wajib? |
|-------|--------|
| `expected_name` | Opsional (disarankan) |
| `job_position` | Opsional — jika `helper`, aktifkan gate perkawinan |

**Yang dicek**

- Jenis dokumen (keyword KTP: `nik`, `nama`, `provinsi`, `kabupaten`, `agama`, `kewarganegaraan`, `status perkawinan`)
- **Nama** pemegang vs `expected_name` (ekstraksi dari baris `Nama` di OCR)
- **Status perkawinan** = `BELUM KAWIN` jika `job_position=helper` (Helper)
- **Info:** NIK 16 digit di OCR (bonus skor jenis dokumen, bukan gate identitas terpisah)

**Error umum:** `ktp.wrong_document` · `ktp.name_mismatch` · `ktp.helper_not_single`

---

## `npwp` — NPWP

**Metode:** **Ornith AI** (bukan Paddle)

**Input:** `expected_name` opsional; **NIK 16 digit** bisa dari filename (`…_NIK16.ext`)

**Yang dicek**

- Jenis dokumen: gambar adalah kartu NPWP
- **Nama** wajib pajak vs `expected_name` (jika diisi)
- **NIK/NPWP 16 digit** vs referensi filename, hanya jika nomor di kartu berformat NIK

**Flag:** `format_lama` — `true` jika nomor NPWP **bukan** 16 digit (format lama / belum pemadanan). `false` jika nomor sama dengan format NIK. Flag ini tidak menggagalkan dokumen.

**Error umum:** `npwp.wrong_document` · `npwp.name_mismatch` · `npwp.nik_mismatch`  
**Error server (503):** `pipeline.npwp_ai_unavailable`

---

## `pemadanan_npwp` — Bukti pemadanan NPWP

**Metode:** **Ornith AI** (bukan Paddle)

**Input:** `expected_name` opsional; nama dan **NIK 16 digit** bisa dari filename (`pemadanan npwp_{Nama}_{NIK16}.ext`)

**Yang dicek**

- Jenis dokumen: kartu NPWP baru (nomor 16 digit), screenshot Ikhtisar Profil DJP, atau Surat Keterangan Aktivasi NIK. Pada surat aktivasi, NIK 16 digit adalah nomor NPWP. Kartu format lama 15 digit tidak lolos.
- **Nama** wajib pajak (boleh terpotong) atau **nomor NPWP 16 digit** yang sama dengan NIK filename

**Error umum:** `pemadanan_npwp.wrong_document` · `pemadanan_npwp.nik_mismatch` · `pemadanan_npwp.name_mismatch`  
**Error server (503):** `pipeline.pemadanan_npwp_ai_unavailable`

---

## `kk` — Kartu Keluarga

**Metode:** OCR

**Input:** `expected_name` opsional; NIK anggota bisa dari filename

**Yang dicek**

- Jenis dokumen (keyword: `kartu keluarga`, `kepala keluarga`, `nik`)
- **Nama atau NIK anggota** vs referensi (scan baris tabel anggota, bukan hanya kepala keluarga)

**Error umum:** `kk.wrong_document` · `kk.identity_mismatch`

---

## `rekening` — Bukti rekening tabungan

**Metode:** **Ornith AI** (bukan Paddle)

Nama pemilik boleh tidak lengkap atau tersensor. `Deva` dan `Dxxa` lolos untuk referensi `Deva Apriana`. Kata lain yang tidak ada di nama referensi tetap gagal.

**Input**

| Field | Wajib? |
|-------|--------|
| `expected_bank` | **Wajib** (`mandiri` atau `mas`) |
| `expected_name` | Salah satu dengan `expected_account` |
| `expected_account` | Salah satu dengan `expected_name` (10–13 digit) |

**Yang dicek**

- Jenis dokumen: ada **`tabungan`**, **bukan** e-statement (`e-statement` = dokumen salah jenis)
- **Bank** (Mandiri vs MAS) vs `expected_bank`
- **Nama pemilik** **atau** **nomor rekening** terlihat & cocok (minimal satu lolos)

**Error umum:** `rekening.wrong_document` · `rekening.wrong_type_statement` · `rekening.wrong_bank` · `rekening.name_mismatch` · `rekening.account_mismatch`  
**Error server (503):** `pipeline.rekening_ai_unavailable`

---

## `mutasi` — Mutasi / e-statement

**Metode:** OCR

**Input**

| Field | Wajib? |
|-------|--------|
| `expected_bank` | **Wajib** |
| `expected_account` | **Wajib** |
| `expected_name` | Tidak dipakai |

**Yang dicek**

- Jenis dokumen: **`tabungan`** dan **`e-statement`** keduanya terdeteksi
- **Bank** vs `expected_bank`
- **Nomor rekening** vs `expected_account`

**Error umum:** `mutasi.wrong_document` · `mutasi.wrong_bank` · `mutasi.account_mismatch`

---

## `skck` — SKCK

**Metode:** OCR

**Input:** `expected_name` opsional

**Yang dicek**

- Jenis dokumen (keyword: `skck`, `kepolisian`)
- **Nama** vs `expected_name` (jika diisi)
- **Tanggal berlaku** (“Sampai dengan …”) — tidak kadaluarsa & terbaca

**Error umum:** `skck.wrong_document` · `skck.name_mismatch` · `skck.expiry_missing` · `skck.expired` · `skck.not_yet_valid`

---

## `keterangan_kesehatan` — Surat keterangan sehat

**Metode:** **Ornith AI** (vision, bukan Paddle)

**Input:** `expected_name` opsional; `expected_nik` dari filename opsional

**Yang dicek**

- **Jenis dokumen** = surat kesehatan / MCU (bukan surat buta warna, vaksin, dll.)
- **Nama** pasien vs `expected_name` (jika diisi)
- Dinyatakan **SEHAT**
- **Tanggal dibuat / pemeriksaan** terbaca (`issued_date`, `issued_date_display`)
- **Kadaluarsa** hanya jika surat menulis masa berlaku / “sampai dengan …”

**Error umum:** `keterangan_kesehatan.wrong_document` · `keterangan_kesehatan.name_mismatch` · `keterangan_kesehatan.not_healthy` · `keterangan_kesehatan.date_missing` · `keterangan_kesehatan.expired`  
**Error server (503):** `pipeline.health_ai_unavailable` (Ornith tidak merespons)

---

## `bpjs` — Kartu Indonesia Sehat (KIS)

**Metode:** **Ornith AI** (bukan Paddle)

Nama di kartu boleh terpotong (`Tiara` lolos untuk `Tiara Putri`). NIK 16 digit dari filename yang cocok dengan kartu tetap lolos meski nama tidak lengkap.

**Input:** `expected_name` disarankan wajib

**Yang dicek**

- Jenis dokumen: kartu BPJS Kesehatan / Kartu Indonesia Sehat (bukan TK, info iuran, atau vaksin)
- **Nama** di kartu vs `expected_name` (boleh terpotong)
- **NIK** 16 digit di kartu vs NIK di nama file; cocok tetap lolos meski nama tidak lengkap

**Error umum:** `bpjs.wrong_document` · `bpjs.name_mismatch`  
**Error server (503):** `pipeline.bpjs_ai_unavailable`

---

## `bpjs_tk` — BPJS Ketenagakerjaan

**Metode:** OCR

**Input:** `expected_name` disarankan wajib

**Yang dicek**

- Jenis dokumen: grup keyword `kartu peserta` **dan** (`ketenagakerjaan` / `bpjs ketenagakerjaan`)
- Exclusion (mis. syarat KIS, info iuran)
- **Nama** peserta vs `expected_name`

**Error umum:** `bpjs_tk.wrong_document` · `bpjs_tk.name_mismatch`

---

## `bpjs_kesanggupan` — Surat kesanggupan BPJS

**Metode:** OCR

**Input:** `expected_name` disarankan wajib

**Yang dicek**

- Jenis dokumen (frasa kesanggupan menanggung biaya BPJS + konten pernyataan)
- **Nama** pada surat vs `expected_name`
- **Materai merah** fisik terdeteksi di gambar

**Error umum:** `bpjs_kesanggupan.wrong_document` · `bpjs_kesanggupan.name_mismatch` · `bpjs_kesanggupan.stamp_missing`

---

## `jkn` — Info Peserta JKN (screenshot app)

**Metode:** OCR

**Input:** `expected_name` disarankan wajib

**Yang dicek**

- Jenis dokumen (minimal satu: `info peserta`, `faskes`, `nama peserta`, `pandawa`, …)
- **Nama** peserta vs `expected_name`
- **Info:** status AKTIF/TIDAK AKTIF (dieksrak, **tidak** memblokir lolos)

**Error umum:** `jkn.wrong_document` · `jkn.name_mismatch`

---

## `iuran` — Info Iuran JKN

**Metode:** OCR

**Input:** `expected_name` disarankan (modal tanpa nama pribadi bisa skip identitas)

**Yang dicek**

- Jenis dokumen: `info iuran` + salah satu keyword konten (tagihan, saldo, dll.)
- **Nama** vs `expected_name` (bila relevan)

**Error umum:** `iuran.wrong_document` · `iuran.name_mismatch`

---

## `vaksinasi_1` — Vaksinasi COVID-19 dosis pertama

**Metode:** **Ornith AI** (bukan Paddle)

Dosis pertama tetap lolos jika riwayat juga memuat tahap 2 atau booster. Nama boleh terpotong. NIK 16 digit dari filename yang cocok dengan kartu tetap lolos meski nama tidak lengkap.

**Input:** `expected_name` disarankan

**Error umum:** `vaksinasi_1.wrong_document` · `vaksinasi_1.name_mismatch`  
**Error server (503):** `pipeline.vaksinasi_1_ai_unavailable`

---

## `vaksinasi_2` — Vaksinasi COVID-19 dosis kedua

**Metode:** **Ornith AI** (bukan Paddle)

Dosis kedua tetap lolos jika riwayat juga memuat tahap 1. Booster saja tidak lolos. Nama boleh terpotong. NIK 16 digit dari filename yang cocok dengan kartu tetap lolos meski nama tidak lengkap.

**Input:** `expected_name` disarankan

**Error umum:** `vaksinasi_2.wrong_document` · `vaksinasi_2.name_mismatch`  
**Error server (503):** `pipeline.vaksinasi_2_ai_unavailable`

---

## `vaksinasi_3` — Vaksinasi COVID-19 booster

**Metode:** **Ornith AI** (bukan Paddle)

Booster tetap lolos jika riwayat juga memuat tahap 1 atau tahap 2. `Vaksin primer 2` bukan booster. Nama boleh terpotong. NIK 16 digit dari filename yang cocok dengan kartu tetap lolos meski nama tidak lengkap.

**Input:** `expected_name` disarankan

**Error umum:** `vaksinasi_3.wrong_document` · `vaksinasi_3.name_mismatch`  
**Error server (503):** `pipeline.vaksinasi_3_ai_unavailable`

---

## `ijasah`

**Metode:** OCR

**Input:** `expected_name` opsional

**Yang dicek**

- Jenis dokumen (keyword: `ijazah`, `pendidikan`, `lulus`, `nomor ijazah`)
- **Nama** vs `expected_name` (jika diisi)

**Error umum:** `ijasah.wrong_document` · `ijasah.name_mismatch`

---

## `transkrip`

**Metode:** OCR

**Input:** `expected_name` opsional

**Yang dicek**

- Jenis dokumen (keyword: `transkrip`, `nilai`, `semester`, `ipk`)
- **Nama** vs `expected_name` (jika diisi)

**Error umum:** `transkrip.wrong_document` · `transkrip.name_mismatch`

---

## `formulir_okb`

**Metode:** OCR

**Input:** `expected_name` opsional

**Yang dicek**

- Jenis dokumen (keyword: `formulir okb`, `okb`, `pemeriksaan`)
- **Nama** vs `expected_name` (jika diisi)

**Error umum:** `formulir_okb.wrong_document` · `formulir_okb.name_mismatch`

---

## `formulir_lamaran`

**Metode:** OCR

**Input:** `expected_name` opsional

**Yang dicek**

- Jenis dokumen (keyword: `formulir lamaran`, `lamaran pekerjaan`, `data pribadi`)
- **Nama** vs `expected_name` (jika diisi)

**Error umum:** `formulir_lamaran.wrong_document` · `formulir_lamaran.name_mismatch`

---

## `surat_lamaran`

**Metode:** OCR

**Input:** `expected_name` opsional

**Yang dicek**

- Jenis dokumen (keyword: `surat lamaran`, `yang bertanda tangan`, `pekerjaan`)
- **Nama** vs `expected_name` (jika diisi)

**Error umum:** `surat_lamaran.wrong_document` · `surat_lamaran.name_mismatch`

---

## `foto_profile` — Pas foto biru

**Metode:** **Ornith AI** (bukan deteksi wajah OpenCV)

**Input:** `expected_name` tidak dipakai

**Yang dicek**

- **Foto profil / pas foto** (wajah orang sebagai subjek, bukan KTP/dokumen/screenshot)
- **Latar belakang biru** solid seperti pas foto resmi

Jumlah wajah, ukuran wajah, dan posisi tidak memblokir.

**Error umum:** `foto_profile.not_profile_photo` · `foto_profile.not_blue`  
**Error server (503):** `pipeline.portrait_ai_unavailable`

---

## `cv` — Curriculum vitae

**Metode:** **Ingest + match** (OpenSearch, bukan OCR keyword)

**Input:** `expected_name` disarankan; query opsional `cv_education_query`, `cv_experience_query`

**Yang dicek**

| Dimensi | Kapan | Lolos (ringkas) |
|---------|--------|------------------|
| **nama** | `expected_name` diisi | Fuzzy nama dari teks CV ≥ **65%** |
| **pendidikan** | selalu | Section pendidikan + query ≥ **70%** atau marker sekolah/universitas |
| **pengalaman** | selalu | Section pengalaman + query ≥ **50%** atau marker kerja / fresh graduate |

**Error umum:** `cv.match_failed`

---

## Ringkasan cepat (tabel)

| Profil | Jenis dokumen | Nama | NIK/identitas lain | Bank | No. rekening | Lainnya |
|--------|---------------|------|--------------------|------|--------------|---------|
| `ktp` | ✅ | Opsional | Info NIK | — | — | Helper → belum kawin |
| `npwp` | ✅ (AI) | Opsional | NIK 16 jika bukan format lama | — | — | `format_lama` |
| `pemadanan_npwp` | ✅ (AI) | Opsional, boleh terpotong | NIK/NPWP 16 | — | — | Bukan format lama 15 digit |
| `kk` | ✅ | Opsional | Nama **atau** NIK anggota | — | — | |
| `rekening` | ✅ | OR akun | — | ✅ | OR nama | Bukan e-statement |
| `mutasi` | ✅ | — | — | ✅ | ✅ | Harus e-statement |
| `skck` | ✅ | Opsional | — | — | — | Masa berlaku |
| `keterangan_kesehatan` | ✅ (AI) | Opsional | — | — | — | SEHAT + tanggal |
| `bpjs` | ✅ (AI) | Opsional, boleh terpotong | NIK filename bisa lolos | — | — | Bukan TK/iuran |
| `bpjs_tk` | ✅ | Opsional | — | — | — | |
| `bpjs_kesanggupan` | ✅ | Opsional | — | — | — | Materai merah |
| `jkn` | ✅ | Opsional | — | — | — | Status JKN = info |
| `iuran` | ✅ | Opsional | — | — | — | |
| `vaksinasi_1` | ✅ (AI) | Opsional, boleh terpotong | NIK filename bisa lolos | — | — | Tahap 1 tetap lolos bila tahap 2 ikut tercantum |
| `vaksinasi_2` | ✅ (AI) | Opsional, boleh terpotong | NIK filename bisa lolos | — | — | Tahap 2 tetap lolos bila tahap 1 ikut tercantum |
| `vaksinasi_3` | ✅ (AI) | Opsional, boleh terpotong | NIK filename bisa lolos | — | — | Booster; primer 2 bukan booster |
| `ijasah` | ✅ | Opsional | — | — | — | |
| `transkrip` | ✅ | Opsional | — | — | — | |
| `formulir_okb` | ✅ | Opsional | — | — | — | |
| `formulir_lamaran` | ✅ | Opsional | — | — | — | |
| `surat_lamaran` | ✅ | Opsional | — | — | — | |
| `foto_profile` | — | — | — | — | — | AI: pas foto + biru |
| `cv` | — | Dimensi nama | — | — | — | Pendidikan + pengalaman |

---

*Diperbarui mengikuti kode validasi di `systems/validation/` — untuk integrasi HTTP lihat [API.md](./API.md).*
