# CV Search — Spesifikasi

Subsistem **CV**: ingest curriculum vitae lalu **match terstruktur** pada 3 dimensi — **nama**, **pendidikan**, **pengalaman**. Tanpa chatbot / hybrid search bebas.

## Dimensi match

| Dimensi | Cara cek | Lolos jika |
|---------|----------|------------|
| **nama** | Ekstrak dari section data pribadi + fuzzy vs `expected_name` | skor ≥ **65%** (hanya jika nama diisi) |
| **pendidikan** | Keyword `pendidikan`, `education` di section pendidikan | partial_ratio ≥ **70%** |
| **pengalaman** | Keyword `pengalaman`, `experience`, `pengalaman kerja` di section pengalaman | partial_ratio ≥ **70%** |

`matched` = semua gate yang aktif lolos. `overall_percent` = rata-rata skor dimensi yang dicek.

## Arsitektur

```
Upload CV → parse (PyMuPDF / Docling / PP-OCRv6)
         → chunk per section + section_kind
         → embed BGE-M3 → OpenSearch
         → match_cv (nama / pendidikan / pengalaman)
```

## API

| Method | Path | Deskripsi |
|--------|------|-----------|
| `POST` | `/systems/cv/api/v1/ingest` | Upload CV (+ form `expected_name`, opsional `education_query`, `experience_query`) |
| `POST` | `/systems/cv/api/v1/match` | Match CV terindeks via `doc_id` |
| `GET` | `/systems/cv/api/v1/documents` | Daftar CV |
| `DELETE` | `/systems/cv/api/v1/documents/{doc_id}` | Hapus CV |

Pipeline: `POST /api/v1/pipeline` + `document_type=cv` + `expected_name`.

## Contoh match

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/pipeline" \
  -F "file=@dataset/cv/cv_Usep_Maulidin.pdf" \
  -F "document_type=cv" \
  -F "expected_name=Usep Maulidin"
```

Respons inti: `cv_match.matched`, `cv_match.overall_percent`, `cv_match.dimensions`.
