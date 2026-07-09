"""Klasifikasi chunk CV: nama | pendidikan | pengalaman | other."""

from __future__ import annotations

_PENDIDIKAN_MARKERS = (
    "pendidikan",
    "education",
    "riwayat pendidikan",
    "sekolah",
    "universitas",
)
_PENGALAMAN_MARKERS = (
    "pengalaman kerja",
    "pengalaman",
    "experience",
    "work experience",
    "riwayat pekerjaan",
    "pekerjaan",
)
_NAMA_MARKERS = (
    "daftar riwayat hidup",
    "data pribadi",
    "personal data",
    "profil",
    "identitas",
    "tentang saya",
    "about me",
    "ringkasan",
)


def classify_section_kind(section_path: list[str]) -> str:
    joined = " ".join(section_path or []).casefold().strip()
    if not joined:
        return "nama"
    for m in _PENDIDIKAN_MARKERS:
        if m in joined:
            return "pendidikan"
    for m in _PENGALAMAN_MARKERS:
        if m in joined:
            return "pengalaman"
    for m in _NAMA_MARKERS:
        if m in joined:
            return "nama"
    return "other"
