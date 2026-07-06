"""Chunking berbasis struktur heading/section (CV)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from systems.cv.chunking.section_kind import classify_section_kind
from systems.cv.config import get_settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Judul section CV umum (Indonesia / Inggris) — OCR atau PDF native sering ALL CAPS tanpa #.
_CV_SECTION_TITLES: frozenset[str] = frozenset(
    {
        "daftar riwayat hidup",
        "data pribadi",
        "profil",
        "ringkasan",
        "pendidikan",
        "riwayat pendidikan",
        "kemampuan",
        "keahlian",
        "skills",
        "pengalaman kerja",
        "riwayat pekerjaan",
        "pengalaman",
        "pelatihan",
        "sertifikasi",
        "organisasi",
        "referensi",
        "bahasa",
        "personal data",
        "education",
        "experience",
        "work experience",
    }
)
DOC_TYPE = "cv"


@dataclass
class ParsedSection:
    title: str
    level: int
    content: str
    page_numbers: list[int] = field(default_factory=list)


def _split_long_text(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paras = re.split(r"\n\s*\n", text)
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        candidate = f"{buf}\n\n{p}".strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                parts.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                for i in range(0, len(p), max_chars):
                    parts.append(p[i : i + max_chars].strip())
                buf = ""
    if buf:
        parts.append(buf)
    return parts


def sections_from_markdown(markdown: str) -> list[ParsedSection]:
    md = (markdown or "").strip()
    if not md:
        return []

    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [ParsedSection(title="", level=0, content=md)]

    sections: list[ParsedSection] = []
    prefix = md[: matches[0].start()].strip()
    if prefix:
        sections.append(ParsedSection(title="", level=0, content=prefix))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        sections.append(ParsedSection(title=title, level=level, content=body))
    return sections


def _normalize_section_title(line: str) -> str:
    return " ".join((line or "").strip().split()).casefold()


def _is_cv_section_header(line: str) -> bool:
    raw = (line or "").strip()
    if not raw or len(raw) > 64:
        return False
    if raw.endswith(":"):
        return False
    letters = [c for c in raw if c.isalpha()]
    if len(letters) < 3:
        return False
    norm = _normalize_section_title(raw)
    if norm in _CV_SECTION_TITLES:
        return True
    return False


_PAGE_MARKER_RE = re.compile(r"^##\s*Halaman\s+\d+\s*$", re.IGNORECASE)
_MIN_CHUNK_CHARS = 24


def _strip_ocr_page_markers(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if _PAGE_MARKER_RE.match(line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def sections_from_cv_plaintext(text: str) -> list[ParsedSection]:
    """Pecah teks OCR/PDF native menurut judul section CV (whitelist judul)."""
    src = _strip_ocr_page_markers(text)
    if not src:
        return []

    lines = src.splitlines()
    sections: list[ParsedSection] = []
    current_title = ""
    current_level = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current_title, current_level
        body = "\n".join(buf).strip()
        if body or current_title:
            sections.append(
                ParsedSection(title=current_title, level=current_level, content=body)
            )
        buf = []

    for line in lines:
        stripped = line.strip()
        if _is_cv_section_header(stripped):
            flush()
            current_title = " ".join(stripped.split())
            current_level = 1
            continue
        buf.append(line)

    flush()

    if len(sections) > 1 or (sections and sections[0].title):
        return sections
    return [ParsedSection(title="", level=0, content=src)]


def sections_from_cv_text(text: str) -> list[ParsedSection]:
    """Markdown # jika ada; jika tidak, deteksi judul section CV pada teks polos."""
    md_sections = sections_from_markdown(text)
    has_structure = len(md_sections) > 1 or (
        len(md_sections) == 1 and bool(md_sections[0].title)
    )
    only_page_heading = (
        len(md_sections) == 1
        and md_sections[0].title.lower().startswith("halaman ")
    )
    if has_structure and not only_page_heading:
        return md_sections
    plain = sections_from_cv_plaintext(text)
    if len(plain) > 1 or (plain and plain[0].title):
        return plain
    return md_sections if md_sections else plain


def sections_from_docling_document(document: Any) -> list[ParsedSection]:
    try:
        md = document.export_to_markdown()
    except Exception:
        md = str(document)
    return sections_from_markdown(md)


def build_chunks(
    sections: list[ParsedSection],
    *,
    doc_id: str,
    doc_title: str,
    source_file: str,
    parse_mode: str,
    version: int = 1,
) -> list[dict[str, Any]]:
    settings = get_settings()
    max_chars = settings.max_chunk_chars
    chunks: list[dict[str, Any]] = []
    path_stack: list[str] = []

    for sec in sections:
        while path_stack and len(path_stack) >= sec.level:
            path_stack.pop()
        if sec.title:
            if sec.level <= 0:
                path_stack = [sec.title] if sec.title else []
            else:
                while len(path_stack) < sec.level - 1:
                    path_stack.append("")
                if len(path_stack) == sec.level - 1:
                    path_stack.append(sec.title)
                else:
                    path_stack[sec.level - 1] = sec.title

        section_path = [p for p in path_stack if p]
        body_parts = _split_long_text(sec.content, max_chars)
        if not body_parts and sec.title:
            body_parts = [sec.title]

        for part in body_parts:
            stripped = part.strip()
            if section_path and stripped and len(stripped) < _MIN_CHUNK_CHARS:
                stripped = f"{section_path[-1]}\n{stripped}".strip()
            if not stripped or len(stripped) < _MIN_CHUNK_CHARS:
                continue
            if _PAGE_MARKER_RE.match(stripped):
                continue
            chunk_key = f"{doc_id}|{'/'.join(section_path)}|{stripped[:80]}"
            chunk_id = hashlib.sha256(chunk_key.encode()).hexdigest()[:32]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "source_file": source_file,
                    "section_path": section_path,
                    "section_kind": classify_section_kind(section_path),
                    "page_numbers": list(sec.page_numbers),
                    "content": stripped,
                    "doc_type": DOC_TYPE,
                    "parse_mode": parse_mode,
                    "version": version,
                }
            )
    return chunks


def make_doc_id(source_file: str, file_bytes: bytes) -> str:
    h = hashlib.sha256(file_bytes).hexdigest()[:16]
    name = source_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return f"{name}:{h}"
