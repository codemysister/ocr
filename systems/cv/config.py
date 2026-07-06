"""Konfigurasi subsistem CV dari environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

VECTOR_DIM: Final[int] = 1024


def _env_str(name: str, default: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    return raw or default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CvSettings:
    opensearch_url: str
    opensearch_index: str
    embed_model: str
    max_chunk_chars: int
    scan_min_chars_per_page: int
    search_top_k: int
    embed_batch_size: int
    vector_dim: int

    @classmethod
    def from_env(cls) -> CvSettings:
        return cls(
            opensearch_url=_env_str("OPENSEARCH_URL", "http://localhost:9200"),
            opensearch_index=_env_str("CV_OPENSEARCH_INDEX", "cv_chunks"),
            embed_model=_env_str("CV_EMBED_MODEL", "BAAI/bge-m3"),
            max_chunk_chars=_env_int("CV_MAX_CHUNK_CHARS", 1500),
            scan_min_chars_per_page=_env_int("CV_SCAN_MIN_CHARS_PER_PAGE", 80),
            search_top_k=_env_int("CV_SEARCH_TOP_K", 10),
            embed_batch_size=_env_int("CV_EMBED_BATCH_SIZE", 16),
            vector_dim=VECTOR_DIM,
        )


_settings: CvSettings | None = None


def get_settings() -> CvSettings:
    global _settings
    if _settings is None:
        _settings = CvSettings.from_env()
    return _settings
