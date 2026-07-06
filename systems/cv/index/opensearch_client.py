"""OpenSearch client untuk CV chunks."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from systems.cv.config import get_settings
from systems.cv.index.mappings import index_body

logger = logging.getLogger(__name__)

_client: OpenSearch | None = None


class OpenSearchUnavailableError(RuntimeError):
    """OpenSearch tidak bisa dihubungi (biasanya belum dijalankan)."""

    def __init__(self, *, url: str, cause: Exception | None = None) -> None:
        self.url = url
        self.cause = cause
        msg = (
            f"OpenSearch tidak dapat dijangkau di {url}. "
            "Jalankan: docker compose -f docker-compose.cv.yml up -d"
        )
        super().__init__(msg)


def opensearch_unavailable_detail(*, exc: Exception | None = None) -> dict[str, Any]:
    settings = get_settings()
    detail: dict[str, Any] = {
        "code": "OPENSEARCH_UNAVAILABLE",
        "message": "OpenSearch tidak dapat dijangkau.",
        "opensearch_url": settings.opensearch_url,
        "start": "docker compose -f docker-compose.cv.yml up -d",
        "health": f"{settings.opensearch_url.rstrip('/')}/_cluster/health",
    }
    if exc is not None:
        detail["error"] = str(exc)
    return detail


def is_opensearch_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, OpenSearchUnavailableError | OpenSearchConnectionError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    msg = str(exc).lower()
    return "connection refused" in msg or (
        "max retries exceeded" in msg and "9200" in msg
    )


def get_opensearch_client() -> OpenSearch:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenSearch(
            hosts=[settings.opensearch_url],
            use_ssl=settings.opensearch_url.startswith("https"),
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )
    return _client


def ensure_index(*, vector_dim: int | None = None) -> str:
    settings = get_settings()
    client = get_opensearch_client()
    index = settings.opensearch_index
    dim = vector_dim or settings.vector_dim
    try:
        if not client.indices.exists(index=index):
            client.indices.create(index=index, body=index_body(vector_dim=dim))
            logger.info("Created OpenSearch index %s", index)
    except OpenSearchConnectionError as e:
        raise OpenSearchUnavailableError(url=settings.opensearch_url, cause=e) from e
    except Exception as e:
        if is_opensearch_connection_error(e):
            raise OpenSearchUnavailableError(url=settings.opensearch_url, cause=e) from e
        raise
    return index


def cluster_health() -> dict[str, Any]:
    settings = get_settings()
    try:
        return get_opensearch_client().cluster.health()
    except OpenSearchConnectionError as e:
        raise OpenSearchUnavailableError(url=settings.opensearch_url, cause=e) from e


def delete_by_doc_id(doc_id: str) -> int:
    settings = get_settings()
    client = get_opensearch_client()
    if not client.indices.exists(index=settings.opensearch_index):
        return 0
    resp = client.delete_by_query(
        index=settings.opensearch_index,
        body={"query": {"term": {"doc_id": doc_id}}},
        refresh=True,
    )
    return int(resp.get("deleted") or 0)


def list_documents(*, size: int = 200) -> list[dict[str, Any]]:
    settings = get_settings()
    client = get_opensearch_client()
    if not client.indices.exists(index=settings.opensearch_index):
        return []
    resp = client.search(
        index=settings.opensearch_index,
        body={
            "size": 0,
            "aggs": {
                "docs": {
                    "terms": {"field": "doc_id", "size": size},
                    "aggs": {
                        "title": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["doc_title", "source_file", "ingested_at"],
                            }
                        },
                        "chunk_count": {"value_count": {"field": "chunk_id"}},
                    },
                }
            },
        },
    )
    buckets = resp.get("aggregations", {}).get("docs", {}).get("buckets") or []
    out: list[dict[str, Any]] = []
    for b in buckets:
        hit = (b.get("title", {}).get("hits", {}).get("hits") or [{}])[0]
        src = hit.get("_source") or {}
        out.append(
            {
                "doc_id": b.get("key"),
                "doc_title": src.get("doc_title"),
                "source_file": src.get("source_file"),
                "ingested_at": src.get("ingested_at"),
                "chunk_count": int((b.get("chunk_count") or {}).get("value") or 0),
            }
        )
    return out


def fetch_chunks_by_doc_id(doc_id: str, *, size: int = 200) -> list[dict[str, Any]]:
    settings = get_settings()
    client = get_opensearch_client()
    if not client.indices.exists(index=settings.opensearch_index):
        return []
    resp = client.search(
        index=settings.opensearch_index,
        body={
            "size": size,
            "query": {"term": {"doc_id": doc_id}},
            "_source": True,
        },
    )
    out: list[dict[str, Any]] = []
    for h in resp.get("hits", {}).get("hits") or []:
        src = dict(h.get("_source") or {})
        src.setdefault("chunk_id", h.get("_id"))
        out.append(src)
    return out
