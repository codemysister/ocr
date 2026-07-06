"""Bulk index CV chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from opensearchpy.helpers import bulk

from systems.cv.config import get_settings
from systems.cv.index.opensearch_client import ensure_index, get_opensearch_client


def index_chunks(chunks: list[dict[str, Any]], *, vectors: list[list[float]]) -> dict[str, Any]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks dan vectors harus sama panjangnya.")
    if not chunks:
        return {"indexed": 0, "doc_id": None}

    settings = get_settings()
    index = ensure_index(vector_dim=len(vectors[0]) if vectors else settings.vector_dim)
    client = get_opensearch_client()
    now = datetime.now(timezone.utc).isoformat()

    actions = []
    for chunk, vec in zip(chunks, vectors):
        doc = {**chunk, "content_vector": vec, "ingested_at": chunk.get("ingested_at") or now}
        actions.append(
            {
                "_op_type": "index",
                "_index": index,
                "_id": chunk["chunk_id"],
                "_source": doc,
            }
        )

    ok, errors = bulk(client, actions, refresh=True, raise_on_error=False)
    err_list = errors if isinstance(errors, list) else []
    return {
        "indexed": ok,
        "errors": len(err_list),
        "doc_id": chunks[0].get("doc_id"),
        "index": index,
    }
