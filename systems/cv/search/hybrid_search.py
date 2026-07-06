"""Hybrid BM25 + vector search untuk CV (tanpa LLM)."""

from __future__ import annotations

from typing import Any

from systems.cv.config import get_settings
from systems.cv.embedding.embedder import embed_query
from systems.cv.index.opensearch_client import ensure_index, get_opensearch_client


def _rrf_fusion(ranked_lists: list[list[dict[str, Any]]], *, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            items[cid] = item
    return [{**items[cid], "score_rrf": sc} for cid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def apply_score_percent(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ubah skor RRF menjadi persentase relatif; hit teratas = 100%."""
    if not hits:
        return hits
    max_rrf = max(float(h.get("score_rrf") or h.get("score") or 0.0) for h in hits) or 1e-9
    out: list[dict[str, Any]] = []
    for h in hits:
        rrf = float(h.get("score_rrf") if "score_rrf" in h else h.get("score") or 0.0)
        pct = round(100.0 * rrf / max_rrf, 1)
        out.append({**h, "score_rrf": rrf, "score": pct, "score_percent": pct})
    return out


def search_cv(query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    client = get_opensearch_client()
    index = ensure_index()
    n = top_k or settings.search_top_k
    qvec = embed_query(query)
    cv_filter = {"term": {"doc_type": "cv"}}

    bm25_body = {
        "size": n,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^3", "doc_title^2", "section_path"],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": [cv_filter],
            }
        },
        "_source": True,
    }
    knn_body = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"knn": {"content_vector": {"vector": qvec, "k": n}}}],
                "filter": [cv_filter],
            }
        },
        "_source": True,
    }

    def _parse(resp: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for h in resp.get("hits", {}).get("hits") or []:
            src = h.get("_source") or {}
            out.append(
                {
                    "chunk_id": src.get("chunk_id") or h.get("_id"),
                    "doc_id": src.get("doc_id"),
                    "doc_title": src.get("doc_title"),
                    "source_file": src.get("source_file"),
                    "section_path": src.get("section_path") or [],
                    "page_numbers": src.get("page_numbers") or [],
                    "content": src.get("content") or "",
                    "score": float(h.get("_score") or 0.0),
                }
            )
        return out

    fused = _rrf_fusion(
        [_parse(client.search(index=index, body=bm25_body)), _parse(client.search(index=index, body=knn_body))]
    )
    return apply_score_percent(fused[:n])
