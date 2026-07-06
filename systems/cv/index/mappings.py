"""OpenSearch index mapping untuk chunk CV."""

from __future__ import annotations

from typing import Any

from systems.cv.config import VECTOR_DIM


def index_body(*, vector_dim: int = VECTOR_DIM) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "analysis": {
                "analyzer": {
                    "cv_text": {"type": "standard"},
                }
            },
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "doc_title": {
                    "type": "text",
                    "analyzer": "cv_text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "source_file": {"type": "keyword"},
                "section_path": {"type": "keyword"},
                "page_numbers": {"type": "integer"},
                "content": {"type": "text", "analyzer": "cv_text"},
                "content_vector": {
                    "type": "knn_vector",
                    "dimension": vector_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                "doc_type": {"type": "keyword"},
                "section_kind": {"type": "keyword"},
                "parse_mode": {"type": "keyword"},
                "ingested_at": {"type": "date"},
                "version": {"type": "integer"},
            }
        },
    }
