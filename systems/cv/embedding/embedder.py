"""BGE-M3 embedder untuk pencarian CV."""

from __future__ import annotations

import logging
import threading
from typing import Any

from systems.cv.config import get_settings

logger = logging.getLogger(__name__)

_model: Any = None
_lock = threading.Lock()


def _get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        logger.info("Loading CV embed model: %s", settings.embed_model)
        _model = SentenceTransformer(settings.embed_model, trust_remote_code=True)
        return _model


def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    model = _get_model()
    bs = batch_size or settings.embed_batch_size
    vecs = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embedder_status() -> dict[str, Any]:
    settings = get_settings()
    return {"loaded": _model is not None, "model": settings.embed_model}
