"""Qdrant QdrantClient.search() / query_points() interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_qdrant(recorder: "Recorder") -> None:
    """Patch ``QdrantClient.search()`` to record retrieval events."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return

    original_search = QdrantClient.search

    @functools.wraps(original_search)
    def patched_search(self_inner: Any, collection_name: str, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        results = original_search(self_inner, collection_name, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        chunks = []
        for point in (results or []):
            cid = str(getattr(point, "id", ""))
            score = float(getattr(point, "score", 0.0) or 0.0)
            payload = dict(getattr(point, "payload", {}) or {})
            text = payload.pop("text", "") or payload.pop("content", "") or ""
            chunks.append({"id": cid, "text": str(text), "score": score, "metadata": payload})

        recorder.record_retrieval(
            query="",  # Qdrant search is vector-based; no text query available here
            retrieved_chunks=chunks,
            vector_store="qdrant",
            collection=collection_name,
            top_k=kwargs.get("limit", len(chunks)),
            duration_ms=duration_ms,
        )
        return results

    QdrantClient.search = patched_search  # type: ignore[method-assign]
    recorder._rag_patches.append(("qdrant_client.QdrantClient.search", original_search, QdrantClient, "search"))
