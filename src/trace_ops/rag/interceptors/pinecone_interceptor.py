"""Pinecone Index.query() interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_pinecone(recorder: Recorder) -> None:
    """Patch Pinecone ``Index.query()`` to record retrieval events."""
    try:
        from pinecone import Index
    except ImportError:
        return

    original = Index.query

    @functools.wraps(original)
    def patched(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self_inner, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        query_text = kwargs.get("filter", {}).get("_query_text", "")
        top_k = kwargs.get("top_k", 0)

        chunks = []
        matches = getattr(result, "matches", []) or []
        for match in matches:
            cid = getattr(match, "id", "") or ""
            score = float(getattr(match, "score", 0.0) or 0.0)
            meta = dict(getattr(match, "metadata", {}) or {})
            text = meta.pop("text", "") or meta.pop("content", "") or ""
            chunks.append({"id": str(cid), "text": str(text), "score": score, "metadata": meta})

        index_name = getattr(self_inner, "_config", None)
        index_name = getattr(index_name, "index_name", "") if index_name else ""

        recorder.record_retrieval(
            query=query_text,
            retrieved_chunks=chunks,
            vector_store="pinecone",
            collection=str(index_name),
            top_k=top_k,
            duration_ms=duration_ms,
        )
        return result

    Index.query = patched  # type: ignore[method-assign]
    recorder._rag_patches.append(("pinecone.Index.query", original, Index, "query"))
