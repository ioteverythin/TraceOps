"""LlamaIndex QueryEngine / Retriever interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_llamaindex(recorder: "Recorder") -> None:
    """Patch LlamaIndex ``BaseRetriever.retrieve()`` to capture retrieval events."""
    try:
        from llama_index.core.base.base_retriever import BaseRetriever
    except ImportError:
        try:
            from llama_index.core import BaseRetriever  # type: ignore[no-redef]
        except ImportError:
            return

    original = BaseRetriever.retrieve

    @functools.wraps(original)
    def patched(self_inner: Any, query: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        nodes = original(self_inner, query, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        query_str = str(getattr(query, "query_str", query) or "")

        chunks = []
        for i, node_with_score in enumerate(nodes):
            # NodeWithScore has .node and .score
            node = getattr(node_with_score, "node", node_with_score)
            score = float(getattr(node_with_score, "score", 0.0) or 0.0)
            text = getattr(node, "text", "") or getattr(node, "get_content", lambda: "")() or ""
            meta = getattr(node, "metadata", {}) or {}
            cid = getattr(node, "node_id", None) or meta.get("id") or str(i)
            chunks.append({"id": str(cid), "text": str(text), "score": score, "metadata": meta})

        recorder.record_retrieval(
            query=query_str,
            retrieved_chunks=chunks,
            vector_store="llamaindex",
            top_k=len(chunks),
            duration_ms=duration_ms,
        )
        return nodes

    BaseRetriever.retrieve = patched  # type: ignore[method-assign]
    recorder._rag_patches.append((
        "llama_index.core.BaseRetriever.retrieve",
        original,
        BaseRetriever,
        "retrieve",
    ))
