"""ChromaDB Collection.query() interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_chromadb(recorder: "Recorder") -> None:
    """Monkey-patch ``chromadb.Collection.query()`` to record retrieval events.

    Silently skips if chromadb is not installed.
    """
    try:
        import chromadb
    except ImportError:
        return

    original_query = chromadb.Collection.query

    @functools.wraps(original_query)
    def patched_query(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_query(self_inner, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        query_texts: list[str] = kwargs.get("query_texts") or (
            list(args[0]) if args else []
        )
        if isinstance(query_texts, str):
            query_texts = [query_texts]

        chunks: list[dict[str, Any]] = []
        if result and result.get("documents"):
            for i, doc_list in enumerate(result["documents"]):
                for j, doc in enumerate(doc_list):
                    score = 0.0
                    if result.get("distances") and i < len(result["distances"]):
                        if j < len(result["distances"][i]):
                            score = round(1.0 - float(result["distances"][i][j]), 6)
                    chunk_id = ""
                    if result.get("ids") and i < len(result["ids"]):
                        if j < len(result["ids"][i]):
                            chunk_id = str(result["ids"][i][j])
                    meta: dict[str, Any] = {}
                    if result.get("metadatas") and i < len(result["metadatas"]):
                        if j < len(result["metadatas"][i]):
                            meta = result["metadatas"][i][j] or {}
                    chunks.append({"id": chunk_id, "text": doc or "", "score": score, "metadata": meta})

        recorder.record_retrieval(
            query=query_texts[0] if query_texts else "",
            retrieved_chunks=chunks,
            vector_store="chromadb",
            collection=getattr(self_inner, "name", ""),
            top_k=kwargs.get("n_results", 10),
            duration_ms=duration_ms,
        )
        return result

    chromadb.Collection.query = patched_query  # type: ignore[method-assign]
    recorder._rag_patches.append(("chromadb.Collection.query", original_query, chromadb.Collection, "query"))
