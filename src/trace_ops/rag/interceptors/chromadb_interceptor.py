"""ChromaDB Collection.query() interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_chromadb(recorder: Recorder) -> None:
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
        documents = result.get("documents") if result else None
        if result and documents:
            distances = result.get("distances")
            ids_list = result.get("ids")
            metadatas = result.get("metadatas")
            for i, doc_list in enumerate(documents):
                for j, doc in enumerate(doc_list):
                    score = 0.0
                    if distances and i < len(distances) and j < len(distances[i]):
                        score = round(1.0 - float(distances[i][j]), 6)
                    chunk_id = ""
                    if ids_list and i < len(ids_list) and j < len(ids_list[i]):
                        chunk_id = str(ids_list[i][j])
                    meta: dict[str, Any] = {}
                    if metadatas and i < len(metadatas) and j < len(metadatas[i]):
                        meta = dict(metadatas[i][j]) if metadatas[i][j] else {}
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

    chromadb.Collection.query = patched_query  # type: ignore[method-assign,assignment]
    recorder._rag_patches.append(("chromadb.Collection.query", original_query, chromadb.Collection, "query"))
