"""LangChain VectorStoreRetriever interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_langchain_retriever(recorder: Recorder) -> None:
    """Patch LangChain ``VectorStoreRetriever._get_relevant_documents``."""
    try:
        from langchain_core.vectorstores import VectorStoreRetriever
    except ImportError:
        return

    original = VectorStoreRetriever._get_relevant_documents

    @functools.wraps(original)
    def patched(self_inner: Any, query: str, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        docs = original(self_inner, query, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        chunks = []
        for i, doc in enumerate(docs):
            score = 0.0
            # LangChain docs may carry a score in metadata
            if isinstance(doc, (list, tuple)) and len(doc) == 2:
                doc, score = doc
            meta = getattr(doc, "metadata", {}) or {}
            cid = meta.get("id") or getattr(doc, "id", None) or str(i)
            chunks.append({
                "id": str(cid),
                "text": getattr(doc, "page_content", "") or "",
                "score": float(score),
                "metadata": meta,
            })

        # Guess collection name from vectorstore if possible
        vs = getattr(self_inner, "vectorstore", None)
        collection = getattr(vs, "_collection_name", "") or getattr(vs, "collection_name", "") or ""

        recorder.record_retrieval(
            query=query,
            retrieved_chunks=chunks,
            vector_store="langchain",
            collection=str(collection),
            top_k=len(chunks),
            duration_ms=duration_ms,
        )
        return docs

    VectorStoreRetriever._get_relevant_documents = patched  # type: ignore[method-assign]
    recorder._rag_patches.append((
        "langchain_core.vectorstores.VectorStoreRetriever._get_relevant_documents",
        original,
        VectorStoreRetriever,
        "_get_relevant_documents",
    ))
