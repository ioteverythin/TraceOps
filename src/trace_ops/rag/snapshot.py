"""Retriever snapshot testing — detect drift in vector-store results.

Take a snapshot of your retriever's current behaviour for a set of queries,
commit it to version control, and use it as a regression guard in CI.

Usage::

    from trace_ops.rag.snapshot import RetrieverSnapshot

    # Once — record the baseline (commit retriever_snapshot.yaml)
    snap = RetrieverSnapshot.record(
        retriever=chroma_collection,
        queries=["refund policy", "shipping times", "returns"],
        save_to="snapshots/retriever_v1.yaml",
        top_k=5,
    )

    # In every CI run
    snap = RetrieverSnapshot.load("snapshots/retriever_v1.yaml")
    result = snap.check(chroma_collection)
    assert result.min_overlap(0.8), result.summary()
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class QueryResult:
    """Snapshot comparison result for one query."""

    query: str
    overlap_ratio: float
    score_delta: float
    added_chunks: list[str] = field(default_factory=list)
    removed_chunks: list[str] = field(default_factory=list)
    has_drift: bool = False

    @property
    def score(self) -> float:
        """Alias for overlap_ratio (for uniform check result API)."""
        return self.overlap_ratio

    @property
    def passed(self) -> bool:
        """True if overlap_ratio >= 0.8 (drift threshold)."""
        return not self.has_drift


@dataclass
class SnapshotCheckResult:
    """Aggregated result of comparing a retriever against its snapshot."""

    queries: list[QueryResult]

    @property
    def has_drift(self) -> bool:
        return any(q.has_drift for q in self.queries)

    def min_overlap(self, threshold: float) -> bool:
        """Return True if every query has overlap >= threshold."""
        return all(q.overlap_ratio >= threshold for q in self.queries)

    def max_score_delta(self, threshold: float) -> bool:
        """Return True if every query's mean score delta <= threshold."""
        return all(abs(q.score_delta) <= threshold for q in self.queries)

    def summary(self) -> str:
        lines = ["Retriever Snapshot Check:"]
        for q in self.queries:
            status = "⚠ DRIFT" if q.has_drift else "✅ OK"
            lines.append(f"  {status} '{q.query[:50]}': overlap={q.overlap_ratio:.0%}")
            if q.added_chunks:
                lines.append(f"    + Added: {', '.join(q.added_chunks[:3])}")
            if q.removed_chunks:
                lines.append(f"    - Removed: {', '.join(q.removed_chunks[:3])}")
        return "\n".join(lines)


class RetrieverSnapshot:
    """Snapshot of retriever behaviour for regression testing."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def record(
        cls,
        retriever: Any,
        queries: list[str],
        save_to: str | Path,
        retriever_fn: Callable[..., list[dict[str, Any]]] | None = None,
        top_k: int = 5,
    ) -> "RetrieverSnapshot":
        """Record current retriever behaviour as a snapshot YAML file.

        Args:
            retriever: A retriever object (ChromaDB Collection, LangChain
                retriever, etc.) or any object with a compatible interface.
            queries: List of query strings to record.
            save_to: Path where the snapshot YAML will be written.
            retriever_fn: Optional callable ``(query, top_k) -> [{"id":..., "score":..., "text":...}]``
                for custom retrievers that don't follow a known interface.
            top_k: Number of results to request per query.

        Returns:
            A :class:`RetrieverSnapshot` loaded from the newly-written file.
        """
        data: dict[str, Any] = {
            "queries": {},
            "top_k": top_k,
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        for query in queries:
            results = cls._fetch(retriever, query, top_k=top_k, retriever_fn=retriever_fn)
            data["queries"][query] = results

        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return cls(data)

    @classmethod
    def load(cls, path: str | Path) -> "RetrieverSnapshot":
        """Load a snapshot from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    # ── Check / update ─────────────────────────────────────────────────────

    def check(
        self,
        retriever: Any,
        retriever_fn: Callable[..., list[dict[str, Any]]] | None = None,
        threshold: float = 0.8,
    ) -> SnapshotCheckResult:
        """Compare the current retriever output against this snapshot.

        Args:
            retriever: Current retriever to test.
            retriever_fn: Optional custom fetch callable (same signature as in
                :meth:`record`).

        Returns:
            A :class:`SnapshotCheckResult` summarising drift per query.
        """
        top_k = self.data.get("top_k", 5)
        query_results: list[QueryResult] = []

        for query, old_results in self.data.get("queries", {}).items():
            new_results = self._fetch(retriever, query, top_k=top_k, retriever_fn=retriever_fn)

            old_ids = {r.get("id", "") for r in old_results}
            new_ids = {r.get("id", "") for r in new_results}
            union = old_ids | new_ids
            overlap = len(old_ids & new_ids) / max(len(union), 1)
            added = list(new_ids - old_ids)
            removed = list(old_ids - new_ids)

            old_scores = {r.get("id", ""): r.get("score", 0.0) for r in old_results}
            new_scores = {r.get("id", ""): r.get("score", 0.0) for r in new_results}
            common = old_ids & new_ids
            delta = (
                sum(new_scores[c] - old_scores[c] for c in common) / len(common)
                if common else 0.0
            )

            has_drift = overlap < threshold or len(added) > 2 or len(removed) > 2

            query_results.append(QueryResult(
                query=query,
                overlap_ratio=overlap,
                score_delta=delta,
                added_chunks=added,
                removed_chunks=removed,
                has_drift=has_drift,
            ))

        return SnapshotCheckResult(queries=query_results)

    def check_trace(
        self,
        trace: Any,
        threshold: float = 0.8,
    ) -> list[Any]:
        """Check retrieval events in a trace against this snapshot.

        Compares each recorded retrieval event's retrieved chunk IDs
        against the corresponding saved query in the snapshot (matched by
        query text).  Returns a list of objects with ``.query``,
        ``.score`` (overlap ratio), and ``.passed`` attributes.

        Args:
            trace: A :class:`trace_ops._types.Trace` with retrieval events.
            threshold: Minimum overlap ratio to pass.

        Returns:
            List of result objects (one per matched retrieval event).
        """
        from types import SimpleNamespace
        from trace_ops._types import EventType

        saved_queries: dict[str, list[dict[str, Any]]] = self.data.get("queries", {})
        results = []
        for event in trace.retrieval_events:
            query = event.query or ""
            # Find closest matching saved query
            saved = saved_queries.get(query)
            if saved is None:
                # Skip unmatched queries (snapshot may not have them)
                continue

            old_ids = {r.get("id", "") for r in saved}
            new_ids = {
                (c.get("id", "") if isinstance(c, dict) else getattr(c, "id", ""))
                for c in (event.chunks or [])
            }
            union = old_ids | new_ids
            overlap = len(old_ids & new_ids) / max(len(union), 1)
            results.append(SimpleNamespace(
                query=query,
                score=overlap,
                passed=overlap >= threshold,
            ))

        return results

    @property
    def results(self) -> list[dict[str, Any]]:
        """List of recorded query result dicts from the snapshot data."""
        queries = self.data.get("queries", {})
        return [
            {"query": q, "chunks": chunks}
            for q, chunks in queries.items()
        ]

    def update(
        self,
        retriever: Any,
        save_to: str | Path | None = None,
        retriever_fn: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> "RetrieverSnapshot":
        """Re-record all queries and overwrite the snapshot.

        Args:
            retriever: Updated retriever to record.
            save_to: Path to save the updated snapshot (required).
            retriever_fn: Optional custom fetch callable.

        Returns:
            The updated :class:`RetrieverSnapshot`.
        """
        if save_to is None:
            raise ValueError("save_to is required for update()")
        queries = list(self.data.get("queries", {}).keys())
        top_k = self.data.get("top_k", 5)
        return self.record(
            retriever, queries, save_to=save_to, retriever_fn=retriever_fn, top_k=top_k
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    @classmethod
    def _fetch(
        cls,
        retriever: Any,
        query: str,
        top_k: int,
        retriever_fn: Callable[..., list[dict[str, Any]]] | None,
    ) -> list[dict[str, Any]]:
        """Dispatch to the appropriate fetch method."""
        if retriever_fn is not None:
            return retriever_fn(query, top_k=top_k)
        if hasattr(retriever, "query"):
            # ChromaDB Collection
            results = retriever.query(query_texts=[query], n_results=top_k)
            return cls._normalize_chromadb(results)
        if hasattr(retriever, "invoke"):
            # LangChain retriever
            docs = retriever.invoke(query)
            return cls._normalize_langchain(docs)
        if hasattr(retriever, "similarity_search_with_score"):
            pairs = retriever.similarity_search_with_score(query, k=top_k)
            return [
                {"id": str(i), "score": float(s), "text": d.page_content[:200]}
                for i, (d, s) in enumerate(pairs)
            ]
        raise ValueError(
            f"Unknown retriever type: {type(retriever).__name__}. "
            "Pass a retriever_fn callable for custom retrievers."
        )

    @staticmethod
    def _normalize_chromadb(results: dict[str, Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if not results or not results.get("ids"):
            return normalized
        for i, id_list in enumerate(results["ids"]):
            for j, chunk_id in enumerate(id_list):
                score = 0.0
                if results.get("distances") and i < len(results["distances"]):
                    dist = results["distances"][i][j]
                    score = round(1.0 - float(dist), 6)
                text = ""
                if results.get("documents") and i < len(results["documents"]):
                    text = results["documents"][i][j] or ""
                normalized.append({"id": str(chunk_id), "score": score, "text": text[:200]})
        return normalized

    @staticmethod
    def _normalize_langchain(docs: list[Any]) -> list[dict[str, Any]]:
        results = []
        for i, doc in enumerate(docs):
            if isinstance(doc, (list, tuple)) and len(doc) == 2:
                doc, score = doc
            else:
                score = 0.0
            results.append({
                "id": getattr(doc, "id", None) or str(i),
                "score": float(score),
                "text": (getattr(doc, "page_content", "") or "")[:200],
            })
        return results
