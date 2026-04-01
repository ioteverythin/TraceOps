"""RAG-aware recording extensions for traceops.

Dataclasses for retrieval/embedding events, plus helpers used by
the main Recorder to write them into the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single retrieved chunk from a vector store."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "text": self.text, "score": self.score}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            score=float(d.get("score", 0.0)),
            metadata=d.get("metadata", {}),
        )


@dataclass
class RetrievalEvent:
    """Recorded retrieval event — one vector-store query."""

    query: str
    chunks: list[Chunk]
    vector_store: str = "unknown"
    collection: str = ""
    top_k: int = 0
    embedding_model: str = ""
    duration_ms: float = 0.0
    total_chunks_searched: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "vector_store": self.vector_store,
        }
        if self.collection:
            d["collection"] = self.collection
        if self.top_k:
            d["top_k"] = self.top_k
        if self.embedding_model:
            d["embedding_model"] = self.embedding_model
        if self.duration_ms:
            d["duration_ms"] = self.duration_ms
        if self.total_chunks_searched:
            d["total_chunks_searched"] = self.total_chunks_searched
        return d

    @classmethod
    def from_trace_event(cls, event: Any) -> RetrievalEvent:
        """Build from a TraceEvent with event_type==RETRIEVAL."""
        chunks = [Chunk.from_dict(c) for c in (event.chunks or [])]
        return cls(
            query=event.query or "",
            chunks=chunks,
            vector_store=event.vector_store or "unknown",
            collection=event.collection or "",
            top_k=event.top_k or 0,
            duration_ms=event.duration_ms or 0.0,
        )


@dataclass
class EmbeddingEvent:
    """Recorded embedding call event."""

    input_text: str
    provider: str
    model: str
    dimensions: int = 0
    input_tokens: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0
