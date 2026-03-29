"""Tests for RAG recording — Recorder.record_retrieval() and auto-intercept data models."""

from __future__ import annotations

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.rag.recorder import Chunk, EmbeddingEvent, RetrievalEvent


# ── Chunk dataclass ────────────────────────────────────────────────────────


def test_chunk_to_dict_basic():
    chunk = Chunk(id="c1", text="hello world", score=0.95)
    d = chunk.to_dict()
    assert d["id"] == "c1"
    assert d["text"] == "hello world"
    assert d["score"] == 0.95


def test_chunk_to_dict_with_metadata():
    chunk = Chunk(id="c2", text="test", score=0.5, metadata={"source": "doc1"})
    d = chunk.to_dict()
    assert d["metadata"] == {"source": "doc1"}


def test_chunk_round_trip():
    chunk = Chunk(id="abc", text="some text", score=0.7, metadata={"page": 3})
    d = chunk.to_dict()
    restored = Chunk.from_dict(d)
    assert restored.id == chunk.id
    assert restored.text == chunk.text
    assert restored.score == chunk.score
    assert restored.metadata == chunk.metadata


def test_chunk_from_dict_defaults():
    chunk = Chunk.from_dict({})
    assert chunk.id == ""
    assert chunk.text == ""
    assert chunk.score == 0.0
    assert chunk.metadata == {}


# ── RetrievalEvent dataclass ────────────────────────────────────────────────


def test_retrieval_event_to_dict():
    chunks = [Chunk(id="c1", text="foo", score=0.9)]
    event = RetrievalEvent(
        query="What is the refund policy?",
        chunks=chunks,
        vector_store="chromadb",
        collection="docs",
        top_k=5,
        duration_ms=45.2,
    )
    d = event.to_dict()
    assert d["query"] == "What is the refund policy?"
    assert len(d["chunks"]) == 1
    assert d["vector_store"] == "chromadb"
    assert d["collection"] == "docs"
    assert d["top_k"] == 5
    assert d["duration_ms"] == 45.2


def test_retrieval_event_optional_fields_omitted():
    event = RetrievalEvent(query="q", chunks=[])
    d = event.to_dict()
    assert "collection" not in d
    assert "top_k" not in d
    assert "embedding_model" not in d
    assert "duration_ms" not in d


def test_retrieval_event_from_trace_event():
    te = TraceEvent(
        event_type=EventType.RETRIEVAL,
        query="test query",
        chunks=[{"id": "c1", "text": "foo", "score": 0.8}],
        vector_store="pinecone",
        collection="knowledge",
        top_k=3,
        duration_ms=22.0,
    )
    re = RetrievalEvent.from_trace_event(te)
    assert re.query == "test query"
    assert len(re.chunks) == 1
    assert re.chunks[0].id == "c1"
    assert re.vector_store == "pinecone"


# ── EmbeddingEvent dataclass ────────────────────────────────────────────────


def test_embedding_event_fields():
    event = EmbeddingEvent(
        input_text="embed this",
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        input_tokens=3,
        duration_ms=20.0,
        cost_usd=0.00001,
    )
    assert event.input_text == "embed this"
    assert event.provider == "openai"
    assert event.dimensions == 1536
    assert event.cost_usd == 0.00001


# ── Trace retrieval_events / embedding_events / mcp_events properties ──────


def _make_trace_with_retrieval() -> Trace:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.RETRIEVAL,
        query="What is AI?",
        chunks=[{"id": "c1", "text": "AI stands for artificial intelligence", "score": 0.9}],
        vector_store="chromadb",
        duration_ms=50.0,
    ))
    return trace


def test_trace_retrieval_events_property():
    trace = _make_trace_with_retrieval()
    events = trace.retrieval_events
    assert len(events) == 1
    assert events[0].event_type == EventType.RETRIEVAL


def test_trace_embedding_events_property():
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.EMBEDDING_CALL,
        dimensions=1536,
        duration_ms=10.0,
    ))
    events = trace.embedding_events
    assert len(events) == 1
    assert events[0].event_type == EventType.EMBEDDING_CALL


def test_trace_mcp_events_property():
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="my-server",
        tool_name="search",
        arguments={"query": "hello"},
    ))
    events = trace.mcp_events
    assert len(events) == 1


def test_trace_rag_scores_property_empty():
    trace = Trace()
    assert trace.rag_scores is None or isinstance(trace.rag_scores, dict)


def test_trace_rag_scores_property_with_event():
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.RAG_SCORES,
        scores={"faithfulness": 0.85, "answer_relevancy": 0.90},
    ))
    scores = trace.rag_scores
    assert scores is not None
    assert scores.get("faithfulness") == 0.85


# ── Recorder.record_retrieval() ────────────────────────────────────────────


def test_recorder_record_retrieval_manual():
    """record_retrieval() should append a RETRIEVAL event to the trace."""
    from trace_ops.recorder import Recorder

    rec = Recorder()
    with rec:
        rec.record_retrieval(
            query="What is the capital of France?",
            retrieved_chunks=[
                {"id": "c1", "content": "Paris is the capital of France", "score": 0.95},
            ],
            vector_store="chromadb",
            collection="world_facts",
            duration_ms=35.0,
        )

    retrieval_events = rec.trace.retrieval_events
    assert len(retrieval_events) == 1
    ev = retrieval_events[0]
    assert ev.query == "What is the capital of France?"
    assert ev.vector_store == "chromadb"
    assert len(ev.chunks) == 1


def test_recorder_record_retrieval_multiple():
    from trace_ops.recorder import Recorder

    rec = Recorder()
    with rec:
        rec.record_retrieval(query="q1", retrieved_chunks=[])
        rec.record_retrieval(query="q2", retrieved_chunks=[{"id": "x", "content": "y", "score": 0.5}])
        rec.record_retrieval(query="q3", retrieved_chunks=[])

    assert len(rec.trace.retrieval_events) == 3


def test_recorder_record_retrieval_top_k():
    from trace_ops.recorder import Recorder

    rec = Recorder()
    with rec:
        rec.record_retrieval(
            query="q",
            retrieved_chunks=[{"id": str(i), "content": f"chunk {i}", "score": 0.9 - i * 0.05} for i in range(5)],
            top_k=5,
        )

    ev = rec.trace.retrieval_events[0]
    assert ev.top_k == 5
    assert len(ev.chunks) == 5
