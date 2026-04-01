"""Tests for RAG diff — diff_rag() and related dataclasses."""

from __future__ import annotations

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.rag.diff import ChunkDiff, RAGDiffResult, RetrievalDiff, diff_rag

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_trace(retrievals: list[dict]) -> Trace:
    """Build a Trace with RETRIEVAL events from the given list of dicts.

    Each dict: {"query": str, "chunks": list[dict], "duration_ms": float}
    """
    trace = Trace()
    for r in retrievals:
        trace.add_event(TraceEvent(
            event_type=EventType.RETRIEVAL,
            query=r.get("query", "q"),
            chunks=r.get("chunks", []),
            duration_ms=r.get("duration_ms", 50.0),
            vector_store=r.get("vector_store", "chromadb"),
        ))
    return trace


def _chunk(id: str, score: float = 0.9, text: str = "text") -> dict:
    return {"id": id, "score": score, "text": text, "content": text}


# ── ChunkDiff ───────────────────────────────────────────────────────────────


def test_chunk_diff_fields():
    cd = ChunkDiff(status="added", chunk_id="c1", new_score=0.85, text_preview="foo")
    assert cd.status == "added"
    assert cd.chunk_id == "c1"
    assert cd.old_score is None
    assert cd.new_score == 0.85


# ── RetrievalDiff ───────────────────────────────────────────────────────────


def test_retrieval_diff_has_drift_with_changes():
    rd = RetrievalDiff(
        query="q",
        chunk_diffs=[
            ChunkDiff(status="added", chunk_id="c_new"),
            ChunkDiff(status="kept", chunk_id="c_old"),
        ],
    )
    assert rd.has_drift is True


def test_retrieval_diff_no_drift_all_kept():
    rd = RetrievalDiff(
        query="q",
        chunk_diffs=[ChunkDiff(status="kept", chunk_id="c1")],
    )
    assert rd.has_drift is False


# ── diff_rag() ──────────────────────────────────────────────────────────────


def test_diff_rag_identical_traces():
    chunks = [_chunk("c1"), _chunk("c2")]
    old = _make_trace([{"query": "q1", "chunks": chunks}])
    new = _make_trace([{"query": "q1", "chunks": chunks}])
    result = diff_rag(old, new)
    assert isinstance(result, RAGDiffResult)
    assert not result.retriever_changed


def test_diff_rag_detects_added_chunk():
    old_chunks = [_chunk("c1"), _chunk("c2")]
    new_chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    old = _make_trace([{"query": "q1", "chunks": old_chunks}])
    new = _make_trace([{"query": "q1", "chunks": new_chunks}])
    result = diff_rag(old, new)
    assert result.retriever_changed


def test_diff_rag_detects_removed_chunk():
    old_chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    new_chunks = [_chunk("c1"), _chunk("c2")]
    old = _make_trace([{"query": "q1", "chunks": old_chunks}])
    new = _make_trace([{"query": "q1", "chunks": new_chunks}])
    result = diff_rag(old, new)
    assert result.retriever_changed


def test_diff_rag_multiple_queries():
    old = _make_trace([
        {"query": "q1", "chunks": [_chunk("c1"), _chunk("c2")]},
        {"query": "q2", "chunks": [_chunk("c3"), _chunk("c4")]},
    ])
    new = _make_trace([
        {"query": "q1", "chunks": [_chunk("c1"), _chunk("c2")]},
        {"query": "q2", "chunks": [_chunk("c3"), _chunk("c5")]},  # c4 → c5
    ])
    result = diff_rag(old, new)
    assert result.retriever_changed
    assert len(result.retrieval_diffs) >= 1


def test_diff_rag_empty_traces():
    old = Trace()
    new = Trace()
    result = diff_rag(old, new)
    assert not result.has_changes
    assert len(result.retrieval_diffs) == 0


def test_diff_rag_has_changes_property():
    old_chunks = [_chunk("c1")]
    new_chunks = [_chunk("c2")]
    old = _make_trace([{"query": "q", "chunks": old_chunks}])
    new = _make_trace([{"query": "q", "chunks": new_chunks}])
    result = diff_rag(old, new)
    assert result.has_changes


def test_diff_rag_summary_str():
    old_chunks = [_chunk("c1"), _chunk("c2")]
    new_chunks = [_chunk("c1"), _chunk("c3")]
    old = _make_trace([{"query": "q1", "chunks": old_chunks}])
    new = _make_trace([{"query": "q1", "chunks": new_chunks}])
    result = diff_rag(old, new)
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_diff_rag_total_retrievals_delta():
    old = _make_trace([{"query": "q1", "chunks": [_chunk("c1")]}])
    new = _make_trace([
        {"query": "q1", "chunks": [_chunk("c1")]},
        {"query": "q2", "chunks": [_chunk("c2")]},
    ])
    result = diff_rag(old, new)
    assert result.total_retrievals_delta == 1


# ── Integration with diff_traces() ─────────────────────────────────────────


def test_diff_traces_with_rag_flag():
    from trace_ops.diff import diff_traces

    old_chunks = [_chunk("c1"), _chunk("c2")]
    new_chunks = [_chunk("c1"), _chunk("c3")]
    old = _make_trace([{"query": "q", "chunks": old_chunks}])
    new = _make_trace([{"query": "q", "chunks": new_chunks}])

    result = diff_traces(old, new, rag=True)
    assert result.rag_diff is not None


def test_diff_traces_without_rag_flag():
    from trace_ops.diff import diff_traces

    old = _make_trace([{"query": "q", "chunks": [_chunk("c1")]}])
    new = _make_trace([{"query": "q", "chunks": [_chunk("c1")]}])
    result = diff_traces(old, new, rag=False)
    assert result.rag_diff is None
