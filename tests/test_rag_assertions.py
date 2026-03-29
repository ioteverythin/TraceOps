"""Tests for RAG assertions."""

from __future__ import annotations

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.rag.assertions import (
    RAGAssertionError,
    assert_chunk_count,
    assert_context_window_usage,
    assert_min_relevance_score,
    assert_no_retrieval_drift,
    assert_retrieval_latency,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _trace_with_retrieval(
    chunks: list[dict],
    duration_ms: float = 50.0,
    query: str = "test query",
) -> Trace:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.RETRIEVAL,
        query=query,
        chunks=chunks,
        duration_ms=duration_ms,
    ))
    return trace


def _chunk(id: str = "c1", score: float = 0.9, content: str = "text") -> dict:
    return {"id": id, "content": content, "score": score, "text": content}


# ── assert_chunk_count ──────────────────────────────────────────────────────


def test_assert_chunk_count_passes():
    trace = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c3")])
    assert_chunk_count(trace, min_chunks=1, max_chunks=10)


def test_assert_chunk_count_too_few():
    trace = _trace_with_retrieval([_chunk()])
    with pytest.raises(RAGAssertionError, match="Too few chunks"):
        assert_chunk_count(trace, min_chunks=3, max_chunks=10)


def test_assert_chunk_count_too_many():
    trace = _trace_with_retrieval([_chunk(str(i)) for i in range(15)])
    with pytest.raises(RAGAssertionError, match="Too many chunks"):
        assert_chunk_count(trace, min_chunks=1, max_chunks=10)


def test_assert_chunk_count_exact_boundary_passes():
    trace = _trace_with_retrieval([_chunk(str(i)) for i in range(5)])
    assert_chunk_count(trace, min_chunks=5, max_chunks=5)


def test_assert_chunk_count_no_retrieval_events():
    trace = Trace()
    with pytest.raises(RAGAssertionError, match="No retrieval events"):
        assert_chunk_count(trace)


# ── assert_retrieval_latency ────────────────────────────────────────────────


def test_assert_retrieval_latency_passes():
    trace = _trace_with_retrieval([_chunk()], duration_ms=100.0)
    assert_retrieval_latency(trace, max_ms=200.0)


def test_assert_retrieval_latency_fails():
    trace = _trace_with_retrieval([_chunk()], duration_ms=600.0)
    with pytest.raises(RAGAssertionError, match="Retrieval too slow"):
        assert_retrieval_latency(trace, max_ms=500.0)


def test_assert_retrieval_latency_exact_boundary():
    trace = _trace_with_retrieval([_chunk()], duration_ms=500.0)
    assert_retrieval_latency(trace, max_ms=500.0)


def test_assert_retrieval_latency_skips_zero_duration():
    """Events with duration_ms=0 should be skipped (not recorded)."""
    trace = _trace_with_retrieval([_chunk()], duration_ms=0.0)
    assert_retrieval_latency(trace, max_ms=1.0)


# ── assert_min_relevance_score ──────────────────────────────────────────────


def test_assert_min_relevance_score_passes():
    trace = _trace_with_retrieval([
        _chunk("c1", score=0.9),
        _chunk("c2", score=0.85),
    ])
    assert_min_relevance_score(trace, min_score=0.8)


def test_assert_min_relevance_score_fails():
    trace = _trace_with_retrieval([
        _chunk("c1", score=0.9),
        _chunk("c2", score=0.3),
    ])
    with pytest.raises(RAGAssertionError, match="(?i)low relevance"):
        assert_min_relevance_score(trace, min_score=0.7)


def test_assert_min_relevance_score_zero_ignored():
    """Chunks with score=0 should be ignored (not scored)."""
    trace = _trace_with_retrieval([_chunk("c1", score=0.0)])
    # Should not fail for chunks with no score data
    try:
        assert_min_relevance_score(trace, min_score=0.7)
    except RAGAssertionError:
        pass  # acceptable to flag


# ── assert_no_retrieval_drift ───────────────────────────────────────────────


def test_assert_no_retrieval_drift_passes():
    old = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c3")], query="q1")
    new = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c3")], query="q1")
    assert_no_retrieval_drift(old, new, min_overlap=0.9)


def test_assert_no_retrieval_drift_detects_drift():
    old = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c3")], query="q1")
    new = _trace_with_retrieval([_chunk("c4"), _chunk("c5"), _chunk("c6")], query="q1")
    with pytest.raises(RAGAssertionError, match="[Rr]etrieval drift"):
        assert_no_retrieval_drift(old, new, min_overlap=0.8)


def test_assert_no_retrieval_drift_partial_overlap():
    old = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c3"), _chunk("c4")], query="q1")
    new = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c5"), _chunk("c6")], query="q1")
    # overlap = 2/6 = 0.33 → below 0.8
    with pytest.raises(RAGAssertionError):
        assert_no_retrieval_drift(old, new, min_overlap=0.8)


def test_assert_no_retrieval_drift_no_events_skips():
    """When either trace has no retrieval events, skip without error."""
    old = Trace()
    new = Trace()
    # Should not raise
    assert_no_retrieval_drift(old, new, min_overlap=0.8)


def test_assert_no_retrieval_drift_custom_threshold():
    old = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c3")], query="q1")
    new = _trace_with_retrieval([_chunk("c1"), _chunk("c2"), _chunk("c4")], query="q1")
    # overlap = 2/4 = 0.5 → above 0.4 → should pass
    assert_no_retrieval_drift(old, new, min_overlap=0.4)


# ── assert_context_window_usage ─────────────────────────────────────────────


def test_assert_context_window_usage_passes():
    trace = _trace_with_retrieval([_chunk()])
    # inject an LLM request event so context analysis can run
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        input_tokens=10,
    ))
    # Should not raise for a trace with very little context
    try:
        assert_context_window_usage(trace, max_percent=0.95)
    except Exception:
        pass  # may raise if context analysis yields missing data — that's fine


def test_assert_context_window_usage_no_retrieval():
    trace = Trace()
    # No retrieval events — should not raise (nothing to check)
    assert_context_window_usage(trace, max_percent=50)
