"""RAG-specific test assertions for traceops.

These assertions operate on recorded ``Trace`` objects to enforce quality
constraints on RAG pipelines: chunk counts, retrieval latency, context window
usage, minimum relevance scores, retrieval drift, and cached RAGAS/DeepEval
metrics.

All assertions raise ``RAGAssertionError`` (a subclass of ``AssertionError``)
on failure so they work naturally inside pytest tests.

Example::

    from trace_ops import Recorder
    from trace_ops.rag.assertions import (
        assert_chunk_count,
        assert_retrieval_latency,
        assert_no_retrieval_drift,
        assert_rag_scores,
    )

    def test_rag_quality():
        old = load_cassette("cassettes/golden.yaml")
        with Recorder(save_to="cassettes/current.yaml", intercept_rag=True) as rec:
            rag_chain.invoke({"question": "What is our refund policy?"})

        assert_chunk_count(rec.trace, min_chunks=3, max_chunks=10)
        assert_retrieval_latency(rec.trace, max_ms=300)
        assert_no_retrieval_drift(old, rec.trace, min_overlap=0.7)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trace_ops._types import Trace


class RAGAssertionError(AssertionError):
    """Raised when a RAG assertion fails."""


# ── Chunk count ──────────────────────────────────────────────────────────────


def assert_chunk_count(trace: Trace, *, min_chunks: int = 1, max_chunks: int = 20) -> None:
    """Assert the number of retrieved chunks is within bounds for every retrieval.

    Args:
        trace: Recorded trace to check.
        min_chunks: Minimum number of chunks expected per retrieval.
        max_chunks: Maximum number of chunks allowed per retrieval.

    Raises:
        RAGAssertionError: If any retrieval returns too few or too many chunks.
    """
    if not trace.retrieval_events:
        raise RAGAssertionError(
            "No retrieval events found in trace. "
            "Record with intercept_rag=True or call rec.record_retrieval()."
        )
    for event in trace.retrieval_events:
        count = len(event.chunks or [])
        q = (event.query or "")[:80]
        if count < min_chunks:
            raise RAGAssertionError(
                f"Too few chunks retrieved: {count} < {min_chunks} "
                f"for query: '{q}'"
            )
        if count > max_chunks:
            raise RAGAssertionError(
                f"Too many chunks retrieved: {count} > {max_chunks} "
                f"for query: '{q}'"
            )


# ── Retrieval latency ────────────────────────────────────────────────────────


def assert_retrieval_latency(trace: Trace, *, max_ms: float = 500) -> None:
    """Assert every retrieval completed within *max_ms* milliseconds.

    Args:
        trace: Recorded trace to check.
        max_ms: Maximum allowed retrieval duration in milliseconds.

    Raises:
        RAGAssertionError: If any retrieval exceeded the threshold.
    """
    for event in trace.retrieval_events:
        ms = event.duration_ms or 0.0
        q = (event.query or "")[:80]
        if ms > max_ms:
            raise RAGAssertionError(
                f"Retrieval too slow: {ms:.0f}ms > {max_ms:.0f}ms "
                f"for query: '{q}'"
            )


# ── Minimum relevance score ──────────────────────────────────────────────────


def assert_min_relevance_score(trace: Trace, *, min_score: float = 0.5) -> None:
    """Assert every retrieved chunk meets a minimum relevance/similarity score.

    Args:
        trace: Recorded trace to check.
        min_score: Minimum acceptable chunk score (0–1).

    Raises:
        RAGAssertionError: If any chunk falls below the threshold.
    """
    for event in trace.retrieval_events:
        q = (event.query or "")[:80]
        for chunk in event.chunks or []:
            score = chunk.get("score", 0.0) if isinstance(chunk, dict) else getattr(chunk, "score", 0.0)
            cid = chunk.get("id", "?") if isinstance(chunk, dict) else getattr(chunk, "id", "?")
            if score < min_score:
                raise RAGAssertionError(
                    f"Low relevance chunk '{cid}': score={score:.3f} < {min_score:.3f} "
                    f"for query: '{q}'"
                )


# ── Context window usage ─────────────────────────────────────────────────────


def assert_context_window_usage(trace: Trace, *, max_percent: float = 70) -> None:
    """Assert retrieved context doesn't consume too much of the context window.

    Uses a rough ``words × 1.3`` token estimate.  For precise measurements
    use ``analyze_context_usage()`` from ``trace_ops.rag.context_analysis``.

    Args:
        trace: Recorded trace to check.
        max_percent: Maximum percentage of input tokens that may be retrieved context.

    Raises:
        RAGAssertionError: If any LLM call's context is predominantly retrieved chunks.
    """
    from trace_ops._types import EventType

    for retrieval in trace.retrieval_events:
        context_tokens = sum(
            len((c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")).split()) * 1.3
            for c in (retrieval.chunks or [])
        )
        for llm_event in trace.events:
            if llm_event.event_type == EventType.LLM_RESPONSE and (llm_event.input_tokens or 0) > 0:
                ratio = context_tokens / llm_event.input_tokens  # type: ignore[operator]
                if ratio > max_percent / 100:
                    raise RAGAssertionError(
                        f"Context window overloaded: ~{ratio:.0%} of input tokens are retrieved "
                        f"context (threshold: {max_percent}%). Consider reducing top_k."
                    )
                break


# ── Retrieval drift ──────────────────────────────────────────────────────────


def assert_no_retrieval_drift(
    old_trace: Trace,
    new_trace: Trace,
    *,
    max_chunk_diff: int = 2,
    min_overlap: float = 0.6,
) -> None:
    """Assert retrieval results haven't drifted too far from a golden baseline.

    Args:
        old_trace: Golden baseline trace (previously recorded).
        new_trace: Newly recorded trace to compare against.
        max_chunk_diff: Maximum number of chunks that may have changed.
        min_overlap: Minimum fraction of chunks that must be shared (0–1).

    Raises:
        RAGAssertionError: If retrieval results have drifted beyond the thresholds.
    """
    old_retrievals = old_trace.retrieval_events
    new_retrievals = new_trace.retrieval_events

    if len(old_retrievals) != len(new_retrievals):
        raise RAGAssertionError(
            f"Different number of retrieval calls: "
            f"{len(old_retrievals)} → {len(new_retrievals)}"
        )

    for old_r, new_r in zip(old_retrievals, new_retrievals, strict=False):
        def _ids(event):  # noqa: E306
            return {
                (c.get("id", "") if isinstance(c, dict) else getattr(c, "id", ""))
                for c in (event.chunks or [])
            }

        old_ids = _ids(old_r)
        new_ids = _ids(new_r)

        union = old_ids | new_ids
        if not union:
            continue

        overlap = len(old_ids & new_ids) / len(union)
        diff_count = len(old_ids.symmetric_difference(new_ids))
        q = (old_r.query or "")[:60]

        if overlap < min_overlap:
            raise RAGAssertionError(
                f"Retrieval drift detected for '{q}': "
                f"overlap={overlap:.0%} < {min_overlap:.0%}. "
                f"Dropped: {old_ids - new_ids}, Added: {new_ids - old_ids}"
            )
        if diff_count > max_chunk_diff:
            raise RAGAssertionError(
                f"Too many chunk changes for '{q}': "
                f"{diff_count} chunks changed > {max_chunk_diff} max"
            )


# ── Cached RAG scores ────────────────────────────────────────────────────────


def assert_rag_scores(
    trace: Trace,
    *,
    min_faithfulness: float | None = None,
    min_context_precision: float | None = None,
    min_answer_relevancy: float | None = None,
    min_context_recall: float | None = None,
) -> None:
    """Assert pre-computed RAG scores from the cassette meet thresholds.

    These scores are cached in the cassette during recording and cost $0
    to check on replay — no judge LLM calls are made.

    Args:
        trace: Recorded (or replayed) trace with cached RAG scores.
        min_faithfulness: Minimum required faithfulness score (0–1).
        min_context_precision: Minimum required context precision (0–1).
        min_answer_relevancy: Minimum required answer relevancy (0–1).
        min_context_recall: Minimum required context recall (0–1).

    Raises:
        RAGAssertionError: If any specified threshold is not met or scores are missing.
    """
    scores = trace.rag_scores
    if scores is None:
        raise RAGAssertionError(
            "No RAG scores found in cassette. Record with a rag_scorer first:\n"
            "  Recorder(save_to=..., rag_scorer=RagasScorer())"
        )

    checks: dict[str, tuple[float | None, float | None]] = {
        "faithfulness": (min_faithfulness, scores.get("faithfulness")),
        "context_precision": (min_context_precision, scores.get("context_precision")),
        "answer_relevancy": (min_answer_relevancy, scores.get("answer_relevancy")),
        "context_recall": (min_context_recall, scores.get("context_recall")),
    }

    for metric, (threshold, actual) in checks.items():
        if threshold is None:
            continue
        if actual is None:
            raise RAGAssertionError(
                f"Metric '{metric}' not found in cached scores. "
                f"Available: {list(scores.keys())}"
            )
        if actual < threshold:
            raise RAGAssertionError(
                f"RAG score regression: {metric}={actual:.3f} < {threshold:.3f}"
            )
