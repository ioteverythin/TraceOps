"""Tests for semantic similarity and assertions."""

from __future__ import annotations

import math

import pytest

from trace_ops._types import EventType, Trace, TraceEvent


# ── ResponseSimilarity dataclass ──────────────────────────────────────────


def test_response_similarity_fields():
    from trace_ops.semantic.similarity import ResponseSimilarity

    rs = ResponseSimilarity(
        index=0,
        similarity=0.97,
        threshold=0.85,
        passed=True,
        old_preview="The capital of France is Paris.",
        new_preview="Paris is the capital of France.",
        verdict="Same meaning",
    )
    assert rs.index == 0
    assert rs.similarity == pytest.approx(0.97)
    assert rs.passed is True


def test_response_similarity_fails_below_threshold():
    from trace_ops.semantic.similarity import ResponseSimilarity

    rs = ResponseSimilarity(
        index=0,
        similarity=0.12,
        threshold=0.85,
        passed=False,
        verdict="Semantically different",
    )
    assert rs.passed is False


# ── SemanticDiffResult dataclass ──────────────────────────────────────────


def test_semantic_diff_result_all_passed():
    from trace_ops.semantic.similarity import ResponseSimilarity, SemanticDiffResult

    sdr = SemanticDiffResult(results=[
        ResponseSimilarity(0, 0.92, 0.85, True, verdict="ok"),
        ResponseSimilarity(1, 0.88, 0.85, True, verdict="ok"),
    ])
    assert sdr.all_passed is True


def test_semantic_diff_result_not_all_passed():
    from trace_ops.semantic.similarity import ResponseSimilarity, SemanticDiffResult

    sdr = SemanticDiffResult(results=[
        ResponseSimilarity(0, 0.92, 0.85, True, verdict="ok"),
        ResponseSimilarity(1, 0.50, 0.85, False, verdict="drift"),
    ])
    assert sdr.all_passed is False


def test_semantic_diff_result_summary():
    from trace_ops.semantic.similarity import ResponseSimilarity, SemanticDiffResult

    sdr = SemanticDiffResult(results=[
        ResponseSimilarity(0, 0.95, 0.85, True, "old text", "new text", "Same meaning"),
    ])
    summary = sdr.summary()
    assert isinstance(summary, str)
    assert "0.9500" in summary or "0.95" in summary


# ── _cosine utility ────────────────────────────────────────────────────────


def test_cosine_identical_vectors():
    from trace_ops.semantic.similarity import _cosine

    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    from trace_ops.semantic.similarity import _cosine

    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_30_degree_angle():
    from trace_ops.semantic.similarity import _cosine

    a = [1.0, 0.0]
    b = [math.cos(math.radians(30)), math.sin(math.radians(30))]
    assert _cosine(a, b) == pytest.approx(math.cos(math.radians(30)), abs=1e-6)


def test_cosine_zero_vector_returns_zero():
    from trace_ops.semantic.similarity import _cosine

    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine(a, b) == 0.0


# ── semantic_similarity() — with mocked _embed ───────────────────────────


def _trace_with_response(text: str) -> Trace:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        response={"choices": [{"message": {"role": "assistant", "content": text}}]},
        provider="openai",
        model="gpt-4o-mini",
    ))
    return trace


def test_semantic_similarity_identical_text(monkeypatch):
    from trace_ops.semantic import similarity as sim_mod

    def fake_embed(texts: list[str], model: str) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(sim_mod, "_embed", fake_embed)

    old = _trace_with_response("hello world")
    new = _trace_with_response("hello world")
    result = sim_mod.semantic_similarity(old, new)
    assert result.all_passed
    assert result.results[0].similarity == pytest.approx(1.0)


def test_semantic_similarity_orthogonal_vectors(monkeypatch):
    from trace_ops.semantic import similarity as sim_mod

    def fake_embed(texts: list[str], model: str) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(sim_mod, "_embed", fake_embed)

    old = _trace_with_response("text1")
    new = _trace_with_response("text2")
    result = sim_mod.semantic_similarity(old, new, min_similarity=0.85)
    assert not result.all_passed
    assert result.results[0].similarity == pytest.approx(0.0)


def test_semantic_similarity_empty_traces():
    from trace_ops.semantic import similarity as sim_mod

    old = Trace()
    new = Trace()
    result = sim_mod.semantic_similarity(old, new)
    assert len(result.results) == 0
    assert result.all_passed


def test_semantic_similarity_no_openai_raises():
    from trace_ops.semantic.similarity import _embed

    try:
        import openai  # noqa: F401
        pytest.skip("openai installed — can't test ImportError path")
    except ImportError:
        with pytest.raises(ImportError, match="openai"):
            _embed(["test"], "text-embedding-3-small")


# ── assert_semantic_similarity() ──────────────────────────────────────────


def test_assert_semantic_similarity_passes(monkeypatch):
    from trace_ops.semantic import similarity as sim_mod
    from trace_ops.semantic.assertions import assert_semantic_similarity

    def fake_embed(texts: list[str], model: str) -> list[list[float]]:
        return [[1.0, 0.0], [1.0, 0.0]]

    monkeypatch.setattr(sim_mod, "_embed", fake_embed)

    old = _trace_with_response("hello")
    new = _trace_with_response("hello")
    # Should not raise
    assert_semantic_similarity(old, new, min_similarity=0.85)


def test_assert_semantic_similarity_fails(monkeypatch):
    from trace_ops.semantic import similarity as sim_mod
    from trace_ops.semantic.assertions import (
        SemanticRegressionError,
        assert_semantic_similarity,
    )

    def fake_embed(texts: list[str], model: str) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(sim_mod, "_embed", fake_embed)

    old = _trace_with_response("completely different text here")
    new = _trace_with_response("an entirely unrelated response")
    with pytest.raises(SemanticRegressionError):
        assert_semantic_similarity(old, new, min_similarity=0.85)


def test_semantic_regression_error_is_assertion_error():
    from trace_ops.semantic.assertions import SemanticRegressionError

    err = SemanticRegressionError("semantic drift detected")
    assert isinstance(err, AssertionError)
    assert "semantic drift detected" in str(err)


# ── diff_traces() with semantic=True ──────────────────────────────────────


def test_diff_traces_semantic_false_no_semantic_diff():
    from trace_ops.diff import diff_traces

    old = _trace_with_response("hello")
    new = _trace_with_response("world")
    result = diff_traces(old, new, semantic=False)
    assert result.semantic_diff is None


def test_diff_traces_semantic_true_populates_on_openai(monkeypatch):
    """With mocked embeddings, diff_traces should populate semantic_diff."""
    from trace_ops.semantic import similarity as sim_mod
    from trace_ops.diff import diff_traces

    def fake_embed(texts: list[str], model: str) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(sim_mod, "_embed", fake_embed)

    old = _trace_with_response("hello")
    new = _trace_with_response("hello")
    result = diff_traces(old, new, semantic=True)
    assert result.semantic_diff is not None
