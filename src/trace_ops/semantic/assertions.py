"""Semantic regression assertions for traceops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trace_ops._types import Trace


class SemanticRegressionError(AssertionError):
    """Raised when semantic similarity drops below the configured threshold."""


def assert_semantic_similarity(
    old_trace: "Trace",
    new_trace: "Trace",
    *,
    min_similarity: float = 0.85,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Assert that all LLM responses are semantically equivalent across traces.

    Uses embedding cosine similarity — passes even if wording changed, fails
    if the *meaning* changed enough to cross the threshold.

    Args:
        old_trace: Baseline (golden) trace.
        new_trace: New trace to compare against.
        min_similarity: Minimum cosine similarity required (0–1).
        embedding_model: OpenAI embedding model for comparison.

    Raises:
        SemanticRegressionError: If any response pair falls below the threshold.
        ImportError: If ``openai`` is not installed.
    """
    from trace_ops.semantic.similarity import semantic_similarity

    result = semantic_similarity(
        old_trace, new_trace,
        min_similarity=min_similarity,
        embedding_model=embedding_model,
    )

    failures = [r for r in result.results if not r.passed]
    if failures:
        lines = [
            f"Semantic regression detected ({len(failures)} of {len(result.results)} response(s) changed):"
        ]
        for r in failures:
            lines.append(
                f"  Response #{r.index + 1}: similarity={r.similarity:.3f} < {r.threshold}\n"
                f'    Old: "{r.old_preview}"\n'
                f'    New: "{r.new_preview}"\n'
                f"    Verdict: {r.verdict}"
            )
        raise SemanticRegressionError("\n".join(lines))
