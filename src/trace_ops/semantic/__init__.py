"""Semantic regression detection for traceops.

Uses embedding cosine similarity to detect when LLM responses change
meaning across traces — even when the exact wording differs.

Usage::

    from trace_ops import load_cassette
    from trace_ops.semantic import assert_semantic_similarity, semantic_similarity

    old = load_cassette("cassettes/golden.yaml")
    new = load_cassette("cassettes/current.yaml")

    # Assertion style (raises on regression)
    assert_semantic_similarity(old, new, min_similarity=0.85)

    # Analysis style (returns scores)
    result = semantic_similarity(old, new)
    print(result.summary())
"""

from trace_ops.semantic.assertions import SemanticRegressionError, assert_semantic_similarity
from trace_ops.semantic.similarity import ResponseSimilarity, SemanticDiffResult, semantic_similarity

__all__ = [
    "semantic_similarity",
    "SemanticDiffResult",
    "ResponseSimilarity",
    "assert_semantic_similarity",
    "SemanticRegressionError",
]
