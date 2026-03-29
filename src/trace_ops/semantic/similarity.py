"""Embedding-based semantic similarity between trace responses.

Uses OpenAI embeddings (or any compatible provider) to compute cosine
similarity between LLM responses across two traces.  If similarity drops
below a threshold the diff is flagged as a semantic regression — even if
the exact wording changed.

Requires ``pip install traceops[semantic]`` (needs ``openai``).

Usage::

    from trace_ops import load_cassette
    from trace_ops.semantic.similarity import semantic_similarity

    old = load_cassette("cassettes/v1.yaml")
    new = load_cassette("cassettes/v2.yaml")
    results = semantic_similarity(old, new)
    for r in results:
        print(f"Response #{r.index}: similarity={r.similarity:.3f}  {'✅' if r.passed else '⚠'}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops._types import Trace

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_THRESHOLD = 0.85


@dataclass
class ResponseSimilarity:
    """Semantic similarity result for one LLM response pair."""

    index: int
    similarity: float
    threshold: float
    passed: bool
    old_preview: str = ""
    new_preview: str = ""
    verdict: str = ""


@dataclass
class SemanticDiffResult:
    """Complete semantic diff across all LLM responses in two traces."""

    results: list[ResponseSimilarity] = field(default_factory=list)
    embedding_model: str = _DEFAULT_MODEL

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = ["Semantic Analysis:"]
        for r in self.results:
            icon = "✅" if r.passed else "⚠ DRIFT"
            lines.append(
                f"  LLM Response #{r.index + 1}: similarity={r.similarity:.3f} "
                f"(threshold: {r.threshold}) {icon}"
            )
            lines.append(f'    Old: "{r.old_preview}"')
            lines.append(f'    New: "{r.new_preview}"')
            lines.append(f"    Verdict: {r.verdict}")
        return "\n".join(lines)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed(texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts using the OpenAI Embeddings API."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "Install openai for semantic similarity: pip install traceops[semantic]"
        ) from exc

    client = openai.OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _extract_responses(trace: "Trace") -> list[str]:
    """Extract final LLM response text strings from a trace."""
    from trace_ops._types import EventType

    responses: list[str] = []
    for e in trace.events:
        if e.event_type != EventType.LLM_RESPONSE:
            continue
        resp = e.response or {}
        choices = resp.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
        else:
            content = str(resp)
        responses.append(content)
    return responses


def semantic_similarity(
    old_trace: "Trace",
    new_trace: "Trace",
    *,
    min_similarity: float = _DEFAULT_THRESHOLD,
    embedding_model: str = _DEFAULT_MODEL,
) -> SemanticDiffResult:
    """Compute embedding-based similarity between LLM responses in two traces.

    Args:
        old_trace: Baseline trace.
        new_trace: New trace to compare.
        min_similarity: Similarity threshold below which a response is flagged.
        embedding_model: OpenAI embedding model name.

    Returns:
        :class:`SemanticDiffResult` with per-response similarity scores.
    """
    old_responses = _extract_responses(old_trace)
    new_responses = _extract_responses(new_trace)

    all_texts = old_responses + new_responses
    if not all_texts:
        return SemanticDiffResult(results=[], embedding_model=embedding_model)

    embeddings = _embed(all_texts, embedding_model)
    n = len(old_responses)

    results: list[ResponseSimilarity] = []
    for i, (old_r, new_r) in enumerate(zip(old_responses, new_responses)):
        sim = _cosine(embeddings[i], embeddings[n + i])
        passed = sim >= min_similarity
        if passed:
            verdict = "Same meaning, possibly different wording"
        elif sim >= 0.7:
            verdict = "Partial semantic match — possible regression"
        else:
            verdict = "Semantically different — likely regression"

        results.append(ResponseSimilarity(
            index=i,
            similarity=round(sim, 4),
            threshold=min_similarity,
            passed=passed,
            old_preview=(old_r or "")[:80],
            new_preview=(new_r or "")[:80],
            verdict=verdict,
        ))

    return SemanticDiffResult(results=results, embedding_model=embedding_model)
