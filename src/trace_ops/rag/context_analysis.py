"""Context window usage analysis for RAG traces.

Shows exactly how much of the LLM's context window is consumed by system
prompts, retrieved chunks, the user query, and the response — broken down
per chunk with "used / not used" heuristics.

Usage::

    from trace_ops import load_cassette
    from trace_ops.rag.context_analysis import analyze_context_usage

    trace = load_cassette("cassettes/qa.yaml")
    analysis = analyze_context_usage(trace)
    print(analysis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trace_ops._types import Trace

# Approximate context-window sizes (in tokens) by model name prefix.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "claude-4-sonnet": 200_000,
    "claude-4-opus": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-opus": 200_000,
}

_DEFAULT_WINDOW = 16_000  # fallback


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words × 1.3 (good enough for diagnostics)."""
    return max(1, int(len(text.split()) * 1.3))


def _context_window(model: str | None) -> int:
    if not model:
        return _DEFAULT_WINDOW
    for key, size in MODEL_CONTEXT_WINDOWS.items():
        if key in model.lower():
            return size
    return _DEFAULT_WINDOW


@dataclass
class ChunkUsage:
    """Token usage and heuristic 'used in response' flag for a single chunk."""

    chunk_id: str
    text_preview: str
    estimated_tokens: int
    relevance_score: float
    used_in_response: bool  # heuristic: does chunk text appear in response?


@dataclass
class ContextAnalysis:
    """Complete context-window usage analysis for one RAG trace."""

    model: str
    context_window_size: int
    system_prompt_tokens: int
    retrieved_context_tokens: int
    user_query_tokens: int
    response_tokens: int
    total_input_tokens: int
    context_percent: float  # fraction of input that is retrieved context
    chunk_usages: list[ChunkUsage] = field(default_factory=list)
    unused_chunk_tokens: int = 0

    def __str__(self) -> str:
        lines = [
            "Context Window Analysis:",
            f"  Model: {self.model} ({self.context_window_size:,} token window)",
            (
                f"  Total input tokens: {self.total_input_tokens:,} / "
                f"{self.context_window_size:,} "
                f"({self.total_input_tokens / max(self.context_window_size, 1):.1%})"
            ),
            "",
            "  Breakdown:",
            f"    System prompt:     {self.system_prompt_tokens:>6,} tokens",
            (
                f"    Retrieved context: {self.retrieved_context_tokens:>6,} tokens "
                f"({self.context_percent:.1%} of input)"
                + ("  ← ⚠ HIGH" if self.context_percent > 0.7 else "")
            ),
        ]

        for cu in self.chunk_usages:
            used_marker = "  ← USED" if cu.used_in_response else "  ← not used"
            lines.append(
                f"      [{cu.relevance_score:.2f}] {cu.chunk_id}: "
                f"{cu.estimated_tokens:,} tok{used_marker}"
            )

        lines.extend([
            f"    User query:        {self.user_query_tokens:>6,} tokens",
            f"    Response:          {self.response_tokens:>6,} tokens",
            "",
        ])

        if self.unused_chunk_tokens > 0:
            lines.append(
                f"  ⚠ {self.unused_chunk_tokens:,} tokens spent on unused chunks. "
                f"Consider reducing top_k."
            )

        return "\n".join(lines)


def analyze_context_usage(trace: Trace) -> ContextAnalysis:
    """Analyse how the context window is used in a RAG trace.

    Pairs the first retrieval event with the first LLM request/response
    pair in the trace to produce a :class:`ContextAnalysis`.

    Args:
        trace: A recorded trace with at least one retrieval and one LLM call.

    Returns:
        :class:`ContextAnalysis` with per-chunk breakdowns.
    """
    from trace_ops._types import EventType

    retrieval_events = trace.retrieval_events
    llm_responses = [
        e for e in trace.events if e.event_type == EventType.LLM_RESPONSE
    ]
    llm_requests = [
        e for e in trace.events if e.event_type == EventType.LLM_REQUEST
    ]

    model = (llm_responses[0].model if llm_responses else None) or "unknown"
    window_size = _context_window(model)

    # Extract system prompt tokens from the first LLM request
    system_tokens = 0
    user_query_tokens = 0
    if llm_requests:
        req = llm_requests[0]
        for msg in (req.messages or []):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            tok = _estimate_tokens(str(content))
            if role == "system":
                system_tokens += tok
            elif role == "user":
                user_query_tokens += tok

    # Response tokens
    response_tokens = 0
    response_text = ""
    if llm_responses:
        resp = llm_responses[0]
        response_tokens = resp.output_tokens or 0
        if isinstance(resp.response, dict):
            choices = resp.response.get("choices") or []
            if choices:
                msg = choices[0].get("message", {})
                response_text = msg.get("content", "") or ""
        if not response_tokens and response_text:
            response_tokens = _estimate_tokens(response_text)

    # Chunk usages
    chunk_usages: list[ChunkUsage] = []
    retrieved_tokens = 0
    unused_tokens = 0

    if retrieval_events:
        for chunk in (retrieval_events[0].chunks or []):
            if isinstance(chunk, dict):
                cid = chunk.get("id", "?")
                text = chunk.get("text", "") or ""
                score = float(chunk.get("score", 0.0))
            else:
                cid = getattr(chunk, "id", "?")
                text = getattr(chunk, "text", "") or ""
                score = float(getattr(chunk, "score", 0.0))

            tok = _estimate_tokens(text)
            retrieved_tokens += tok
            used = bool(response_text and len(text) > 20 and text[:40] in response_text)
            if not used:
                unused_tokens += tok

            chunk_usages.append(ChunkUsage(
                chunk_id=str(cid),
                text_preview=text[:80],
                estimated_tokens=tok,
                relevance_score=score,
                used_in_response=used,
            ))

    total_input = (
        (llm_responses[0].input_tokens if llm_responses and llm_responses[0].input_tokens else None)
        or (system_tokens + retrieved_tokens + user_query_tokens)
    )

    context_percent = retrieved_tokens / max(total_input, 1)

    return ContextAnalysis(
        model=model,
        context_window_size=window_size,
        system_prompt_tokens=system_tokens,
        retrieved_context_tokens=retrieved_tokens,
        user_query_tokens=user_query_tokens,
        response_tokens=response_tokens,
        total_input_tokens=total_input,
        context_percent=context_percent,
        chunk_usages=chunk_usages,
        unused_chunk_tokens=unused_tokens,
    )
