"""Diff engine — compares two traces and produces human-readable diffs.

This is the core of regression detection: given a recorded trace and
a new trace, what changed? Did the agent take a different path? Did
it make more/fewer LLM calls? Did it call different tools?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepdiff import DeepDiff

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.normalize import normalize_for_comparison


@dataclass
class TraceDiff:
    """Result of comparing two traces."""

    has_changes: bool = False

    # Trajectory-level changes
    trajectory_changed: bool = False
    old_trajectory: list[str] = field(default_factory=list)
    new_trajectory: list[str] = field(default_factory=list)
    old_fingerprint: str = ""
    new_fingerprint: str = ""

    # Summary changes
    llm_calls_delta: int = 0  # positive = more calls, negative = fewer
    tool_calls_delta: int = 0
    token_delta: int = 0
    cost_delta: float = 0.0

    # Specific changes
    added_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)
    changed_models: list[dict[str, str]] = field(default_factory=list)
    response_diffs: list[dict[str, Any]] = field(default_factory=list)

    # Optional RAG diff (populated when rag=True passed to diff_traces)
    rag_diff: Any | None = None  # RAGDiffResult from trace_ops.rag.diff

    # Optional semantic diff (populated when semantic=True passed to diff_traces)
    semantic_diff: Any | None = None  # SemanticDiffResult from trace_ops.semantic

    def summary(self) -> str:
        """Generate a human-readable summary of the diff."""
        if not self.has_changes:
            return "No changes detected — traces are identical."

        lines = ["Trace comparison:"]

        if self.trajectory_changed:
            lines.append("  ⚠ TRAJECTORY CHANGED (agent took a different path)")
            lines.append(f"    Old: {' → '.join(self.old_trajectory)}")
            lines.append(f"    New: {' → '.join(self.new_trajectory)}")

        if self.llm_calls_delta != 0:
            direction = "more" if self.llm_calls_delta > 0 else "fewer"
            lines.append(f"  LLM calls: {abs(self.llm_calls_delta)} {direction}")

        if self.tool_calls_delta != 0:
            direction = "more" if self.tool_calls_delta > 0 else "fewer"
            lines.append(f"  Tool calls: {abs(self.tool_calls_delta)} {direction}")

        if self.added_tools:
            lines.append(f"  New tools used: {', '.join(self.added_tools)}")
        if self.removed_tools:
            lines.append(f"  Tools no longer used: {', '.join(self.removed_tools)}")

        if self.changed_models:
            for change in self.changed_models:
                lines.append(
                    f"  Model changed: {change['old']} → {change['new']} "
                    f"(call #{change['index']})"
                )

        if self.token_delta != 0:
            direction = "more" if self.token_delta > 0 else "fewer"
            lines.append(f"  Tokens: {abs(self.token_delta)} {direction}")

        if abs(self.cost_delta) > 0.001:
            direction = "higher" if self.cost_delta > 0 else "lower"
            lines.append(f"  Cost: ${abs(self.cost_delta):.4f} {direction}")

        if self.response_diffs:
            lines.append(f"  {len(self.response_diffs)} response(s) changed in content")

        if self.rag_diff is not None and self.rag_diff.has_changes:
            lines.append(f"  RAG: {self.rag_diff.total_retrievals_delta:+d} retrievals, "
                         f"{self.rag_diff.chunks_changed} chunk change(s)")

        if self.semantic_diff is not None:
            status = "PASS" if self.semantic_diff.all_passed else "FAIL"
            lines.append(f"  Semantic similarity: [{status}] {self.semantic_diff.summary()}")

        return "\n".join(lines)


def diff_traces(
    old: Trace,
    new: Trace,
    *,
    rag: bool = False,
    semantic: bool = False,
    semantic_threshold: float = 0.85,
) -> TraceDiff:
    """Compare two traces and return a structured diff.

    Args:
        old: The reference trace (from the cassette).
        new: The new trace (from the current run).
        rag: If True, also diff RAG retrieval events.
        semantic: If True, also compute semantic similarity of LLM responses.
        semantic_threshold: Minimum cosine similarity threshold (default 0.85).

    Returns:
        A TraceDiff describing all differences.
    """
    result = TraceDiff(
        old_trajectory=old.trajectory,
        new_trajectory=new.trajectory,
        old_fingerprint=old.fingerprint(),
        new_fingerprint=new.fingerprint(),
    )

    # Check trajectory
    if old.trajectory != new.trajectory:
        result.trajectory_changed = True
        result.has_changes = True

    if old.fingerprint() != new.fingerprint():
        result.has_changes = True

    # Summary deltas
    result.llm_calls_delta = new.total_llm_calls - old.total_llm_calls
    result.tool_calls_delta = new.total_tool_calls - old.total_tool_calls
    result.token_delta = new.total_tokens - old.total_tokens
    result.cost_delta = new.total_cost_usd - old.total_cost_usd

    if any([
        result.llm_calls_delta, result.tool_calls_delta,
        result.token_delta, abs(result.cost_delta) > 0.001,
    ]):
        result.has_changes = True

    # Tool-level diff
    old_tools = {
        e.tool_name for e in old.events
        if e.event_type == EventType.TOOL_CALL and e.tool_name
    }
    new_tools = {
        e.tool_name for e in new.events
        if e.event_type == EventType.TOOL_CALL and e.tool_name
    }
    result.added_tools = sorted(new_tools - old_tools)
    result.removed_tools = sorted(old_tools - new_tools)
    if result.added_tools or result.removed_tools:
        result.has_changes = True

    # Model-level diff (compare LLM calls pairwise)
    old_llm = [e for e in old.events if e.event_type == EventType.LLM_REQUEST]
    new_llm = [e for e in new.events if e.event_type == EventType.LLM_REQUEST]
    for i, (o, n) in enumerate(zip(old_llm, new_llm)):
        if o.model != n.model:
            result.changed_models.append({
                "index": i + 1,
                "old": o.model or "unknown",
                "new": n.model or "unknown",
            })
            result.has_changes = True

    # Response content diff (compare LLM responses pairwise using normalization)
    old_resp = [e for e in old.events if e.event_type == EventType.LLM_RESPONSE]
    new_resp = [e for e in new.events if e.event_type == EventType.LLM_RESPONSE]
    for i, (o, n) in enumerate(zip(old_resp, new_resp)):
        o_norm = normalize_for_comparison(o.response or {}, o.provider or "openai")
        n_norm = normalize_for_comparison(n.response or {}, n.provider or "openai")
        if o_norm != n_norm:
            diff = DeepDiff(o_norm, n_norm, verbose_level=1)
            if diff:
                result.response_diffs.append({
                    "call_index": i + 1,
                    "model": o.model,
                    "diff": dict(diff),
                })
                result.has_changes = True

    # Optional RAG diff
    if rag:
        try:
            from trace_ops.rag.diff import diff_rag
            result.rag_diff = diff_rag(old, new)
            if result.rag_diff.has_changes:
                result.has_changes = True
        except ImportError:
            pass

    # Optional semantic diff (cosine similarity of LLM response embeddings)
    if semantic:
        try:
            from trace_ops.semantic.similarity import semantic_similarity
            result.semantic_diff = semantic_similarity(
                old, new, min_similarity=semantic_threshold
            )
            if not result.semantic_diff.all_passed:
                result.has_changes = True
        except ImportError:
            pass

    return result


def assert_trace_unchanged(
    old: Trace,
    new: Trace,
    *,
    ignore_trajectory: bool = False,
    ignore_responses: bool = False,
    ignore_costs: bool = True,
    ignore_timing: bool = True,
) -> None:
    """Assert that two traces are equivalent for regression testing.

    This is the main assertion used in tests:
        old_trace = load_cassette("cassettes/test_math.yaml")
        # ... run agent ...
        assert_trace_unchanged(old_trace, recorder.trace)

    Args:
        old: The reference trace.
        new: The current trace.
        ignore_trajectory: Don't fail on different agent paths.
        ignore_responses: Don't fail on different LLM response content.
        ignore_costs: Don't fail on cost differences (default: True).
        ignore_timing: Don't fail on timing differences (default: True).

    Raises:
        AssertionError: If the traces differ in checked dimensions.
    """
    diff = diff_traces(old, new)

    if not diff.has_changes:
        return

    # Build failure message from checked dimensions
    failures: list[str] = []

    if diff.trajectory_changed and not ignore_trajectory:
        failures.append(
            f"Trajectory changed:\n"
            f"  Old: {' → '.join(diff.old_trajectory)}\n"
            f"  New: {' → '.join(diff.new_trajectory)}"
        )

    if (diff.added_tools or diff.removed_tools) and not ignore_trajectory:
        if diff.added_tools:
            failures.append(f"New tools used: {diff.added_tools}")
        if diff.removed_tools:
            failures.append(f"Tools removed: {diff.removed_tools}")

    if diff.changed_models:
        failures.append(
            "Models changed: " +
            "; ".join(
                f"call #{c['index']}: {c['old']} → {c['new']}"
                for c in diff.changed_models
            )
        )

    if diff.response_diffs and not ignore_responses:
        failures.append(
            f"{len(diff.response_diffs)} LLM response(s) changed in content"
        )

    if diff.llm_calls_delta and not ignore_trajectory:
        direction = "more" if diff.llm_calls_delta > 0 else "fewer"
        failures.append(
            f"{abs(diff.llm_calls_delta)} {direction} LLM call(s)"
        )

    if diff.cost_delta and not ignore_costs:
        direction = "higher" if diff.cost_delta > 0 else "lower"
        failures.append(f"Cost ${abs(diff.cost_delta):.4f} {direction}")

    if failures:
        raise AssertionError(
            "Agent regression detected:\n  " +
            "\n  ".join(failures) +
            "\n\nRe-record the cassette if this change is intentional."
        )
