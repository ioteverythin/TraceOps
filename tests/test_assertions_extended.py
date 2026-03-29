"""Extended assertion tests — edge cases and new helper functions.

Covers boundary conditions for existing assertions and adds
assert_trajectory_includes / assert_no_tool_used.
"""

from __future__ import annotations

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.assertions import (
    AgentLoopError,
    BudgetExceededError,
    assert_cost_under,
    assert_max_llm_calls,
    assert_no_loops,
    assert_tokens_under,
)


def _make_trace_with_events(*events: TraceEvent) -> Trace:
    t = Trace()
    for e in events:
        t.add_event(e)
    return t


# ── assert_cost_under edge cases ─────────────────────────────────────


class TestAssertCostUnderEdgeCases:
    def test_zero_budget_with_any_cost(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE, cost_usd=0.0001)
        )
        with pytest.raises(BudgetExceededError):
            assert_cost_under(t, max_usd=0.0)

    def test_zero_cost_passes_zero_budget(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE, cost_usd=None)
        )
        assert_cost_under(t, max_usd=0.0)  # should not raise

    def test_empty_trace(self):
        t = Trace()
        assert_cost_under(t, max_usd=0.0)  # should not raise

    def test_error_message_includes_details(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE, cost_usd=1.50, input_tokens=1000, output_tokens=500)
        )
        with pytest.raises(BudgetExceededError, match=r"\$1\.5000"):
            assert_cost_under(t, max_usd=1.0)


# ── assert_tokens_under edge cases ───────────────────────────────────


class TestAssertTokensUnderEdgeCases:
    def test_zero_limit_with_tokens(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE, input_tokens=1, output_tokens=0)
        )
        with pytest.raises(BudgetExceededError):
            assert_tokens_under(t, max_tokens=0)

    def test_none_tokens_counted_as_zero(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE, input_tokens=None, output_tokens=None)
        )
        assert_tokens_under(t, max_tokens=0)  # total=0, not > 0

    def test_empty_trace(self):
        t = Trace()
        assert_tokens_under(t, max_tokens=0)


# ── assert_max_llm_calls edge cases ─────────────────────────────────


class TestAssertMaxLLMCallsEdgeCases:
    def test_exactly_at_limit(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE),
            TraceEvent(event_type=EventType.LLM_RESPONSE),
        )
        assert_max_llm_calls(t, max_calls=2)  # exactly at limit — should pass

    def test_one_over_limit(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_RESPONSE),
            TraceEvent(event_type=EventType.LLM_RESPONSE),
            TraceEvent(event_type=EventType.LLM_RESPONSE),
        )
        with pytest.raises(BudgetExceededError, match="3 LLM calls"):
            assert_max_llm_calls(t, max_calls=2)

    def test_zero_calls_zero_limit(self):
        t = Trace()
        assert_max_llm_calls(t, max_calls=0)

    def test_error_message_includes_trajectory(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"),
            TraceEvent(event_type=EventType.LLM_RESPONSE),
            TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"),
            TraceEvent(event_type=EventType.LLM_RESPONSE),
        )
        with pytest.raises(BudgetExceededError, match="Trajectory"):
            assert_max_llm_calls(t, max_calls=1)


# ── assert_no_loops edge cases ───────────────────────────────────────


class TestAssertNoLoopsEdgeCases:
    def test_empty_trace(self):
        t = Trace()
        assert_no_loops(t)  # no tool events — should not raise

    def test_single_tool_call(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search")
        )
        assert_no_loops(t)

    def test_exactly_at_threshold(self):
        """3 consecutive same-tool = OK with default max=3."""
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        )
        assert_no_loops(t)  # exactly at threshold — should pass

    def test_one_over_threshold(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        )
        with pytest.raises(AgentLoopError, match="4 consecutive"):
            assert_no_loops(t)

    def test_custom_threshold_1(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        )
        with pytest.raises(AgentLoopError):
            assert_no_loops(t, max_consecutive_same_tool=1)

    def test_interleaved_resets_counter(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="read"),  # reset
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        )
        assert_no_loops(t)  # never exceeds 3 consecutive

    def test_error_message_includes_tool_name(self):
        t = _make_trace_with_events(
            *[TraceEvent(event_type=EventType.TOOL_CALL, tool_name="retry_api") for _ in range(5)]
        )
        with pytest.raises(AgentLoopError, match="retry_api"):
            assert_no_loops(t)


# ── New: assert_trajectory_includes ──────────────────────────────────


def assert_trajectory_includes(
    trace: Trace,
    *,
    expected_tools: list[str] | None = None,
    expected_decisions: list[str] | None = None,
) -> None:
    """Assert that a trace's trajectory includes expected tool calls and/or decisions.

    Args:
        trace: The recorded agent trace.
        expected_tools: Tool names that must appear in the trajectory.
        expected_decisions: Decision labels that must appear in the trajectory.

    Raises:
        AssertionError: If any expected item is missing from the trajectory.
    """
    trajectory = trace.trajectory

    if expected_tools:
        tool_names = {
            step.split(":", 1)[1]
            for step in trajectory
            if step.startswith("tool:")
        }
        for tool in expected_tools:
            if tool not in tool_names:
                raise AssertionError(
                    f"Expected tool '{tool}' in trajectory, but it was not found.\n"
                    f"Trajectory: {' → '.join(trajectory)}\n"
                    f"Tools used: {sorted(tool_names)}"
                )

    if expected_decisions:
        decision_names = {
            step.split(":", 1)[1]
            for step in trajectory
            if step.startswith("decision:")
        }
        for decision in expected_decisions:
            if decision not in decision_names:
                raise AssertionError(
                    f"Expected decision '{decision}' in trajectory, but not found.\n"
                    f"Trajectory: {' → '.join(trajectory)}\n"
                    f"Decisions made: {sorted(decision_names)}"
                )


def assert_no_tool_used(trace: Trace, tool_name: str) -> None:
    """Assert that a specific tool was NOT used in the trace.

    Args:
        trace: The recorded agent trace.
        tool_name: The tool name that must NOT appear.

    Raises:
        AssertionError: If the tool appears in the trajectory.
    """
    tool_names = {
        step.split(":", 1)[1]
        for step in trace.trajectory
        if step.startswith("tool:")
    }
    if tool_name in tool_names:
        raise AssertionError(
            f"Tool '{tool_name}' was used but should not have been.\n"
            f"Trajectory: {' → '.join(trace.trajectory)}"
        )


# ── Tests for new assertions ─────────────────────────────────────────


class TestAssertTrajectoryIncludes:
    def test_tools_present(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="read_file"),
        )
        assert_trajectory_includes(t, expected_tools=["search", "read_file"])

    def test_tool_missing_raises(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        )
        with pytest.raises(AssertionError, match="Expected tool 'browse'"):
            assert_trajectory_includes(t, expected_tools=["search", "browse"])

    def test_decisions_present(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.AGENT_DECISION, decision="delegate"),
            TraceEvent(event_type=EventType.AGENT_DECISION, decision="respond"),
        )
        assert_trajectory_includes(t, expected_decisions=["delegate", "respond"])

    def test_decision_missing_raises(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.AGENT_DECISION, decision="delegate"),
        )
        with pytest.raises(AssertionError, match="Expected decision 'escalate'"):
            assert_trajectory_includes(t, expected_decisions=["escalate"])

    def test_empty_trace(self):
        t = Trace()
        assert_trajectory_includes(t, expected_tools=[])  # empty list OK

    def test_none_params_ok(self):
        t = Trace()
        assert_trajectory_includes(t, expected_tools=None, expected_decisions=None)


class TestAssertNoToolUsed:
    def test_tool_not_present(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        )
        assert_no_tool_used(t, "dangerous_tool")

    def test_tool_present_raises(self):
        t = _make_trace_with_events(
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="delete_all"),
        )
        with pytest.raises(AssertionError, match="delete_all"):
            assert_no_tool_used(t, "delete_all")

    def test_empty_trace(self):
        t = Trace()
        assert_no_tool_used(t, "anything")
