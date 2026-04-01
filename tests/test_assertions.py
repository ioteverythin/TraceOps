"""Tests for the assertions module."""

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

# ── Helpers ─────────────────────────────────────────────────────────


def _llm_event(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost_usd: float = 0.001,
    model: str = "gpt-4o",
) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        response={"choices": [{"message": {"content": "hello"}}]},
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _tool_event(name: str = "search") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        provider="openai",
        tool_name=name,
        tool_input={"q": "test"},
        tool_output="ok",
    )


def _make_trace(events: list[TraceEvent]) -> Trace:
    trace = Trace()
    for e in events:
        trace.add_event(e)
    trace.finalize()
    return trace


# ── assert_cost_under ───────────────────────────────────────────────


class TestAssertCostUnder:
    def test_within_budget(self):
        trace = _make_trace([_llm_event(cost_usd=0.01), _llm_event(cost_usd=0.02)])
        assert_cost_under(trace, max_usd=0.05)  # should not raise

    def test_over_budget(self):
        trace = _make_trace([_llm_event(cost_usd=0.03), _llm_event(cost_usd=0.04)])
        with pytest.raises(BudgetExceededError, match="0.05"):
            assert_cost_under(trace, max_usd=0.05)

    def test_exact_budget(self):
        trace = _make_trace([_llm_event(cost_usd=0.025), _llm_event(cost_usd=0.025)])
        # Exact match should be within budget (<=)
        assert_cost_under(trace, max_usd=0.05)

    def test_no_cost_metadata(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
        )
        trace = _make_trace([event])
        assert_cost_under(trace, max_usd=1.0)  # should not raise — 0 cost


# ── assert_tokens_under ─────────────────────────────────────────────


class TestAssertTokensUnder:
    def test_within_budget(self):
        trace = _make_trace([
            _llm_event(input_tokens=100, output_tokens=50),
            _llm_event(input_tokens=200, output_tokens=100),
        ])
        assert_tokens_under(trace, max_tokens=500)

    def test_over_budget(self):
        trace = _make_trace([
            _llm_event(input_tokens=300, output_tokens=200),
            _llm_event(input_tokens=200, output_tokens=100),
        ])
        with pytest.raises(BudgetExceededError, match="token"):
            assert_tokens_under(trace, max_tokens=500)


# ── assert_max_llm_calls ────────────────────────────────────────────


class TestAssertMaxLLMCalls:
    def test_within_limit(self):
        trace = _make_trace([_llm_event() for _ in range(5)])
        assert_max_llm_calls(trace, max_calls=10)

    def test_over_limit(self):
        trace = _make_trace([_llm_event() for _ in range(11)])
        with pytest.raises(BudgetExceededError, match="11"):
            assert_max_llm_calls(trace, max_calls=10)

    def test_tool_calls_not_counted(self):
        events = [_llm_event() for _ in range(3)] + [_tool_event() for _ in range(10)]
        trace = _make_trace(events)
        assert_max_llm_calls(trace, max_calls=5)  # only 3 LLM calls


# ── assert_no_loops ─────────────────────────────────────────────────


class TestAssertNoLoops:
    def test_no_loop(self):
        events = [
            _tool_event("search"),
            _tool_event("read"),
            _tool_event("write"),
        ]
        trace = _make_trace(events)
        assert_no_loops(trace, max_consecutive_same_tool=3)

    def test_loop_detected(self):
        events = [
            _tool_event("search"),
            _tool_event("search"),
            _tool_event("search"),
            _tool_event("search"),
        ]
        trace = _make_trace(events)
        with pytest.raises(AgentLoopError, match="search"):
            assert_no_loops(trace, max_consecutive_same_tool=3)

    def test_interleaved_same_tool_ok(self):
        events = [
            _tool_event("search"),
            _tool_event("read"),
            _tool_event("search"),
            _tool_event("read"),
            _tool_event("search"),
        ]
        trace = _make_trace(events)
        assert_no_loops(trace, max_consecutive_same_tool=3)

    def test_custom_threshold(self):
        events = [_tool_event("write")] * 5
        trace = _make_trace(events)
        with pytest.raises(AgentLoopError):
            assert_no_loops(trace, max_consecutive_same_tool=4)
