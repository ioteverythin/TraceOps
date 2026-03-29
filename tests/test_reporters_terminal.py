"""Tests for the terminal reporter / time-travel debugger."""

from __future__ import annotations

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.reporters.terminal import TraceDebugger


# ── Helpers ─────────────────────────────────────────────────────────


def _llm_event(content: str = "hello", model: str = "gpt-4o") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        response={"choices": [{"message": {"content": content}}]},
        input_tokens=10,
        output_tokens=5,
    )


def _tool_event(name: str = "search") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        provider="openai",
        tool_name=name,
        tool_input={"q": "test"},
        tool_output="ok",
    )


def _make_trace(events: list[TraceEvent] | None = None) -> Trace:
    trace = Trace()
    for e in events or []:
        trace.add_event(e)
    trace.finalize()
    return trace


# ── Tests ───────────────────────────────────────────────────────────


class TestTraceDebugger:
    def test_init(self):
        trace = _make_trace([_llm_event(), _tool_event()])
        debugger = TraceDebugger(trace)
        assert debugger.trace is trace
        assert debugger._index == 0

    def test_event_count(self):
        events = [_llm_event(), _tool_event(), _llm_event()]
        trace = _make_trace(events)
        debugger = TraceDebugger(trace)
        assert len(debugger._events) == 3

    def test_event_filter_tools_only(self):
        events = [_llm_event(), _tool_event("a"), _llm_event(), _tool_event("b")]
        trace = _make_trace(events)
        debugger = TraceDebugger(
            trace,
            event_filter={EventType.TOOL_CALL, EventType.TOOL_RESULT},
        )
        assert all(
            e.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
            for e in debugger._events
        )
        assert len(debugger._events) == 2

    def test_event_filter_llm_only(self):
        events = [_llm_event(), _tool_event(), _llm_event()]
        trace = _make_trace(events)
        debugger = TraceDebugger(
            trace,
            event_filter={EventType.LLM_REQUEST, EventType.LLM_RESPONSE},
        )
        assert all(
            e.event_type in (EventType.LLM_REQUEST, EventType.LLM_RESPONSE)
            for e in debugger._events
        )
        assert len(debugger._events) == 2

    def test_navigation_next(self):
        events = [_llm_event(), _tool_event(), _llm_event()]
        trace = _make_trace(events)
        debugger = TraceDebugger(trace)
        assert debugger._index == 0
        debugger._step(1)
        assert debugger._index == 1
        debugger._step(1)
        assert debugger._index == 2

    def test_navigation_prev(self):
        events = [_llm_event(), _tool_event()]
        trace = _make_trace(events)
        debugger = TraceDebugger(trace)
        debugger._index = 1
        debugger._step(-1)
        assert debugger._index == 0

    def test_navigation_prev_at_start(self):
        trace = _make_trace([_llm_event()])
        debugger = TraceDebugger(trace)
        debugger._step(-1)  # should not go negative
        assert debugger._index == 0

    def test_navigation_next_at_end(self):
        events = [_llm_event(), _tool_event()]
        trace = _make_trace(events)
        debugger = TraceDebugger(trace)
        debugger._index = 1
        debugger._step(1)  # should stay at last
        assert debugger._index == 1

    def test_goto(self):
        events = [_llm_event() for _ in range(5)]
        trace = _make_trace(events)
        debugger = TraceDebugger(trace)
        debugger._goto("g 4")  # 1-indexed → goes to index 3
        assert debugger._index == 3

    def test_goto_out_of_range(self):
        events = [_llm_event() for _ in range(3)]
        trace = _make_trace(events)
        debugger = TraceDebugger(trace)
        debugger._goto("g 100")  # prints error, index stays
        assert debugger._index == 0

    def test_with_comparison_trace(self):
        trace1 = _make_trace([_llm_event("response A")])
        trace2 = _make_trace([_llm_event("response B")])
        debugger = TraceDebugger(trace1, compare_trace=trace2)
        assert debugger.compare_trace is trace2
