"""Tests for the terminal debugger rendering layer.

Covers _render_messages, _render_response, _render_tool_call,
_render_tool_result, _render_decision, _render_error, _trace_summary,
_role_colour, _extract_content, and navigation edge cases.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from trace_ops._types import EventType, Trace, TraceEvent, TraceMetadata
from trace_ops.reporters.terminal import (
    TraceDebugger,
    _extract_content,
    _role_colour,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _capture_console(debugger: TraceDebugger) -> str:
    """Replace the debugger's console with a capturing one and render current."""
    buf = StringIO()
    debugger.console = Console(
        file=buf, force_terminal=False, no_color=True, highlight=False, width=120
    )
    debugger._render_current()
    return buf.getvalue()


def _make_trace(*events: TraceEvent) -> Trace:
    trace = Trace(metadata=TraceMetadata(description="test"))
    for e in events:
        trace.events.append(e)
    trace.finalize()
    return trace


# ── _role_colour tests ───────────────────────────────────────────────


class TestRoleColour:
    def test_known_roles(self):
        assert _role_colour("system") == "dim"
        assert _role_colour("user") == "cyan"
        assert _role_colour("assistant") == "green"
        assert _role_colour("tool") == "yellow"

    def test_unknown_role(self):
        assert _role_colour("unknown") == "white"
        assert _role_colour("") == "white"


# ── _extract_content tests ───────────────────────────────────────────


class TestExtractContent:
    def test_openai_style(self):
        resp = {
            "choices": [
                {"message": {"content": "Hello from OpenAI"}}
            ]
        }
        assert _extract_content(resp) == "Hello from OpenAI"

    def test_anthropic_style(self):
        resp = {
            "content": [
                {"type": "text", "text": "Hello from Anthropic"}
            ]
        }
        assert _extract_content(resp) == "Hello from Anthropic"

    def test_generic_content_key(self):
        resp = {"content": "Just a string"}
        assert _extract_content(resp) == "Just a string"

    def test_no_content(self):
        resp = {"other": "data"}
        assert _extract_content(resp) is None

    def test_empty_choices(self):
        resp = {"choices": []}
        # Falls through — tries anthropic then generic
        assert _extract_content(resp) is None

    def test_anthropic_non_text_block(self):
        # When no text block is found, _extract_content falls through to
        # the generic response.get("content") which returns the list
        resp = {
            "content": [
                {"type": "tool_use", "name": "search"}
            ]
        }
        result = _extract_content(resp)
        # Falls through to generic: returns the content list itself
        assert result is not None


# ── _render_messages ─────────────────────────────────────────────────


class TestRenderMessages:
    def test_renders_message_roles(self):
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        assert "system" in output
        assert "user" in output
        assert "What is 2+2?" in output

    def test_truncates_to_last_3(self):
        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            messages=msgs,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        # Only last 3 messages shown
        assert "msg-7" in output
        assert "msg-8" in output
        assert "msg-9" in output
        # Earlier messages NOT shown
        assert "msg-0" not in output

    def test_empty_messages(self):
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            messages=[],
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        # Should not crash
        output = _capture_console(dbg)
        assert "LLM Request" in output

    def test_none_messages(self):
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            messages=None,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)
        assert "LLM Request" in output


# ── _render_response ─────────────────────────────────────────────────


class TestRenderResponse:
    def test_renders_openai_response(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            response={
                "choices": [{"message": {"content": "The answer is 4."}}]
            },
            input_tokens=100,
            output_tokens=25,
            cost_usd=0.0005,
            duration_ms=350.0,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        assert "The answer is 4." in output
        assert "350ms" in output
        assert "in=100" in output
        assert "out=25" in output
        assert "$0.0005" in output

    def test_renders_anthropic_response(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="anthropic",
            response={"content": [{"type": "text", "text": "From Claude"}]},
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)
        assert "From Claude" in output

    def test_none_response(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            response=None,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)
        assert "LLM Response" in output


# ── _render_tool_call ────────────────────────────────────────────────


class TestRenderToolCall:
    def test_renders_tool_name_and_input(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="search_files",
            tool_input={"query": "python", "limit": 10},
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        assert "search_files" in output
        assert "python" in output

    def test_empty_input(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="list_all",
            tool_input=None,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)
        assert "list_all" in output


# ── _render_tool_result ──────────────────────────────────────────────


class TestRenderToolResult:
    def test_renders_output(self):
        event = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_name="search",
            tool_output="Found 3 results",
            duration_ms=50.0,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        assert "search" in output
        assert "Found 3 results" in output

    def test_none_output(self):
        event = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_name="delete",
            tool_output=None,
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)
        assert "(empty)" in output


# ── _render_decision ─────────────────────────────────────────────────


class TestRenderDecision:
    def test_renders_decision_and_reasoning(self):
        event = TraceEvent(
            event_type=EventType.AGENT_DECISION,
            decision="delegate_to_search",
            reasoning="User asked a factual question about Python.",
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        assert "delegate_to_search" in output
        assert "factual question" in output

    def test_no_reasoning(self):
        event = TraceEvent(
            event_type=EventType.AGENT_DECISION,
            decision="skip",
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)
        assert "skip" in output


# ── _render_error ────────────────────────────────────────────────────


class TestRenderError:
    def test_renders_error(self):
        event = TraceEvent(
            event_type=EventType.ERROR,
            error_type="ValueError",
            error_message="Something went wrong with the API call",
        )
        trace = _make_trace(event)
        dbg = TraceDebugger(trace)
        output = _capture_console(dbg)

        assert "ValueError" in output
        assert "Something went wrong" in output


# ── _trace_summary ───────────────────────────────────────────────────


class TestTraceSummary:
    def test_summary_contains_all_stats(self):
        trace = _make_trace(
            TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.002,
                duration_ms=500.0,
            ),
            TraceEvent(
                event_type=EventType.TOOL_RESULT,
                tool_name="search",
                duration_ms=100.0,
            ),
        )
        dbg = TraceDebugger(trace)
        summary = dbg._trace_summary()

        assert trace.trace_id in summary
        assert "LLM: 1" in summary
        assert "Tools: 1" in summary
        assert "150" in summary  # total tokens
        assert "$0.0020" in summary
        assert "600" in summary  # total duration ms

    def test_summary_with_comparison(self):
        trace = _make_trace()
        compare = _make_trace()
        dbg = TraceDebugger(trace, compare_trace=compare)
        summary = dbg._trace_summary()
        assert "Comparing against" in summary


# ── Comparison rendering ─────────────────────────────────────────────


class TestComparisonRendering:
    def test_mismatched_event_type(self):
        event1 = TraceEvent(event_type=EventType.LLM_RESPONSE, provider="openai")
        event2 = TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search")
        trace1 = _make_trace(event1)
        trace2 = _make_trace(event2)

        dbg = TraceDebugger(trace1, compare_trace=trace2)
        output = _capture_console(dbg)
        assert "Reference has" in output

    def test_mismatched_model(self):
        event1 = TraceEvent(event_type=EventType.LLM_RESPONSE, provider="openai", model="gpt-4o")
        event2 = TraceEvent(event_type=EventType.LLM_RESPONSE, provider="openai", model="gpt-3.5-turbo")
        trace1 = _make_trace(event1)
        trace2 = _make_trace(event2)

        dbg = TraceDebugger(trace1, compare_trace=trace2)
        output = _capture_console(dbg)
        assert "Reference used model" in output


# ── Navigation ───────────────────────────────────────────────────────


class TestDebuggerNavigation:
    def test_step_and_goto(self):
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, provider="openai"),
            TraceEvent(event_type=EventType.LLM_RESPONSE, provider="openai"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
        ]
        trace = _make_trace(*events)
        dbg = TraceDebugger(trace)

        assert dbg._index == 0
        dbg._step(1)
        assert dbg._index == 1
        dbg._step(1)
        assert dbg._index == 2
        dbg._step(1)  # boundary
        assert dbg._index == 2

        dbg._goto("g 1")
        assert dbg._index == 0

    def test_empty_trace(self):
        trace = _make_trace()
        dbg = TraceDebugger(trace)
        buf = StringIO()
        dbg.console = Console(file=buf, force_terminal=True, width=120)
        dbg.run()
        assert "No events" in buf.getvalue()

    def test_event_filter(self):
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, provider="openai"),
            TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"),
            TraceEvent(event_type=EventType.LLM_RESPONSE, provider="openai"),
        ]
        trace = _make_trace(*events)
        dbg = TraceDebugger(trace, event_filter={EventType.TOOL_CALL})
        assert len(dbg._events) == 1
        assert dbg._events[0].event_type == EventType.TOOL_CALL
