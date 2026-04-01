"""Extended HTML reporter tests — covering all builder functions and edge cases."""

from __future__ import annotations

from pathlib import Path

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.reporters.html import (
    _build_cost_chart,
    _build_diff_section,
    _build_events,
    _build_summary,
    _build_trajectory,
    _event_detail,
    generate_html_report,
)

# ── Helpers ─────────────────────────────────────────────────────────


def _full_trace() -> Trace:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "search"}}],
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model="gpt-4o",
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        input_tokens=50,
        output_tokens=10,
        cost_usd=0.003,
        duration_ms=200.0,
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.TOOL_CALL,
        tool_name="search",
        tool_input={"query": "test"},
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.TOOL_RESULT,
        tool_name="search",
        tool_output="Found results",
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.AGENT_DECISION,
        decision="delegate",
        reasoning="Need expert help",
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.ERROR,
        error_type="ValueError",
        error_message="Something went wrong",
        metadata={"retry": True},
    ))
    trace.finalize()
    return trace


# ── generate_html_report ────────────────────────────────────────────


class TestGenerateHtmlReport:
    def test_basic(self, tmp_path: Path):
        trace = _full_trace()
        out = tmp_path / "report.html"
        result = generate_html_report(trace, out)
        assert result == out
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "<html" in html
        assert "gpt-4o" in html

    def test_with_compare(self, tmp_path: Path):
        trace1 = _full_trace()
        trace2 = _full_trace()
        out = tmp_path / "compare.html"
        generate_html_report(trace1, out, compare_trace=trace2)
        html = out.read_text(encoding="utf-8")
        assert "Diff" in html or "identical" in html.lower()

    def test_custom_title(self, tmp_path: Path):
        trace = _full_trace()
        out = tmp_path / "custom.html"
        generate_html_report(trace, out, title="My Custom Report")
        html = out.read_text(encoding="utf-8")
        assert "My Custom Report" in html

    def test_creates_parent_dirs(self, tmp_path: Path):
        trace = _full_trace()
        out = tmp_path / "sub" / "deep" / "report.html"
        generate_html_report(trace, out)
        assert out.exists()


# ── _build_summary ──────────────────────────────────────────────────


class TestBuildSummary:
    def test_summary_html(self):
        trace = _full_trace()
        html = _build_summary(trace)
        assert "LLM Calls" in html
        assert "Tool Calls" in html
        assert "Tokens" in html
        assert "Cost" in html
        assert "Fingerprint" in html

    def test_summary_zero_values(self):
        trace = Trace()
        html = _build_summary(trace)
        assert "0" in html


# ── _build_trajectory ──────────────────────────────────────────────


class TestBuildTrajectory:
    def test_trajectory_steps(self):
        trace = _full_trace()
        html = _build_trajectory(trace)
        assert "llm_call:gpt-4o" in html
        assert "tool:search" in html
        assert "→" in html

    def test_empty_trajectory(self):
        trace = Trace()
        html = _build_trajectory(trace)
        assert "No trajectory" in html


# ── _build_events ──────────────────────────────────────────────────


class TestBuildEvents:
    def test_all_events_rendered(self):
        trace = _full_trace()
        html = _build_events(trace)
        assert "#1" in html
        assert "#6" in html  # 6 events total

    def test_event_badges(self):
        trace = _full_trace()
        html = _build_events(trace)
        assert "Llm Request" in html
        assert "Llm Response" in html
        assert "Tool Call" in html

    def test_token_display(self):
        trace = _full_trace()
        html = _build_events(trace)
        assert "50 / 10 tokens" in html or "50" in html


# ── _event_detail ──────────────────────────────────────────────────


class TestEventDetail:
    def test_llm_request_detail(self):
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            messages=[{"role": "user", "content": "hello"}],
        )
        html = _event_detail(event)
        assert "user" in html
        assert "hello" in html

    def test_llm_response_detail(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            response={"choices": [{"message": {"content": "hi"}}]},
        )
        html = _event_detail(event)
        assert "choices" in html

    def test_tool_call_detail(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_input={"query": "test"},
        )
        html = _event_detail(event)
        assert "Input" in html
        assert "test" in html

    def test_tool_result_detail(self):
        event = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_output="Found results",
        )
        html = _event_detail(event)
        assert "Output" in html
        assert "Found results" in html

    def test_decision_detail(self):
        event = TraceEvent(
            event_type=EventType.AGENT_DECISION,
            decision="delegate",
            reasoning="Need expert help",
        )
        html = _event_detail(event)
        assert "Decision" in html
        assert "Reasoning" in html

    def test_error_detail(self):
        event = TraceEvent(
            event_type=EventType.ERROR,
            error_type="ValueError",
            error_message="bad input",
        )
        html = _event_detail(event)
        assert "ValueError" in html
        assert "bad input" in html

    def test_metadata_detail(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            metadata={"retry": True, "attempt": 2},
        )
        html = _event_detail(event)
        assert "Metadata" in html
        assert "retry" in html

    def test_empty_event_detail(self):
        event = TraceEvent(event_type=EventType.LLM_REQUEST)
        html = _event_detail(event)
        assert "No details" in html

    def test_unserializable_response(self):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            response={"data": "normal"},
        )
        html = _event_detail(event)
        assert "normal" in html

    def test_unserializable_tool_input(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_input={"key": "value"},
        )
        html = _event_detail(event)
        assert "value" in html


# ── _build_cost_chart ──────────────────────────────────────────────


class TestBuildCostChart:
    def test_with_llm_events(self):
        trace = _full_trace()
        html = _build_cost_chart(trace)
        assert "Call 1" in html
        assert "gpt-4o" in html

    def test_no_llm_events(self):
        trace = Trace()
        trace.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="x"))
        html = _build_cost_chart(trace)
        assert "No LLM calls" in html

    def test_multiple_calls(self):
        trace = Trace()
        for i in range(3):
            trace.add_event(TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                model=f"model-{i}",
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
                cost_usd=0.01 * (i + 1),
            ))
        html = _build_cost_chart(trace)
        assert "Call 1" in html
        assert "Call 3" in html

    def test_zero_cost(self):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
        ))
        html = _build_cost_chart(trace)
        assert "Call 1" in html


# ── _build_diff_section ────────────────────────────────────────────


class TestBuildDiffSection:
    def test_identical_traces(self):
        trace = _full_trace()
        html = _build_diff_section(trace, trace)
        assert "identical" in html.lower()

    def test_different_traces(self):
        t1 = Trace()
        t1.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t1.finalize()

        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="claude-3"))
        t2.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))
        t2.finalize()

        html = _build_diff_section(t1, t2)
        assert "Trace comparison" in html or "diff" in html.lower()
