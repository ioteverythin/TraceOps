"""Tests for HTML reporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.reporters.html import generate_html_report


# ── Helpers ─────────────────────────────────────────────────────────


def _make_trace(events: list[TraceEvent] | None = None) -> Trace:
    trace = Trace()
    for e in events or []:
        trace.add_event(e)
    trace.finalize()
    return trace


def _llm_event(content: str = "hello") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response={"choices": [{"message": {"content": content}}]},
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
    )


def _tool_event(name: str = "search") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        provider="openai",
        tool_name=name,
        tool_input={"q": "test"},
        tool_output="found it",
    )


# ── Tests ───────────────────────────────────────────────────────────


class TestGenerateHtmlReport:
    def test_creates_file(self, tmp_path: Path):
        trace = _make_trace([_llm_event(), _tool_event()])
        out = tmp_path / "report.html"
        generate_html_report(trace, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_contains_html_structure(self, tmp_path: Path):
        trace = _make_trace([_llm_event()])
        out = tmp_path / "report.html"
        generate_html_report(trace, str(out))
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html
        assert "TraceOps" in html

    def test_custom_title(self, tmp_path: Path):
        trace = _make_trace([_llm_event()])
        out = tmp_path / "report.html"
        generate_html_report(trace, str(out), title="Custom Title")
        html = out.read_text(encoding="utf-8")
        assert "Custom Title" in html

    def test_includes_events(self, tmp_path: Path):
        trace = _make_trace([
            _llm_event("The answer is 42"),
            _tool_event("calculator"),
        ])
        out = tmp_path / "report.html"
        generate_html_report(trace, str(out))
        html = out.read_text(encoding="utf-8")
        assert "gpt-4o" in html
        assert "calculator" in html

    def test_with_comparison_trace(self, tmp_path: Path):
        trace1 = _make_trace([_llm_event("first")])
        trace2 = _make_trace([_llm_event("second")])
        out = tmp_path / "report.html"
        generate_html_report(trace1, str(out), compare_trace=trace2)
        html = out.read_text(encoding="utf-8")
        assert out.exists()
        # Should contain diff section
        assert "diff" in html.lower() or "compare" in html.lower() or "Comparison" in html

    def test_empty_trace(self, tmp_path: Path):
        trace = _make_trace([])
        out = tmp_path / "report.html"
        generate_html_report(trace, str(out))
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "TraceOps" in html
