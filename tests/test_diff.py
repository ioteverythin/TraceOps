"""Tests for the trace diff engine."""

import pytest
from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.diff import diff_traces, assert_trace_unchanged


def _make_trace(*steps: tuple[str, str]) -> Trace:
    """Helper to build a trace from (event_type, name) tuples."""
    trace = Trace()
    for step_type, name in steps:
        if step_type == "llm":
            trace.add_event(TraceEvent(
                event_type=EventType.LLM_REQUEST, model=name,
            ))
            trace.add_event(TraceEvent(
                event_type=EventType.LLM_RESPONSE, model=name,
                response={"content": f"response from {name}"},
                input_tokens=100, output_tokens=50,
            ))
        elif step_type == "tool":
            trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL, tool_name=name,
            ))
            trace.add_event(TraceEvent(
                event_type=EventType.TOOL_RESULT, tool_name=name,
                tool_output=f"result from {name}",
            ))
    trace.finalize()
    return trace


class TestDiffTraces:
    def test_identical_traces(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        diff = diff_traces(t1, t2)
        assert not diff.has_changes

    def test_different_trajectory(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "read_file"))
        diff = diff_traces(t1, t2)
        assert diff.has_changes
        assert diff.trajectory_changed
        assert "search" in diff.removed_tools
        assert "read_file" in diff.added_tools

    def test_different_model(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "claude-4-sonnet"))
        diff = diff_traces(t1, t2)
        assert diff.has_changes
        assert len(diff.changed_models) == 1
        assert diff.changed_models[0]["old"] == "gpt-4o"
        assert diff.changed_models[0]["new"] == "claude-4-sonnet"

    def test_extra_llm_call(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"), ("llm", "gpt-4o"))
        diff = diff_traces(t1, t2)
        assert diff.has_changes
        assert diff.llm_calls_delta == 1

    def test_summary_readable(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "claude-4"), ("tool", "browse"))
        diff = diff_traces(t1, t2)
        summary = diff.summary()
        assert "TRAJECTORY CHANGED" in summary
        assert "search" in summary
        assert "browse" in summary


class TestAssertTraceUnchanged:
    def test_identical_passes(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"))
        assert_trace_unchanged(t1, t2)  # should not raise

    def test_changed_trajectory_fails(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "browse"))
        with pytest.raises(AssertionError, match="regression"):
            assert_trace_unchanged(t1, t2)

    def test_ignore_trajectory(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "browse"))
        # Should pass with trajectory ignored
        assert_trace_unchanged(t1, t2, ignore_trajectory=True)

    def test_changed_model_fails(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o-mini"))
        with pytest.raises(AssertionError, match="Model"):
            assert_trace_unchanged(t1, t2)
