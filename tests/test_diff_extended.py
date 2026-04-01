"""Extended tests for diff.py — trace comparison and regression assertions.

Covers: TraceDiff.summary() branches, diff_traces() with response diffs,
assert_trace_unchanged() with various ignore flags, cost/timing checks.
"""

from __future__ import annotations

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.diff import TraceDiff, assert_trace_unchanged, diff_traces

# ── Helpers ─────────────────────────────────────────────────────────


def _make_trace(*steps: tuple[str, str], cost: float | None = None) -> Trace:
    """Build a trace from (event_type, name) tuples."""
    trace = Trace()
    for step_type, name in steps:
        if step_type == "llm":
            trace.add_event(TraceEvent(
                event_type=EventType.LLM_REQUEST, model=name,
            ))
            trace.add_event(TraceEvent(
                event_type=EventType.LLM_RESPONSE, model=name,
                provider="openai",
                response={
                    "choices": [{
                        "message": {"role": "assistant", "content": f"resp-{name}"},
                        "finish_reason": "stop",
                    }],
                    "model": name,
                },
                input_tokens=100, output_tokens=50,
                cost_usd=cost,
            ))
        elif step_type == "tool":
            trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL, tool_name=name,
            ))
            trace.add_event(TraceEvent(
                event_type=EventType.TOOL_RESULT, tool_name=name,
                tool_output=f"result-{name}",
            ))
        elif step_type == "decision":
            trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION, decision=name,
            ))
        elif step_type == "error":
            trace.add_event(TraceEvent(
                event_type=EventType.ERROR, error_type=name,
                error_message=f"Error: {name}",
            ))
    trace.finalize()
    return trace


# ── TraceDiff.summary() ────────────────────────────────────────────


class TestTraceDiffSummary:
    def test_no_changes(self):
        diff = TraceDiff()
        assert "No changes" in diff.summary()
        assert "identical" in diff.summary()

    def test_trajectory_changed(self):
        diff = TraceDiff(
            has_changes=True,
            trajectory_changed=True,
            old_trajectory=["llm_call:gpt-4o", "tool:search"],
            new_trajectory=["llm_call:gpt-4o", "tool:browse"],
        )
        s = diff.summary()
        assert "TRAJECTORY CHANGED" in s
        assert "search" in s
        assert "browse" in s

    def test_llm_calls_more(self):
        diff = TraceDiff(has_changes=True, llm_calls_delta=2)
        s = diff.summary()
        assert "2 more" in s
        assert "LLM calls" in s

    def test_llm_calls_fewer(self):
        diff = TraceDiff(has_changes=True, llm_calls_delta=-1)
        s = diff.summary()
        assert "1 fewer" in s

    def test_tool_calls_delta(self):
        diff = TraceDiff(has_changes=True, tool_calls_delta=3)
        assert "3 more" in diff.summary()

    def test_tool_calls_fewer(self):
        diff = TraceDiff(has_changes=True, tool_calls_delta=-2)
        assert "2 fewer" in diff.summary()

    def test_added_tools(self):
        diff = TraceDiff(has_changes=True, added_tools=["browse", "execute"])
        s = diff.summary()
        assert "browse" in s
        assert "execute" in s
        assert "New tools" in s

    def test_removed_tools(self):
        diff = TraceDiff(has_changes=True, removed_tools=["search"])
        s = diff.summary()
        assert "search" in s
        assert "no longer used" in s

    def test_changed_models(self):
        diff = TraceDiff(
            has_changes=True,
            changed_models=[{"index": 1, "old": "gpt-4o", "new": "gpt-4o-mini"}],
        )
        s = diff.summary()
        assert "gpt-4o" in s
        assert "gpt-4o-mini" in s
        assert "call #1" in s

    def test_token_delta(self):
        diff = TraceDiff(has_changes=True, token_delta=500)
        assert "500 more" in diff.summary()

    def test_token_delta_fewer(self):
        diff = TraceDiff(has_changes=True, token_delta=-200)
        assert "200 fewer" in diff.summary()

    def test_cost_delta(self):
        diff = TraceDiff(has_changes=True, cost_delta=0.05)
        s = diff.summary()
        assert "$0.05" in s
        assert "higher" in s

    def test_cost_delta_lower(self):
        diff = TraceDiff(has_changes=True, cost_delta=-0.02)
        s = diff.summary()
        assert "lower" in s

    def test_cost_delta_tiny_ignored(self):
        # cost_delta <= 0.001 is not shown
        diff = TraceDiff(has_changes=True, cost_delta=0.0005)
        s = diff.summary()
        assert "Cost" not in s

    def test_response_diffs(self):
        diff = TraceDiff(
            has_changes=True,
            response_diffs=[{"call_index": 1, "model": "gpt-4o", "diff": {}}],
        )
        s = diff.summary()
        assert "1 response(s) changed" in s


# ── diff_traces() ──────────────────────────────────────────────────


class TestDiffTraces:
    def test_identical(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"))
        diff = diff_traces(t1, t2)
        assert not diff.has_changes

    def test_different_fingerprint_but_same_trajectory(self):
        # Same trajectory but possibly different content
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"))
        diff = diff_traces(t1, t2)
        # Same trajectory → same fingerprint
        assert diff.old_fingerprint == diff.new_fingerprint

    def test_cost_delta(self):
        t1 = _make_trace(("llm", "gpt-4o"), cost=0.01)
        t2 = _make_trace(("llm", "gpt-4o"), cost=0.05)
        diff = diff_traces(t1, t2)
        assert diff.has_changes
        assert abs(diff.cost_delta - 0.04) < 0.001

    def test_token_delta(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t2.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE, model="gpt-4o", provider="openai",
            response={"choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]},
            input_tokens=500, output_tokens=250,
        ))
        t2.finalize()
        diff = diff_traces(t1, t2)
        assert diff.has_changes
        assert diff.token_delta != 0

    def test_response_content_diff(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t2.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE, model="gpt-4o", provider="openai",
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "Different!"},
                    "finish_reason": "stop",
                }],
                "model": "gpt-4o",
            },
            input_tokens=100, output_tokens=50,
        ))
        t2.finalize()
        diff = diff_traces(t1, t2)
        assert diff.has_changes
        assert len(diff.response_diffs) >= 1

    def test_added_and_removed_tools(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"), ("tool", "read"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "search"), ("tool", "write"))
        diff = diff_traces(t1, t2)
        assert "read" in diff.removed_tools
        assert "write" in diff.added_tools

    def test_model_change_detected(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"), ("llm", "claude-4"))
        diff = diff_traces(t1, t2)
        assert len(diff.changed_models) == 1
        assert diff.changed_models[0]["old"] == "gpt-4o"
        assert diff.changed_models[0]["new"] == "claude-4"
        assert diff.changed_models[0]["index"] == 2

    def test_empty_traces(self):
        t1 = Trace()
        t2 = Trace()
        t1.finalize()
        t2.finalize()
        diff = diff_traces(t1, t2)
        assert not diff.has_changes


# ── assert_trace_unchanged() ───────────────────────────────────────


class TestAssertTraceUnchanged:
    def test_identical_passes(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"))
        assert_trace_unchanged(t1, t2)

    def test_trajectory_fails(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "browse"))
        with pytest.raises(AssertionError, match="regression"):
            assert_trace_unchanged(t1, t2)

    def test_ignore_trajectory(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "browse"))
        assert_trace_unchanged(t1, t2, ignore_trajectory=True)

    def test_model_change_fails(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "claude-4"))
        with pytest.raises(AssertionError, match="Model"):
            assert_trace_unchanged(t1, t2)

    def test_response_content_change_fails(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t2.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE, model="gpt-4o", provider="openai",
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "different-response"},
                    "finish_reason": "stop",
                }],
                "model": "gpt-4o",
            },
            input_tokens=100, output_tokens=50,
        ))
        t2.finalize()
        with pytest.raises(AssertionError, match="response"):
            assert_trace_unchanged(t1, t2)

    def test_ignore_responses(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t2.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE, model="gpt-4o", provider="openai",
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "different-response"},
                    "finish_reason": "stop",
                }],
                "model": "gpt-4o",
            },
            input_tokens=100, output_tokens=50,
        ))
        t2.finalize()
        # Should pass when ignoring responses (and trajectory matches)
        assert_trace_unchanged(t1, t2, ignore_responses=True)

    def test_cost_change_ignored_by_default(self):
        t1 = _make_trace(("llm", "gpt-4o"), cost=0.01)
        t2 = _make_trace(("llm", "gpt-4o"), cost=0.99)
        # ignore_costs=True by default
        assert_trace_unchanged(t1, t2)

    def test_cost_change_detected_when_enabled(self):
        t1 = _make_trace(("llm", "gpt-4o"), cost=0.01)
        t2 = _make_trace(("llm", "gpt-4o"), cost=0.99)
        with pytest.raises(AssertionError, match="Cost"):
            assert_trace_unchanged(t1, t2, ignore_costs=False)

    def test_extra_llm_calls_fail(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"), ("llm", "gpt-4o"))
        with pytest.raises(AssertionError, match="more LLM"):
            assert_trace_unchanged(t1, t2)

    def test_added_tools_fail(self):
        t1 = _make_trace(("llm", "gpt-4o"))
        t2 = _make_trace(("llm", "gpt-4o"), ("tool", "new_tool"))
        with pytest.raises(AssertionError, match="regression"):
            assert_trace_unchanged(t1, t2)

    def test_removed_tools_fail(self):
        t1 = _make_trace(("llm", "gpt-4o"), ("tool", "search"))
        t2 = _make_trace(("llm", "gpt-4o"))
        with pytest.raises(AssertionError, match="regression"):
            assert_trace_unchanged(t1, t2)
