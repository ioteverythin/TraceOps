"""Tests for the Recorder."""

from trace_ops._types import EventType
from trace_ops.recorder import Recorder


class TestRecorderManual:
    """Test manual event recording (no SDK patching)."""

    def test_record_tool_call(self):
        with Recorder() as rec:
            rec.record_tool_call(
                tool_name="search",
                tool_input={"query": "test"},
                tool_output="found 3 results",
                duration_ms=150.0,
            )

        trace = rec.trace
        assert len(trace.events) == 2
        assert trace.events[0].event_type == EventType.TOOL_CALL
        assert trace.events[0].tool_name == "search"
        assert trace.events[1].event_type == EventType.TOOL_RESULT
        assert trace.events[1].tool_output == "found 3 results"

    def test_record_decision(self):
        with Recorder() as rec:
            rec.record_decision(
                decision="delegate_to_search_agent",
                reasoning="User asked a factual question",
            )

        trace = rec.trace
        assert len(trace.events) == 1
        assert trace.events[0].event_type == EventType.AGENT_DECISION
        assert trace.events[0].decision == "delegate_to_search_agent"

    def test_trace_finalized_on_exit(self):
        with Recorder() as rec:
            rec.record_tool_call("a", {}, "result_a")
            rec.record_tool_call("b", {}, "result_b")

        trace = rec.trace
        assert trace.total_tool_calls == 2

    def test_save_to_file(self, tmp_path):
        path = str(tmp_path / "test.yaml")
        with Recorder(save_to=path) as rec:
            rec.record_tool_call("search", {"q": "test"}, "ok")

        assert (tmp_path / "test.yaml").exists()

    def test_metadata(self):
        with Recorder(description="math test", tags=["ci", "regression"]) as rec:
            pass

        assert rec.trace.metadata.description == "math test"
        assert "ci" in rec.trace.metadata.tags

    def test_multiple_tool_calls(self):
        with Recorder() as rec:
            rec.record_tool_call("search", {"q": "a"}, "result_a")
            rec.record_decision("need_more_info")
            rec.record_tool_call("search", {"q": "b"}, "result_b")

        trace = rec.trace
        assert len(trace.events) == 5  # 2+1+2
        assert trace.trajectory == [
            "tool:search",
            "decision:need_more_info",
            "tool:search",
        ]


class TestRecorderDecorator:
    def test_decorator_records(self, tmp_path):
        path = str(tmp_path / "decorated.yaml")

        @Recorder.record(path)
        def my_agent():
            pass  # In real usage, LLM calls here would be intercepted

        my_agent()
        assert (tmp_path / "decorated.yaml").exists()
