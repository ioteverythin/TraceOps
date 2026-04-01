"""Tests for the trace data model."""

from trace_ops._types import EventType, Trace, TraceEvent


class TestTraceEvent:
    def test_create_llm_event(self):
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert event.provider == "openai"
        assert event.model == "gpt-4o"

    def test_to_dict_drops_none(self):
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
        )
        d = event.to_dict()
        assert "model" not in d  # None fields omitted
        assert d["provider"] == "openai"

    def test_roundtrip(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="search",
            tool_input={"query": "test"},
        )
        d = event.to_dict()
        restored = TraceEvent.from_dict(d)
        assert restored.event_type == EventType.TOOL_CALL
        assert restored.tool_name == "search"
        assert restored.tool_input == {"query": "test"}


class TestTrace:
    def test_empty_trace(self):
        trace = Trace()
        assert trace.total_llm_calls == 0
        assert trace.total_tool_calls == 0
        assert trace.trajectory == []

    def test_add_events(self):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="search",
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_name="search",
            tool_output="found 3 results",
        ))
        trace.finalize()

        assert trace.total_llm_calls == 1
        assert trace.total_tool_calls == 1
        assert trace.total_tokens == 150

    def test_trajectory(self):
        trace = Trace()
        trace.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        trace.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))
        trace.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))

        assert trace.trajectory == [
            "llm_call:gpt-4o",
            "tool:search",
            "llm_call:gpt-4o",
        ]

    def test_fingerprint_deterministic(self):
        trace = Trace()
        trace.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        trace.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))

        fp1 = trace.fingerprint()
        fp2 = trace.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_different_trajectories_different_fingerprints(self):
        t1 = Trace()
        t1.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))

        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="claude-4"))

        assert t1.fingerprint() != t2.fingerprint()

    def test_roundtrip(self):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            response={"choices": [{"message": {"content": "Hi!"}}]},
            input_tokens=10,
            output_tokens=5,
        ))
        trace.finalize()

        d = trace.to_dict()
        restored = Trace.from_dict(d)

        assert restored.total_llm_calls == 1
        assert len(restored.events) == 2
        assert restored.events[0].provider == "openai"
        assert restored.fingerprint() == trace.fingerprint()
