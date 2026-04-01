"""Comprehensive tests for _types.py serialization.

Covers TraceEvent.to_dict/from_dict, TraceMetadata.to_dict/from_dict,
Trace.to_dict/from_dict, computed properties (llm_events, tool_events,
trajectory), finalize(), _update_stats(), and fingerprint().
"""

from __future__ import annotations

import pytest

from trace_ops._types import EventType, Trace, TraceEvent, TraceMetadata

# ── TraceEvent serialization ─────────────────────────────────────────


class TestTraceEventToDict:
    """Test TraceEvent.to_dict with various field combinations."""

    def test_minimal_event(self):
        e = TraceEvent(event_type=EventType.LLM_REQUEST)
        d = e.to_dict()
        assert d["event_type"] == "llm_request"
        assert "event_id" in d
        assert "timestamp" in d
        # None fields should NOT appear
        assert "provider" not in d
        assert "model" not in d
        assert "messages" not in d

    def test_all_fields_populated(self):
        e = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            response={"choices": [{"message": {"content": "hello"}}]},
            temperature=0.7,
            max_tokens=100,
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_name="search",
            tool_input={"q": "test"},
            tool_output="3 results",
            decision="proceed",
            reasoning="looks good",
            error_type="ValueError",
            error_message="bad input",
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.001,
            duration_ms=200.0,
            metadata={"custom": "data"},
        )
        d = e.to_dict()
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["temperature"] == 0.7
        assert d["max_tokens"] == 100
        assert d["tool_name"] == "search"
        assert d["tool_output"] == "3 results"
        assert d["decision"] == "proceed"
        assert d["reasoning"] == "looks good"
        assert d["error_type"] == "ValueError"
        assert d["input_tokens"] == 50
        assert d["output_tokens"] == 25
        assert d["cost_usd"] == 0.001
        assert d["duration_ms"] == 200.0
        assert d["metadata"] == {"custom": "data"}

    def test_empty_metadata_omitted(self):
        e = TraceEvent(event_type=EventType.ERROR, metadata={})
        d = e.to_dict()
        assert "metadata" not in d

    def test_zero_values_included(self):
        """Zero is not None — should be serialized."""
        e = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            duration_ms=0.0,
        )
        d = e.to_dict()
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0
        assert d["cost_usd"] == 0.0
        assert d["duration_ms"] == 0.0


class TestTraceEventFromDict:
    """Test TraceEvent.from_dict deserialization."""

    def test_roundtrip(self):
        original = TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="search",
            tool_input={"query": "python"},
            metadata={"source": "test"},
        )
        d = original.to_dict()
        restored = TraceEvent.from_dict(d)

        assert restored.event_type == EventType.TOOL_CALL
        assert restored.tool_name == "search"
        assert restored.tool_input == {"query": "python"}
        assert restored.metadata == {"source": "test"}

    def test_ignores_unknown_keys(self):
        d = {
            "event_type": "llm_request",
            "timestamp": 1234567890.0,
            "event_id": "abc123",
            "unknown_field": "should be ignored",
            "another_unknown": 42,
        }
        e = TraceEvent.from_dict(d)
        assert e.event_type == EventType.LLM_REQUEST
        assert e.event_id == "abc123"
        assert not hasattr(e, "unknown_field")

    def test_all_event_types(self):
        for et in EventType:
            d = {"event_type": et.value, "timestamp": 0.0, "event_id": "x"}
            e = TraceEvent.from_dict(d)
            assert e.event_type == et


# ── TraceMetadata serialization ──────────────────────────────────────


class TestTraceMetadataToDict:
    """Test TraceMetadata.to_dict with various field combinations."""

    def test_minimal(self):
        m = TraceMetadata()
        d = m.to_dict()
        assert "recorded_at" in d
        assert "trace_ops_version" in d
        # Optional fields omitted when empty
        assert "python_version" not in d
        assert "framework" not in d
        assert "description" not in d
        assert "tags" not in d
        assert "env" not in d

    def test_all_fields_populated(self):
        m = TraceMetadata(
            python_version="3.13.0",
            framework="langchain",
            description="integration test",
            tags=["ci", "nightly"],
            env={"CI": "true", "BRANCH": "main"},
        )
        d = m.to_dict()
        assert d["python_version"] == "3.13.0"
        assert d["framework"] == "langchain"
        assert d["description"] == "integration test"
        assert d["tags"] == ["ci", "nightly"]
        assert d["env"] == {"CI": "true", "BRANCH": "main"}


class TestTraceMetadataFromDict:
    def test_roundtrip(self):
        original = TraceMetadata(
            python_version="3.12.0",
            framework="crewai",
            description="test",
            tags=["fast"],
        )
        d = original.to_dict()
        restored = TraceMetadata.from_dict(d)
        assert restored.python_version == "3.12.0"
        assert restored.framework == "crewai"
        assert restored.tags == ["fast"]

    def test_from_empty_dict(self):
        m = TraceMetadata.from_dict({})
        assert m.trace_ops_version == "0.5.0"  # default

    def test_ignores_unknown_keys(self):
        d = {"recorded_at": 0.0, "unknown": "ignored"}
        m = TraceMetadata.from_dict(d)
        assert m.recorded_at == 0.0


# ── Trace serialization ─────────────────────────────────────────────


class TestTraceToDict:
    def test_complete_trace(self):
        t = Trace(metadata=TraceMetadata(description="full test"))
        t.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        ))
        t.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            response={"choices": []},
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            duration_ms=200.0,
        ))
        t.add_event(TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="search",
            tool_input={"q": "test"},
        ))
        t.add_event(TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_name="search",
            tool_output="found it",
            duration_ms=50.0,
        ))

        d = t.to_dict()
        assert d["version"] == "1"
        assert "trace_id" in d
        assert d["metadata"]["description"] == "full test"
        assert len(d["events"]) == 4
        assert "summary" in d
        assert d["summary"]["total_llm_calls"] == 1
        assert d["summary"]["total_tool_calls"] == 1
        assert "trajectory" in d["summary"]
        assert "fingerprint" in d["summary"]

    def test_empty_trace(self):
        t = Trace()
        d = t.to_dict()
        assert d["events"] == []
        assert d["summary"]["total_llm_calls"] == 0


class TestTraceFromDict:
    def test_roundtrip(self):
        original = Trace(metadata=TraceMetadata(description="roundtrip"))
        original.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST, provider="openai", model="gpt-4o",
        ))
        original.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE, provider="openai", model="gpt-4o",
            input_tokens=10, output_tokens=5, cost_usd=0.001,
        ))

        d = original.to_dict()
        restored = Trace.from_dict(d)

        assert restored.trace_id == original.trace_id
        assert len(restored.events) == 2
        assert restored.total_llm_calls == 1
        assert restored.metadata.description == "roundtrip"

    def test_from_empty_dict(self):
        t = Trace.from_dict({})
        assert len(t.events) == 0
        assert t.total_llm_calls == 0

    def test_missing_events_key(self):
        t = Trace.from_dict({"trace_id": "test123"})
        assert t.trace_id == "test123"
        assert len(t.events) == 0

    def test_missing_metadata_key(self):
        t = Trace.from_dict({"events": []})
        assert t.metadata is not None


# ── Trace computed properties ────────────────────────────────────────


class TestTraceProperties:
    def test_llm_events(self):
        t = Trace()
        t.add_event(TraceEvent(event_type=EventType.LLM_REQUEST))
        t.add_event(TraceEvent(event_type=EventType.LLM_RESPONSE))
        t.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="x"))

        assert len(t.llm_events) == 2
        assert all(
            e.event_type in (EventType.LLM_REQUEST, EventType.LLM_RESPONSE)
            for e in t.llm_events
        )

    def test_tool_events(self):
        t = Trace()
        t.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="a"))
        t.add_event(TraceEvent(event_type=EventType.TOOL_RESULT, tool_name="a"))
        t.add_event(TraceEvent(event_type=EventType.LLM_REQUEST))

        assert len(t.tool_events) == 2

    def test_trajectory(self):
        t = Trace()
        t.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))
        t.add_event(TraceEvent(event_type=EventType.AGENT_DECISION, decision="continue"))
        t.add_event(TraceEvent(event_type=EventType.ERROR, error_type="Timeout"))

        assert t.trajectory == [
            "llm_call:gpt-4o",
            "tool:search",
            "decision:continue",
            "error:Timeout",
        ]

    def test_trajectory_unknown_fallbacks(self):
        t = Trace()
        t.add_event(TraceEvent(event_type=EventType.LLM_REQUEST))
        t.add_event(TraceEvent(event_type=EventType.TOOL_CALL))
        t.add_event(TraceEvent(event_type=EventType.AGENT_DECISION))
        t.add_event(TraceEvent(event_type=EventType.ERROR))

        assert t.trajectory == [
            "llm_call:unknown",
            "tool:unknown",
            "decision:unknown",
            "error:unknown",
        ]


# ── Trace._update_stats / finalize ──────────────────────────────────


class TestTraceStats:
    def test_incremental_stats(self):
        t = Trace()
        t.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            input_tokens=100, output_tokens=50, cost_usd=0.002, duration_ms=200.0,
        ))
        assert t.total_llm_calls == 1
        assert t.total_tokens == 150
        assert t.total_cost_usd == pytest.approx(0.002)
        assert t.total_duration_ms == pytest.approx(200.0)

        t.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            input_tokens=200, output_tokens=100, cost_usd=0.005, duration_ms=300.0,
        ))
        assert t.total_llm_calls == 2
        assert t.total_tokens == 450
        assert t.total_cost_usd == pytest.approx(0.007)

    def test_tool_result_increments_tool_count(self):
        t = Trace()
        t.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="a"))
        assert t.total_tool_calls == 0  # TOOL_CALL doesn't increment

        t.add_event(TraceEvent(event_type=EventType.TOOL_RESULT, tool_name="a"))
        assert t.total_tool_calls == 1

    def test_none_tokens_treated_as_zero(self):
        t = Trace()
        t.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            input_tokens=None, output_tokens=None, cost_usd=None,
        ))
        assert t.total_llm_calls == 1
        assert t.total_tokens == 0
        assert t.total_cost_usd == 0.0

    def test_finalize_recomputes(self):
        t = Trace()
        # Add events directly (bypassing _update_stats)
        t.events.append(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            input_tokens=100, output_tokens=50, cost_usd=0.002, duration_ms=200.0,
        ))
        t.events.append(TraceEvent(
            event_type=EventType.TOOL_RESULT, tool_name="search", duration_ms=50.0,
        ))

        # Stats are NOT updated yet
        assert t.total_llm_calls == 0

        t.finalize()

        assert t.total_llm_calls == 1
        assert t.total_tool_calls == 1
        assert t.total_tokens == 150
        assert t.total_cost_usd == pytest.approx(0.002)
        assert t.total_duration_ms == pytest.approx(250.0)


# ── Trace.fingerprint ────────────────────────────────────────────────


class TestTraceFingerprint:
    def test_deterministic(self):
        t = Trace()
        t.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))

        assert t.fingerprint() == t.fingerprint()

    def test_different_trajectories(self):
        t1 = Trace()
        t1.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))

        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="claude-3"))

        assert t1.fingerprint() != t2.fingerprint()

    def test_empty_trace_fingerprint(self):
        t = Trace()
        fp = t.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16
