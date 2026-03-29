"""Tests for cassette storage (save/load)."""

import pytest
from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import (
    CassetteNotFoundError,
    load_cassette,
    save_cassette,
    cassette_path_for_test,
)


class TestSaveCassette:
    def test_saves_yaml_file(self, tmp_path):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
        ))
        trace.finalize()

        path = tmp_path / "test.yaml"
        result = save_cassette(trace, path)

        assert result == path
        assert path.exists()
        content = path.read_text()
        assert "openai" in content
        assert "gpt-4o" in content

    def test_creates_parent_dirs(self, tmp_path):
        trace = Trace()
        path = tmp_path / "deep" / "nested" / "test.yaml"
        save_cassette(trace, path)
        assert path.exists()

    def test_redacts_api_keys(self, tmp_path):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            metadata={"api_key": "sk-1234567890abcdef1234567890abcdef"},
        ))
        trace.finalize()

        path = tmp_path / "test.yaml"
        save_cassette(trace, path)

        content = path.read_text()
        assert "sk-1234567890abcdef1234567890abcdef" not in content
        assert "REDACTED" in content


class TestLoadCassette:
    def test_loads_saved_cassette(self, tmp_path):
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

        path = tmp_path / "test.yaml"
        save_cassette(trace, path)

        loaded = load_cassette(path)
        assert len(loaded.events) == 2
        assert loaded.events[0].provider == "openai"
        assert loaded.events[1].response == {"choices": [{"message": {"content": "Hi!"}}]}
        assert loaded.total_llm_calls == 1

    def test_missing_cassette_raises(self, tmp_path):
        with pytest.raises(CassetteNotFoundError):
            load_cassette(tmp_path / "nonexistent.yaml")


class TestCassettePath:
    def test_generates_correct_path(self):
        path = cassette_path_for_test(
            "/home/user/tests/test_agent.py",
            "test_math_addition",
        )
        assert path.name == "test_math_addition.yaml"
        assert "test_agent" in str(path)
        assert "cassettes" in str(path)

    def test_sanitizes_parametrize_names(self):
        path = cassette_path_for_test(
            "test_agent.py",
            "test_math[2+2=4]",
        )
        assert "[" not in path.name
        assert "]" not in path.name
