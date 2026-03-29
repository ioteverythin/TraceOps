"""Integration tests for agent-replay ↔ OpenAI SDK.

Uses ``unittest.mock.patch`` to mock HTTP-level responses so the
OpenAI SDK thinks it got a real response from the API.  No API key
or network access needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trace_ops._types import EventType
from trace_ops.cassette import load_cassette
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer

# ── openai availability ──────────────────────────────────────────────
openai = pytest.importorskip("openai", reason="openai not installed")
from openai import OpenAI
from openai.resources.chat.completions import Completions


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_chat_response(
    content: str = "Hello!",
    model: str = "gpt-4o-mini",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Build a fake ``ChatCompletion``-like object the OpenAI SDK returns."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        function_call=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason="stop" if not tool_calls else "tool_calls",
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    resp = SimpleNamespace(
        id="chatcmpl-test123",
        object="chat.completion",
        model=model,
        choices=[choice],
        usage=usage,
        created=1700000000,
    )
    return resp


def _fake_tool_call(name: str, arguments: dict, call_id: str = "call_abc"):
    """Build a fake tool call object."""
    fn = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return SimpleNamespace(id=call_id, type="function", function=fn)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def cassette_path(tmp_path: Path) -> str:
    return str(tmp_path / "openai.yaml")


# ── Test: basic chat completion ──────────────────────────────────────


class TestOpenAIRecordReplay:
    """Record + replay with mocked OpenAI chat completions."""

    def test_record_captures_events(self, cassette_path: str) -> None:
        fake_resp = _fake_chat_response(content="Paris is the capital.")

        with patch.object(
            Completions, "create", return_value=fake_resp
        ) as mock_create:
            client = OpenAI(api_key="sk-fake")

            with Recorder(save_to=cassette_path) as rec:
                result = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Capital of France?"}],
                )

        assert result.choices[0].message.content == "Paris is the capital."

        events = rec.trace.events
        req_events = [e for e in events if e.event_type == EventType.LLM_REQUEST]
        res_events = [e for e in events if e.event_type == EventType.LLM_RESPONSE]

        assert len(req_events) == 1
        assert req_events[0].provider == "openai"
        assert req_events[0].model == "gpt-4o-mini"
        assert len(res_events) == 1
        assert res_events[0].input_tokens == 10
        assert res_events[0].output_tokens == 5

    def test_replay_returns_recorded_response(self, cassette_path: str) -> None:
        fake_resp = _fake_chat_response(content="42 is the answer.")

        # Record
        with patch.object(Completions, "create", return_value=fake_resp):
            client = OpenAI(api_key="sk-fake")
            with Recorder(save_to=cassette_path):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "What is 6*7?"}],
                )

        # Replay — no mock needed, replayer intercepts
        client2 = OpenAI(api_key="sk-fake")
        with Replayer(cassette_path) as rep:
            result = client2.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What is 6*7?"}],
            )

        # Replayed result comes back as a SimpleNamespace-like object
        content = result.choices[0].message.content
        assert content == "42 is the answer."

    def test_multiple_calls(self, cassette_path: str) -> None:
        responses = [
            _fake_chat_response(content="Answer 1"),
            _fake_chat_response(content="Answer 2"),
        ]
        call_count = 0

        original_create = Completions.create

        def mock_create(self_inner, *args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch.object(Completions, "create", mock_create):
            client = OpenAI(api_key="sk-fake")
            with Recorder(save_to=cassette_path) as rec:
                r1 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Q1"}],
                )
                r2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Q2"}],
                )

        assert r1.choices[0].message.content == "Answer 1"
        assert r2.choices[0].message.content == "Answer 2"

        llm_responses = [
            e for e in rec.trace.events if e.event_type == EventType.LLM_RESPONSE
        ]
        assert len(llm_responses) == 2

        # Replay both
        client2 = OpenAI(api_key="sk-fake")
        with Replayer(cassette_path):
            rr1 = client2.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Q1"}],
            )
            rr2 = client2.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Q2"}],
            )

        assert rr1.choices[0].message.content == "Answer 1"
        assert rr2.choices[0].message.content == "Answer 2"


# ── Test: tool calls ────────────────────────────────────────────────


class TestOpenAIToolCalls:
    """Record + replay with tool calls in the response."""

    def test_tool_call_recording(self, cassette_path: str) -> None:
        tc = _fake_tool_call("get_weather", {"city": "NYC"})
        fake_resp = _fake_chat_response(content="", tool_calls=[tc])

        with patch.object(Completions, "create", return_value=fake_resp):
            client = OpenAI(api_key="sk-fake")
            with Recorder(save_to=cassette_path) as rec:
                result = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Weather?"}],
                )

        tool_events = [
            e for e in rec.trace.events if e.event_type == EventType.TOOL_CALL
        ]
        assert len(tool_events) >= 1
        assert tool_events[0].tool_name == "get_weather"


# ── Test: cassette persistence ───────────────────────────────────────


class TestOpenAICassette:
    """Verify cassette YAML is valid and contains expected data."""

    def test_cassette_roundtrip(self, cassette_path: str) -> None:
        fake_resp = _fake_chat_response(content="test response")

        with patch.object(Completions, "create", return_value=fake_resp):
            client = OpenAI(api_key="sk-fake")
            with Recorder(save_to=cassette_path):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                )

        trace = load_cassette(cassette_path)
        providers = {e.provider for e in trace.events if e.provider}
        assert "openai" in providers
        assert len(trace.events) >= 2  # request + response


# ── Test: patch cleanup ─────────────────────────────────────────────


class TestOpenAIPatchCleanup:
    """Verify monkey-patches are removed after context exit."""

    def test_recorder_cleans_up(self) -> None:
        original = Completions.create

        with Recorder() as rec:
            pass

        assert Completions.create is original

    def test_replayer_cleans_up(self, cassette_path: str) -> None:
        fake_resp = _fake_chat_response(content="x")

        with patch.object(Completions, "create", return_value=fake_resp):
            client = OpenAI(api_key="sk-fake")
            with Recorder(save_to=cassette_path):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "x"}],
                )

        original = Completions.create

        with Replayer(cassette_path, strict=False):
            pass

        assert Completions.create is original
