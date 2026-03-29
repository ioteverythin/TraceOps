"""Integration tests for agent-replay ↔ Anthropic SDK.

Uses ``unittest.mock.patch`` to mock HTTP-level responses so the
Anthropic SDK thinks it got a real response from the API.  No API key
or network access needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trace_ops._types import EventType
from trace_ops.cassette import load_cassette
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer

# ── anthropic availability ──────────────────────────────────────────
anthropic = pytest.importorskip("anthropic", reason="anthropic not installed")
from anthropic import Anthropic
from anthropic.resources.messages import Messages


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_anthropic_response(
    content: str = "Hello!",
    model: str = "claude-3-5-sonnet-20241022",
    tool_use_blocks: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    """Build a fake ``Message``-like object the Anthropic SDK returns."""
    blocks = []
    if content:
        blocks.append(SimpleNamespace(type="text", text=content))
    if tool_use_blocks:
        blocks.extend(tool_use_blocks)

    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    resp = SimpleNamespace(
        id="msg-test123",
        type="message",
        role="assistant",
        model=model,
        content=blocks,
        stop_reason="end_turn" if not tool_use_blocks else "tool_use",
        usage=usage,
    )
    return resp


def _fake_tool_use_block(name: str, input_: dict, tool_use_id: str = "toolu_abc"):
    """Build a fake tool_use content block."""
    return SimpleNamespace(
        type="tool_use",
        id=tool_use_id,
        name=name,
        input=input_,
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def cassette_path(tmp_path) -> str:
    return str(tmp_path / "anthropic.yaml")


# ── Test: basic message ──────────────────────────────────────────────


class TestAnthropicRecordReplay:
    """Record + replay with mocked Anthropic message creation."""

    def test_record_captures_events(self, cassette_path: str) -> None:
        fake_resp = _fake_anthropic_response(content="Bonjour!")

        with patch.object(Messages, "create", return_value=fake_resp):
            client = Anthropic(api_key="sk-ant-fake")

            with Recorder(save_to=cassette_path) as rec:
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Salut!"}],
                )

        assert result.content[0].text == "Bonjour!"

        events = rec.trace.events
        req_events = [e for e in events if e.event_type == EventType.LLM_REQUEST]
        res_events = [e for e in events if e.event_type == EventType.LLM_RESPONSE]

        assert len(req_events) == 1
        assert req_events[0].provider == "anthropic"
        assert req_events[0].model == "claude-3-5-sonnet-20241022"
        assert len(res_events) == 1
        assert res_events[0].input_tokens == 10
        assert res_events[0].output_tokens == 5

    def test_replay_returns_recorded_response(self, cassette_path: str) -> None:
        fake_resp = _fake_anthropic_response(content="Correct answer.")

        # Record
        with patch.object(Messages, "create", return_value=fake_resp):
            client = Anthropic(api_key="sk-ant-fake")
            with Recorder(save_to=cassette_path):
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "question"}],
                )

        # Replay
        client2 = Anthropic(api_key="sk-ant-fake")
        with Replayer(cassette_path) as rep:
            result = client2.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "question"}],
            )

        # Replayed result is SimpleNamespace-like
        assert result.content[0].text == "Correct answer."


# ── Test: tool use ───────────────────────────────────────────────────


class TestAnthropicToolUse:
    """Record + replay with tool_use content blocks."""

    def test_tool_use_recording(self, cassette_path: str) -> None:
        tc = _fake_tool_use_block("get_weather", {"location": "NYC"})
        fake_resp = _fake_anthropic_response(content="", tool_use_blocks=[tc])

        with patch.object(Messages, "create", return_value=fake_resp):
            client = Anthropic(api_key="sk-ant-fake")
            with Recorder(save_to=cassette_path) as rec:
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Weather in NYC?"}],
                )

        tool_events = [
            e for e in rec.trace.events if e.event_type == EventType.TOOL_CALL
        ]
        assert len(tool_events) >= 1
        assert tool_events[0].tool_name == "get_weather"

    def test_multiple_tool_calls(self, cassette_path: str) -> None:
        tc1 = _fake_tool_use_block("search", {"query": "python"}, "call_1")
        tc2 = _fake_tool_use_block("search", {"query": "rust"}, "call_2")
        fake_resp = _fake_anthropic_response(content="", tool_use_blocks=[tc1, tc2])

        with patch.object(Messages, "create", return_value=fake_resp):
            client = Anthropic(api_key="sk-ant-fake")
            with Recorder(save_to=cassette_path) as rec:
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Compare"}],
                )

        tool_events = [
            e for e in rec.trace.events if e.event_type == EventType.TOOL_CALL
        ]
        assert len(tool_events) >= 2


# ── Test: cassette persistence ───────────────────────────────────────


class TestAnthropicCassette:
    """Verify cassette YAML roundtrip."""

    def test_cassette_roundtrip(self, cassette_path: str) -> None:
        fake_resp = _fake_anthropic_response(content="test")

        with patch.object(Messages, "create", return_value=fake_resp):
            client = Anthropic(api_key="sk-ant-fake")
            with Recorder(save_to=cassette_path):
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "test"}],
                )

        trace = load_cassette(cassette_path)
        providers = {e.provider for e in trace.events if e.provider}
        assert "anthropic" in providers


# ── Test: patch cleanup ─────────────────────────────────────────────


class TestAnthropicPatchCleanup:
    """Verify patches are cleaned up."""

    def test_recorder_cleans_up(self) -> None:
        original = Messages.create

        with Recorder():
            pass

        assert Messages.create is original

    def test_replayer_cleans_up(self, cassette_path: str) -> None:
        fake_resp = _fake_anthropic_response(content="x")

        with patch.object(Messages, "create", return_value=fake_resp):
            client = Anthropic(api_key="sk-ant-fake")
            with Recorder(save_to=cassette_path):
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "x"}],
                )

        original = Messages.create

        with Replayer(cassette_path, strict=False):
            pass

        assert Messages.create is original
