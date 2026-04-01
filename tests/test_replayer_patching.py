"""Tests for the Replayer SDK monkey-patching machinery.

Validates that the Replayer correctly patches OpenAI and Anthropic to
return recorded responses from cassettes, handles streaming replay,
mismatch detection, and restores originals on exit.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from trace_ops._types import EventType, Trace, TraceEvent, TraceMetadata
from trace_ops.cassette import CassetteMismatchError, save_cassette
from trace_ops.replayer import (
    Replayer,
    _dict_to_anthropic_response,
    _dict_to_openai_response,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _build_cassette(
    tmp_path,
    *,
    responses: list[dict[str, Any]],
    provider: str = "openai",
    model: str = "gpt-4o",
) -> str:
    """Create a cassette file with the given response events."""
    trace = Trace(metadata=TraceMetadata(description="test"))
    for i, resp in enumerate(responses):
        # Add a request event
        trace.events.append(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": f"msg-{i}"}],
        ))
        # Add a response event
        trace.events.append(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider=provider,
            model=model,
            response=resp,
            input_tokens=10,
            output_tokens=5,
        ))
    trace.finalize()
    path = str(tmp_path / "test_cassette.yaml")
    save_cassette(trace, path)
    return path


OPENAI_RESPONSE_DICT = {
    "id": "chatcmpl-abc",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The answer is 4."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}

ANTHROPIC_RESPONSE_DICT = {
    "id": "msg-abc",
    "model": "claude-3-5-sonnet",
    "content": [{"type": "text", "text": "Hello from Anthropic!"}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "stop_reason": "end_turn",
    "role": "assistant",
}


# ── _dict_to_openai_response tests ──────────────────────────────────


class TestDictToOpenAIResponse:
    """Test conversion from dict to OpenAI-like namespace."""

    def test_basic_response(self):
        result = _dict_to_openai_response(OPENAI_RESPONSE_DICT)
        assert result.id == "chatcmpl-abc"
        assert result.model == "gpt-4o"
        assert result.choices[0].message.content == "The answer is 4."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10

    def test_nested_access(self):
        data = {"a": {"b": {"c": 42}}}
        result = _dict_to_openai_response(data)
        assert result.a.b.c == 42

    def test_list_in_response(self):
        data = {"items": [{"name": "first"}, {"name": "second"}]}
        result = _dict_to_openai_response(data)
        assert result.items[0].name == "first"
        assert result.items[1].name == "second"

    def test_empty_dict(self):
        result = _dict_to_openai_response({})
        # Should not crash
        assert result is not None


# ── _dict_to_anthropic_response tests ────────────────────────────────


class TestDictToAnthropicResponse:
    """Test conversion from dict to Anthropic-like namespace."""

    def test_basic_response(self):
        result = _dict_to_anthropic_response(ANTHROPIC_RESPONSE_DICT)
        assert result.id == "msg-abc"
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello from Anthropic!"
        assert result.usage.input_tokens == 10

    def test_empty_dict(self):
        result = _dict_to_anthropic_response({})
        assert result is not None


# ── Replayer OpenAI integration ──────────────────────────────────────


class TestReplayerOpenAISync:
    """Test that patched OpenAI.create returns recorded responses."""

    def test_returns_recorded_response(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            with Replayer(path):
                result = Completions.create(
                    MagicMock(),
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "2+2?"}],
                )
                assert result.choices[0].message.content == "The answer is 4."
        finally:
            Completions.create = original  # type: ignore[assignment]

    def test_multiple_calls_in_order(self, tmp_path):
        resp1 = {**OPENAI_RESPONSE_DICT, "choices": [{"index": 0, "message": {"role": "assistant", "content": "First"}, "finish_reason": "stop"}]}
        resp2 = {**OPENAI_RESPONSE_DICT, "choices": [{"index": 0, "message": {"role": "assistant", "content": "Second"}, "finish_reason": "stop"}]}
        path = _build_cassette(tmp_path, responses=[resp1, resp2])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            with Replayer(path):
                r1 = Completions.create(MagicMock(), model="gpt-4o", messages=[])
                r2 = Completions.create(MagicMock(), model="gpt-4o", messages=[])
                assert r1.choices[0].message.content == "First"
                assert r2.choices[0].message.content == "Second"
        finally:
            Completions.create = original  # type: ignore[assignment]

    def test_too_many_calls_strict_raises(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            with pytest.raises(CassetteMismatchError, match="more LLM calls"), Replayer(path, strict=True):
                Completions.create(MagicMock(), model="gpt-4o", messages=[])
                Completions.create(MagicMock(), model="gpt-4o", messages=[])  # extra
        finally:
            Completions.create = original  # type: ignore[assignment]

    def test_too_few_calls_strict_raises(self, tmp_path):
        resp1 = OPENAI_RESPONSE_DICT
        resp2 = OPENAI_RESPONSE_DICT
        path = _build_cassette(tmp_path, responses=[resp1, resp2])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            with pytest.raises(CassetteMismatchError, match="fewer LLM calls"), Replayer(path, strict=True):
                Completions.create(MagicMock(), model="gpt-4o", messages=[])
                # Only 1 of 2 expected calls
        finally:
            Completions.create = original  # type: ignore[assignment]

    def test_provider_mismatch_strict_raises(self, tmp_path):
        path = _build_cassette(
            tmp_path, responses=[ANTHROPIC_RESPONSE_DICT], provider="anthropic", model="claude-3-5-sonnet"
        )

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            with pytest.raises(CassetteMismatchError, match="expected provider"), Replayer(path, strict=True):
                Completions.create(MagicMock(), model="gpt-4o", messages=[])
        finally:
            Completions.create = original  # type: ignore[assignment]

    def test_patches_restored_after_exit(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        with Replayer(path, strict=False):
            assert Completions.create is not original

        assert Completions.create is original

    def test_non_strict_allows_extra_calls(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            # Should NOT raise even though we make 2 calls for 1 recording
            with Replayer(path, strict=False):
                Completions.create(MagicMock(), model="gpt-4o", messages=[])
                Completions.create(MagicMock(), model="gpt-4o", messages=[])
                # Non-strict returns empty dict when exhausted
        finally:
            Completions.create = original  # type: ignore[assignment]


# ── Replayer Anthropic integration ───────────────────────────────────


class TestReplayerAnthropicSync:
    """Test that patched Anthropic Messages.create returns recorded responses."""

    def test_returns_recorded_response(self, tmp_path):
        path = _build_cassette(
            tmp_path,
            responses=[ANTHROPIC_RESPONSE_DICT],
            provider="anthropic",
            model="claude-3-5-sonnet",
        )

        from anthropic.resources.messages import Messages

        original = Messages.create

        try:
            with Replayer(path):
                result = Messages.create(
                    MagicMock(),
                    model="claude-3-5-sonnet",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1024,
                )
                assert result.content[0].text == "Hello from Anthropic!"
        finally:
            Messages.create = original  # type: ignore[assignment]

    def test_patches_restored(self, tmp_path):
        path = _build_cassette(
            tmp_path,
            responses=[ANTHROPIC_RESPONSE_DICT],
            provider="anthropic",
            model="claude-3-5-sonnet",
        )

        from anthropic.resources.messages import Messages

        original = Messages.create

        with Replayer(path, strict=False):
            assert Messages.create is not original

        assert Messages.create is original


# ── Replayer async tests ─────────────────────────────────────────────


class TestReplayerAsync:
    """Test async context manager and async OpenAI patching."""

    @pytest.mark.asyncio
    async def test_async_enter_exit(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        r = Replayer(path, strict=False)
        await r.__aenter__()
        assert Completions.create is not original
        await r.__aexit__(None, None, None)
        assert Completions.create is original

    @pytest.mark.asyncio
    async def test_async_too_few_calls_raises(self, tmp_path):
        resp1 = OPENAI_RESPONSE_DICT
        resp2 = OPENAI_RESPONSE_DICT
        path = _build_cassette(tmp_path, responses=[resp1, resp2])

        from openai.resources.chat.completions import AsyncCompletions

        original = AsyncCompletions.create

        try:
            with pytest.raises(CassetteMismatchError, match="fewer LLM calls"):
                async with Replayer(path, strict=True):
                    await AsyncCompletions.create(MagicMock(), model="gpt-4o", messages=[])
                    # only 1 of 2
        finally:
            AsyncCompletions.create = original  # type: ignore[assignment]


# ── Replayer streaming replay ────────────────────────────────────────


class TestReplayerStreaming:
    """Test that stream=True returns StreamReplay objects."""

    def test_stream_replay_returned(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        try:
            with Replayer(path, strict=False):
                result = Completions.create(
                    MagicMock(), model="gpt-4o", messages=[], stream=True
                )
                # Should be a StreamReplay, not a SimpleNamespace
                from trace_ops.streaming import StreamReplay
                assert isinstance(result, StreamReplay)
        finally:
            Completions.create = original  # type: ignore[assignment]


# ── Replayer decorator ───────────────────────────────────────────────


class TestReplayerDecorator:
    """Test the @Replayer.replay decorator."""

    def test_decorator(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        from openai.resources.chat.completions import Completions

        original = Completions.create

        @Replayer.replay(path, strict=False)
        def my_function():
            result = Completions.create(
                MagicMock(), model="gpt-4o", messages=[]
            )
            return result.choices[0].message.content

        try:
            content = my_function()
            assert content == "The answer is 4."
        finally:
            Completions.create = original  # type: ignore[assignment]


# ── Replayer load / queue tests ──────────────────────────────────────


class TestReplayerLoad:
    """Test cassette loading and response queue construction."""

    def test_load_builds_queue(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT, OPENAI_RESPONSE_DICT])

        r = Replayer(path)
        r._load()

        assert r.recorded_trace is not None
        assert len(r._response_queue) == 2
        assert r._call_index == 0

    def test_get_next_response_advances_index(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        r = Replayer(path)
        r._load()

        resp = r._get_next_response("openai", "gpt-4o")
        assert resp == OPENAI_RESPONSE_DICT
        assert r._call_index == 1

    def test_get_next_response_exhausted_strict(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        r = Replayer(path)
        r._load()
        r._get_next_response("openai", "gpt-4o")  # consume the one response

        with pytest.raises(CassetteMismatchError, match="more LLM calls"):
            r._get_next_response("openai", "gpt-4o")

    def test_get_next_response_exhausted_non_strict(self, tmp_path):
        path = _build_cassette(tmp_path, responses=[OPENAI_RESPONSE_DICT])

        r = Replayer(path, strict=False)
        r._load()
        r._get_next_response("openai", "gpt-4o")  # consume

        resp = r._get_next_response("openai", "gpt-4o")
        assert resp == {}  # returns empty dict in non-strict
