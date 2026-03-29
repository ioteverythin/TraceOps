"""Tests for the streaming interception module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from trace_ops.streaming import (
    AsyncStreamCapture,
    AsyncStreamReplay,
    StreamCapture,
    StreamReplay,
    _assemble_openai_chunks,
    _assemble_anthropic_chunks,
    _split_openai_response,
    _split_anthropic_response,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_openai_chunk(
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    model: str = "gpt-4o",
    chunk_id: str = "chatcmpl-abc",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _fake_stream(chunks: list[Any]):
    """Create a simple iterable from a list of chunks."""
    yield from chunks


class _FakeAsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


# ── StreamCapture tests ─────────────────────────────────────────────


class TestStreamCapture:
    def test_yields_all_chunks(self):
        raw_chunks = [
            _make_openai_chunk(role="assistant", content=""),
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(content=" world"),
            _make_openai_chunk(finish_reason="stop"),
        ]
        captured: list[Any] = []
        assembled_result: dict[str, Any] = {}

        def on_complete(assembled: dict[str, Any]) -> None:
            nonlocal assembled_result
            assembled_result = assembled

        stream = StreamCapture(_fake_stream(raw_chunks), "openai", on_complete)
        for chunk in stream:
            captured.append(chunk)

        assert len(captured) == 4
        assert assembled_result["choices"][0]["message"]["content"] == "Hello world"

    def test_callback_invoked_once(self):
        call_count = 0

        def on_complete(assembled: dict[str, Any]) -> None:
            nonlocal call_count
            call_count += 1

        stream = StreamCapture(
            _fake_stream([_make_openai_chunk(content="hi")]),
            "openai",
            on_complete,
        )
        list(stream)  # consume
        assert call_count == 1

    def test_context_manager(self):
        assembled_result: dict[str, Any] = {}

        def on_complete(assembled: dict[str, Any]) -> None:
            nonlocal assembled_result
            assembled_result = assembled

        raw_chunks = [_make_openai_chunk(content="test")]
        stream = StreamCapture(_fake_stream(raw_chunks), "openai", on_complete)
        with stream:
            pass  # don't iterate — __exit__ should still finish
        assert assembled_result is not None


# ── AsyncStreamCapture tests ────────────────────────────────────────


class TestAsyncStreamCapture:
    @pytest.mark.asyncio
    async def test_yields_all_chunks(self):
        raw_chunks = [
            _make_openai_chunk(role="assistant"),
            _make_openai_chunk(content="async hello"),
            _make_openai_chunk(finish_reason="stop"),
        ]
        assembled_result: dict[str, Any] = {}

        def on_complete(assembled: dict[str, Any]) -> None:
            nonlocal assembled_result
            assembled_result = assembled

        stream = AsyncStreamCapture(
            _FakeAsyncStream(raw_chunks), "openai", on_complete
        )
        captured = []
        async for chunk in stream:
            captured.append(chunk)

        assert len(captured) == 3
        assert assembled_result["choices"][0]["message"]["content"] == "async hello"


# ── StreamReplay tests ──────────────────────────────────────────────


class TestStreamReplay:
    def test_replays_openai_response_as_chunks(self):
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, how can I help?"},
                "finish_reason": "stop",
            }],
        }
        stream = StreamReplay(response, "openai")
        chunks = list(stream)

        # Should have: role chunk + content chunks + finish chunk
        assert len(chunks) >= 3

        # First chunk should have role
        assert chunks[0].choices[0].delta.role == "assistant"

        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_replays_anthropic_response_as_chunks(self):
        response = {
            "id": "msg-123",
            "type": "message",
            "role": "assistant",
            "model": "claude-4-sonnet",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        stream = StreamReplay(response, "anthropic")
        chunks = list(stream)

        # Should have: message_start, content_block_start, delta, stop, message_delta, message_stop
        assert len(chunks) >= 4
        assert chunks[0].type == "message_start"
        assert chunks[-1].type == "message_stop"

    def test_context_manager(self):
        response = {
            "id": "test",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
        }
        with StreamReplay(response, "openai") as stream:
            chunks = list(stream)
        assert len(chunks) >= 2


# ── AsyncStreamReplay tests ─────────────────────────────────────────


class TestAsyncStreamReplay:
    @pytest.mark.asyncio
    async def test_replays_async(self):
        response = {
            "id": "test",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
        }
        stream = AsyncStreamReplay(response, "openai")
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) >= 2


# ── Assembly roundtrip tests ────────────────────────────────────────


class TestAssemblyRoundtrip:
    def test_openai_roundtrip(self):
        """Assemble chunks, then split, then re-assemble should preserve content."""
        original = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                },
                "finish_reason": "stop",
            }],
        }
        # Split into chunks
        chunks = _split_openai_response(original)
        assert len(chunks) >= 3

        # Re-assemble from chunks (already dicts, not SDK objects)
        reassembled = _assemble_openai_chunks(chunks)
        assert reassembled["choices"][0]["message"]["content"] == "The answer is 42."
        assert reassembled["model"] == "gpt-4o"
        assert reassembled["choices"][0]["finish_reason"] == "stop"

    def test_openai_tool_call_roundtrip(self):
        original = {
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        chunks = _split_openai_response(original)
        reassembled = _assemble_openai_chunks(chunks)
        tc = reassembled["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert '{"query": "test"}' == tc["function"]["arguments"]

    def test_anthropic_roundtrip(self):
        original = {
            "id": "msg-abc",
            "type": "message",
            "role": "assistant",
            "model": "claude-4-sonnet",
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
        chunks = _split_anthropic_response(original)
        reassembled = _assemble_anthropic_chunks(chunks)
        assert reassembled["content"][0]["text"] == "Let me search for that."
        assert reassembled["content"][1]["name"] == "search"
        assert reassembled["content"][1]["input"] == {"q": "test"}
        assert reassembled["stop_reason"] == "tool_use"
