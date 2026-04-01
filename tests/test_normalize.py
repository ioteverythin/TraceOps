"""Tests for the normalization module."""

from __future__ import annotations

from trace_ops.normalize import (
    NormalizedResponse,
    NormalizedToolCall,
    normalize_anthropic_response,
    normalize_for_comparison,
    normalize_openai_response,
    normalize_response,
)

# ── NormalizedToolCall ──────────────────────────────────────────────


class TestNormalizedToolCall:
    def test_from_dict(self):
        tc = NormalizedToolCall(name="search", arguments={"q": "test"}, id="call_1")
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}
        assert tc.id == "call_1"

    def test_defaults(self):
        tc = NormalizedToolCall(name="run", arguments={})
        assert tc.id is None

    def test_to_dict(self):
        tc = NormalizedToolCall(name="search", arguments={"q": "t"}, id="c1")
        d = tc.__dict__
        assert d["name"] == "search"


# ── NormalizedResponse ──────────────────────────────────────────────


class TestNormalizedResponse:
    def test_defaults(self):
        nr = NormalizedResponse(content="Hello", model="gpt-4o")
        assert nr.content == "Hello"
        assert nr.tool_calls == []
        assert nr.role == "assistant"
        assert nr.finish_reason is None
        assert nr.input_tokens is None
        assert nr.output_tokens is None


# ── normalize_openai_response ───────────────────────────────────────


class TestNormalizeOpenAI:
    def test_simple_text_response(self):
        response = {
            "id": "chatcmpl-abc",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        nr = normalize_openai_response(response)
        assert nr.content == "Hello!"
        assert nr.model == "gpt-4o"
        assert nr.role == "assistant"
        assert nr.finish_reason == "stop"
        assert nr.input_tokens == 10
        assert nr.output_tokens == 5
        assert nr.tool_calls == []

    def test_tool_call_response(self):
        response = {
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        nr = normalize_openai_response(response)
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "get_weather"
        assert nr.tool_calls[0].arguments == {"city": "NYC"}
        # OpenAI normalizer does not preserve call IDs
        assert nr.tool_calls[0].id is None

    def test_no_usage(self):
        response = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        nr = normalize_openai_response(response)
        assert nr.input_tokens is None
        assert nr.output_tokens is None


# ── normalize_anthropic_response ────────────────────────────────────


class TestNormalizeAnthropic:
    def test_simple_text_response(self):
        response = {
            "id": "msg-123",
            "type": "message",
            "role": "assistant",
            "model": "claude-4-sonnet",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 8},
        }
        nr = normalize_anthropic_response(response)
        assert nr.content == "Hello!"
        assert nr.model == "claude-4-sonnet"
        assert nr.role == "assistant"
        assert nr.finish_reason == "end_turn"
        assert nr.input_tokens == 15
        assert nr.output_tokens == 8
        assert nr.tool_calls == []

    def test_tool_use_response(self):
        response = {
            "id": "msg-tc",
            "type": "message",
            "role": "assistant",
            "model": "claude-4-sonnet",
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {"query": "test"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 12},
        }
        nr = normalize_anthropic_response(response)
        assert nr.content == "Let me search."
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "search"
        assert nr.tool_calls[0].arguments == {"query": "test"}
        # Anthropic normalizer doesn't preserve tool_use block IDs
        assert nr.tool_calls[0].id is None

    def test_multiple_content_blocks(self):
        response = {
            "id": "msg-multi",
            "model": "claude-4-sonnet",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First "},
                {"type": "text", "text": "Second"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        nr = normalize_anthropic_response(response)
        assert nr.content == "First \nSecond"


# ── normalize_response (auto-detect) ───────────────────────────────


class TestNormalizeResponse:
    def test_detect_openai(self):
        response = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        }
        nr = normalize_response(response, provider="openai")
        assert nr.content == "hi"

    def test_detect_anthropic(self):
        response = {
            "id": "msg-1",
            "model": "claude-4-sonnet",
            "role": "assistant",
            "content": [{"type": "text", "text": "yo"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        nr = normalize_response(response, provider="anthropic")
        assert nr.content == "yo"

    def test_litellm_uses_openai_format(self):
        response = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        nr = normalize_response(response, provider="litellm")
        assert nr.content == "ok"


# ── normalize_for_comparison ────────────────────────────────────────


class TestNormalizeForComparison:
    def test_strips_volatile_fields(self):
        response = {
            "id": "chatcmpl-abc",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        clean = normalize_for_comparison(response, "openai")
        # Should not have IDs or token counts
        assert "id" not in str(clean) or clean.get("tool_calls", [{}])[0].get("id") is None
        # Content should still be present
        assert clean["content"] == "Hello!"

    def test_openai_and_anthropic_produce_same_shape(self):
        openai_resp = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        }
        anthropic_resp = {
            "model": "claude-4-sonnet",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        o = normalize_for_comparison(openai_resp, "openai")
        a = normalize_for_comparison(anthropic_resp, "anthropic")
        # Both should have the same keys
        assert set(o.keys()) == set(a.keys())
        assert o["content"] == a["content"]
