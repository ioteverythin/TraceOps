"""Extended tests for normalize.py — provider-agnostic response normalization.

Covers: NormalizedToolCall, NormalizedResponse, normalize_openai_response,
normalize_anthropic_response, normalize_response, normalize_for_comparison.
"""

from __future__ import annotations

import pytest

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
    def test_to_dict_without_id(self):
        tc = NormalizedToolCall(name="search", arguments={"q": "test"})
        d = tc.to_dict()
        assert d == {"name": "search", "arguments": {"q": "test"}}
        assert "id" not in d

    def test_to_dict_with_id(self):
        tc = NormalizedToolCall(name="search", arguments={"q": "test"}, id="call_123")
        d = tc.to_dict()
        assert d["id"] == "call_123"

    def test_from_dict_full(self):
        tc = NormalizedToolCall.from_dict(
            {"name": "search", "arguments": {"q": "hello"}, "id": "c1"}
        )
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}
        assert tc.id == "c1"

    def test_from_dict_minimal(self):
        tc = NormalizedToolCall.from_dict({})
        assert tc.name == ""
        assert tc.arguments == {}
        assert tc.id is None

    def test_roundtrip(self):
        original = NormalizedToolCall(name="read", arguments={"path": "/a"}, id="x")
        restored = NormalizedToolCall.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.arguments == original.arguments
        assert restored.id == original.id


# ── NormalizedResponse ──────────────────────────────────────────────


class TestNormalizedResponse:
    def test_to_dict_minimal(self):
        resp = NormalizedResponse()
        d = resp.to_dict()
        assert d == {"role": "assistant", "model": ""}
        assert "content" not in d
        assert "tool_calls" not in d

    def test_to_dict_full(self):
        resp = NormalizedResponse(
            content="Hello!",
            tool_calls=[NormalizedToolCall(name="search", arguments={"q": "t"})],
            model="gpt-4o",
            role="assistant",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=50,
        )
        d = resp.to_dict()
        assert d["content"] == "Hello!"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "search"
        assert d["finish_reason"] == "stop"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50

    def test_to_dict_no_content(self):
        resp = NormalizedResponse(content=None, finish_reason="tool_calls")
        d = resp.to_dict()
        assert "content" not in d
        assert d["finish_reason"] == "tool_calls"

    def test_to_dict_no_finish_reason(self):
        resp = NormalizedResponse(content="hi", finish_reason=None)
        d = resp.to_dict()
        assert "finish_reason" not in d

    def test_from_dict_minimal(self):
        resp = NormalizedResponse.from_dict({})
        assert resp.content is None
        assert resp.tool_calls == []
        assert resp.model == ""
        assert resp.role == "assistant"
        assert resp.finish_reason is None
        assert resp.input_tokens is None
        assert resp.output_tokens is None

    def test_from_dict_full(self):
        resp = NormalizedResponse.from_dict({
            "content": "Answer",
            "tool_calls": [{"name": "fn", "arguments": {"x": 1}}],
            "model": "gpt-4o",
            "role": "assistant",
            "finish_reason": "stop",
            "input_tokens": 200,
            "output_tokens": 100,
        })
        assert resp.content == "Answer"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fn"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.input_tokens == 200

    def test_roundtrip(self):
        original = NormalizedResponse(
            content="Test",
            tool_calls=[NormalizedToolCall(name="a", arguments={"k": "v"})],
            model="model-x",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=20,
        )
        restored = NormalizedResponse.from_dict(original.to_dict())
        assert restored.content == original.content
        assert restored.model == original.model
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "a"


# ── normalize_openai_response ──────────────────────────────────────


class TestNormalizeOpenAIResponse:
    def test_basic_response(self):
        resp = normalize_openai_response({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        assert resp.content == "Hello, world!"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.tool_calls == []

    def test_empty_choices(self):
        resp = normalize_openai_response({"model": "gpt-4o", "choices": []})
        assert resp.content is None
        assert resp.model == "gpt-4o"
        assert resp.tool_calls == []

    def test_no_choices_key(self):
        resp = normalize_openai_response({"model": "gpt-4o"})
        assert resp.content is None

    def test_tool_calls_with_json_arguments(self):
        resp = normalize_openai_response({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"query": "test"}

    def test_tool_calls_with_dict_arguments(self):
        resp = normalize_openai_response({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": "run",
                            "arguments": {"cmd": "ls"},
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })
        assert resp.tool_calls[0].arguments == {"cmd": "ls"}

    def test_tool_calls_with_malformed_json(self):
        resp = normalize_openai_response({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": "fn",
                            "arguments": "not-json{",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })
        assert resp.tool_calls[0].arguments == {"_raw": "not-json{"}

    def test_multiple_tool_calls(self):
        resp = normalize_openai_response({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "a", "arguments": "{}"}},
                        {"function": {"name": "b", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
        })
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "a"
        assert resp.tool_calls[1].name == "b"

    def test_no_usage(self):
        resp = normalize_openai_response({
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        })
        assert resp.input_tokens is None
        assert resp.output_tokens is None


# ── normalize_anthropic_response ───────────────────────────────────


class TestNormalizeAnthropicResponse:
    def test_text_response(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        })
        assert resp.content == "Hello!"
        assert resp.model == "claude-3-opus"
        assert resp.finish_reason == "end_turn"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 10

    def test_tool_use_response(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "name": "search", "input": {"q": "test"}},
            ],
        })
        assert resp.content == "Let me search."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"q": "test"}

    def test_tool_only_no_text(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "content": [
                {"type": "tool_use", "name": "run", "input": {}},
            ],
        })
        assert resp.content is None
        assert len(resp.tool_calls) == 1

    def test_multiple_text_blocks(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "content": [
                {"type": "text", "text": "Line one"},
                {"type": "text", "text": "Line two"},
            ],
        })
        assert resp.content == "Line one\nLine two"

    def test_empty_content(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "content": [],
        })
        assert resp.content is None
        assert resp.tool_calls == []

    def test_unknown_block_type_ignored(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "After image"},
            ],
        })
        assert resp.content == "After image"
        assert resp.tool_calls == []

    def test_no_usage(self):
        resp = normalize_anthropic_response({
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hi"}],
        })
        assert resp.input_tokens is None
        assert resp.output_tokens is None


# ── normalize_response dispatcher ──────────────────────────────────


class TestNormalizeResponse:
    def test_openai_provider(self):
        resp = normalize_response(
            {"model": "gpt-4o", "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]},
            "openai",
        )
        assert resp.content == "Hi"

    def test_litellm_uses_openai_format(self):
        resp = normalize_response(
            {"model": "gpt-4o", "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]},
            "litellm",
        )
        assert resp.content == "Hi"

    def test_anthropic_provider(self):
        resp = normalize_response(
            {"model": "claude-3", "content": [{"type": "text", "text": "Hello"}]},
            "anthropic",
        )
        assert resp.content == "Hello"

    def test_unknown_provider_fallback(self):
        resp = normalize_response({"custom": "data"}, "some_unknown_provider")
        assert resp.content is not None
        assert "custom" in resp.content


# ── normalize_for_comparison ───────────────────────────────────────


class TestNormalizeForComparison:
    def test_strips_token_counts(self):
        d = normalize_for_comparison(
            {
                "model": "gpt-4o",
                "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
            "openai",
        )
        assert "input_tokens" not in d
        assert "output_tokens" not in d

    def test_strips_tool_call_ids(self):
        d = normalize_for_comparison(
            {
                "model": "gpt-4o",
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"function": {"name": "a", "arguments": "{}"}, "id": "call_123"},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
            "openai",
        )
        for tc in d.get("tool_calls", []):
            assert "id" not in tc

    def test_anthropic_comparison(self):
        d = normalize_for_comparison(
            {
                "model": "claude-3",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"input_tokens": 50, "output_tokens": 25},
            },
            "anthropic",
        )
        assert d["content"] == "Hi"
        assert "input_tokens" not in d
        assert "output_tokens" not in d

    def test_unknown_provider_comparison(self):
        d = normalize_for_comparison({"foo": "bar"}, "custom")
        assert "content" in d
