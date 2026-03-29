"""Response normalization — provider-agnostic tool call and response comparison.

LLM providers return tool calls in different formats.  Comparing raw
response dicts across providers (or even across model versions of the
same provider) is fragile because IDs, ordering, and field names differ.

This module provides normalized dataclasses and conversion functions
that the diff engine uses to compare *semantics*, not syntax.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedToolCall:
    """Provider-agnostic representation of a single tool call.

    Attributes:
        name: The function / tool name.
        arguments: Parsed argument dict.
        id: Optional call ID (stripped during comparison because
            providers regenerate IDs on each call).
    """

    name: str
    arguments: dict[str, Any]
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d: dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.id is not None:
            d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedToolCall:
        """Deserialize from a plain dict."""
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
            id=data.get("id"),
        )


@dataclass
class NormalizedResponse:
    """Provider-agnostic LLM response representation.

    Captures the essential semantic content from any provider's format.

    Attributes:
        content: Text content of the response (``None`` if tool-call-only).
        tool_calls: Normalized tool calls, in order.
        model: The model that generated the response.
        role: Response role (typically ``"assistant"``).
        finish_reason: Why the model stopped (``"stop"``, ``"tool_calls"``, etc.).
        input_tokens: Prompt / input token count.
        output_tokens: Completion / output token count.
    """

    content: str | None = None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    model: str = ""
    role: str = "assistant"
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d: dict[str, Any] = {"role": self.role, "model": self.model}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.finish_reason:
            d["finish_reason"] = self.finish_reason
        if self.input_tokens is not None:
            d["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            d["output_tokens"] = self.output_tokens
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedResponse:
        """Deserialize from a plain dict."""
        tcs = [NormalizedToolCall.from_dict(tc) for tc in data.get("tool_calls", [])]
        return cls(
            content=data.get("content"),
            tool_calls=tcs,
            model=data.get("model", ""),
            role=data.get("role", "assistant"),
            finish_reason=data.get("finish_reason"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
        )


# ── Provider-specific normalizers ───────────────────────────────────


def normalize_openai_response(response: dict[str, Any]) -> NormalizedResponse:
    """Normalize an OpenAI chat-completion response dict.

    Args:
        response: A dict produced by ``response.model_dump()`` or loaded
            from a cassette.

    Returns:
        A :class:`NormalizedResponse`.
    """
    choices = response.get("choices", [])
    if not choices:
        return NormalizedResponse(model=response.get("model", ""))

    choice = choices[0]
    message = choice.get("message", {})

    tool_calls: list[NormalizedToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                parsed = {"_raw": raw_args}
        else:
            parsed = raw_args
        tool_calls.append(NormalizedToolCall(
            name=fn.get("name", ""),
            arguments=parsed,
        ))

    usage = response.get("usage", {})

    return NormalizedResponse(
        content=message.get("content"),
        tool_calls=tool_calls,
        model=response.get("model", ""),
        role=message.get("role", "assistant"),
        finish_reason=choice.get("finish_reason"),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def normalize_anthropic_response(response: dict[str, Any]) -> NormalizedResponse:
    """Normalize an Anthropic message response dict.

    Args:
        response: A dict from ``response.model_dump()`` or cassette.

    Returns:
        A :class:`NormalizedResponse`.
    """
    content_blocks = response.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[NormalizedToolCall] = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(NormalizedToolCall(
                name=block.get("name", ""),
                arguments=block.get("input", {}),
            ))

    content = "\n".join(text_parts) if text_parts else None
    usage = response.get("usage", {})

    return NormalizedResponse(
        content=content,
        tool_calls=tool_calls,
        model=response.get("model", ""),
        role=response.get("role", "assistant"),
        finish_reason=response.get("stop_reason"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


# ── Dispatcher ──────────────────────────────────────────────────────


def normalize_response(
    response: dict[str, Any], provider: str
) -> NormalizedResponse:
    """Normalize a response dict based on provider.

    Args:
        response: The raw response dict.
        provider: ``"openai"``, ``"anthropic"``, ``"litellm"``, etc.

    Returns:
        A :class:`NormalizedResponse`.
    """
    if provider in ("openai", "litellm"):
        return normalize_openai_response(response)
    elif provider == "anthropic":
        return normalize_anthropic_response(response)
    # Fallback – wrap entire dict as content
    return NormalizedResponse(content=str(response))


def normalize_for_comparison(
    response: dict[str, Any], provider: str
) -> dict[str, Any]:
    """Normalize a response dict for diffing, stripping volatile fields.

    This is the function the diff engine calls.  It normalizes the
    response *and* removes fields that commonly change between runs
    (token counts, IDs) so that comparisons focus on semantics.

    Args:
        response: The raw response dict.
        provider: Provider name.

    Returns:
        A plain dict suitable for :class:`deepdiff.DeepDiff`.
    """
    normalized = normalize_response(response, provider)
    d = normalized.to_dict()
    # Strip volatile fields
    d.pop("input_tokens", None)
    d.pop("output_tokens", None)
    # Strip tool-call IDs (providers regenerate them each run)
    for tc in d.get("tool_calls", []):
        tc.pop("id", None)
    return d
