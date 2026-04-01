"""Streaming interception — capture and replay streamed LLM responses.

Production LLM calls commonly use ``stream=True``, returning chunks
incrementally. This module provides wrappers that:

  - **Recording**: wrap real streams to capture chunks while yielding
    them through to the caller. When the stream completes, the
    ``on_complete`` callback is invoked with the assembled response.
  - **Replay**: split a recorded complete response back into chunks
    for a realistic streaming experience.

Design decision: cassettes store the *assembled* complete response,
not individual chunks.  This keeps files compact and human-readable.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

# ── Recording wrappers ──────────────────────────────────────────────


class StreamCapture:
    """Wraps a real LLM stream and records chunks as they flow through.

    When iteration completes (or the context manager exits), the
    ``on_complete`` callback is invoked with the assembled response.

    Args:
        stream: The original stream object from the LLM SDK.
        provider: Provider name (``"openai"``, ``"anthropic"``, ``"litellm"``).
        on_complete: Callback receiving the assembled response dict.
    """

    def __init__(
        self,
        stream: Any,
        provider: str,
        on_complete: Callable[[dict[str, Any]], None],
    ) -> None:
        self._stream = stream
        self._provider = provider
        self._on_complete = on_complete
        self._chunks: list[Any] = []
        self._done = False

    # -- iterator protocol --

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._finish()

    def __next__(self) -> Any:  # pragma: no cover – used via __iter__
        chunk = next(self._stream)
        self._chunks.append(chunk)
        return chunk

    # -- context-manager protocol (OpenAI Stream is a CM) --

    def __enter__(self) -> StreamCapture:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)
        self._finish()

    # -- internals --

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            assembled = _assemble_chunks(self._chunks, self._provider)
            self._on_complete(assembled)


class AsyncStreamCapture:
    """Async variant of :class:`StreamCapture`.

    Args:
        stream: The original async stream object.
        provider: Provider name.
        on_complete: Callback receiving the assembled response dict.
    """

    def __init__(
        self,
        stream: Any,
        provider: str,
        on_complete: Callable[[dict[str, Any]], None],
    ) -> None:
        self._stream = stream
        self._provider = provider
        self._on_complete = on_complete
        self._chunks: list[Any] = []
        self._done = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._stream:
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._finish()

    async def __aenter__(self) -> AsyncStreamCapture:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)
        self._finish()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            assembled = _assemble_chunks(self._chunks, self._provider)
            self._on_complete(assembled)


# ── Replay wrappers ─────────────────────────────────────────────────


class StreamReplay:
    """Replays a recorded complete response as a stream of chunks.

    Splits the recorded response into realistic-looking chunks so that
    code which iterates over a stream behaves identically during replay.

    Args:
        response_dict: The recorded complete response dict.
        provider: Provider name (``"openai"``, ``"anthropic"``, ``"litellm"``).
    """

    def __init__(self, response_dict: dict[str, Any], provider: str) -> None:
        self._chunks = _split_into_chunks(response_dict, provider)

    def __iter__(self) -> Iterator[Any]:
        for chunk in self._chunks:
            yield _dict_to_namespace(chunk)

    def __enter__(self) -> StreamReplay:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class AsyncStreamReplay:
    """Async variant of :class:`StreamReplay`.

    Args:
        response_dict: The recorded complete response dict.
        provider: Provider name.
    """

    def __init__(self, response_dict: dict[str, Any], provider: str) -> None:
        self._chunks = _split_into_chunks(response_dict, provider)

    async def __aiter__(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield _dict_to_namespace(chunk)

    async def __aenter__(self) -> AsyncStreamReplay:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ── Chunk assembly (chunks → complete response) ────────────────────


def _assemble_chunks(chunks: list[Any], provider: str) -> dict[str, Any]:
    """Assemble captured chunks into a complete response dict.

    Args:
        chunks: Raw chunk objects from the LLM SDK.
        provider: Provider name.

    Returns:
        A complete response dict suitable for cassette storage.
    """
    if provider in ("openai", "litellm"):
        return _assemble_openai_chunks(chunks)
    elif provider == "anthropic":
        return _assemble_anthropic_chunks(chunks)
    return {"_raw_chunks": [_to_dict(c) for c in chunks]}


def _assemble_openai_chunks(chunks: list[Any]) -> dict[str, Any]:
    """Assemble OpenAI ``ChatCompletionChunk`` objects into a full completion."""
    content_parts: list[str] = []
    tool_calls_map: dict[int, dict[str, Any]] = {}
    model = ""
    finish_reason: str | None = None
    role = "assistant"
    usage: dict[str, Any] = {}
    chunk_id = ""

    for raw in chunks:
        c = _to_dict(raw)
        if not c:
            continue

        chunk_id = c.get("id", chunk_id)
        model = c.get("model", model)
        if c.get("usage"):
            usage = c["usage"]

        choices = c.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if delta.get("role"):
            role = delta["role"]
        if delta.get("content"):
            content_parts.append(delta["content"])

        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            if idx not in tool_calls_map:
                tool_calls_map[idx] = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            if tc.get("id"):
                tool_calls_map[idx]["id"] = tc["id"]
            fn = tc.get("function", {})
            if fn.get("name"):
                tool_calls_map[idx]["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

    content = "".join(content_parts) or None
    tool_calls = (
        [tool_calls_map[i] for i in sorted(tool_calls_map)]
        if tool_calls_map
        else None
    )

    message: dict[str, Any] = {"role": role}
    if content is not None:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls

    result: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        result["usage"] = usage
    return result


def _assemble_anthropic_chunks(chunks: list[Any]) -> dict[str, Any]:
    """Assemble Anthropic streaming events into a complete message."""
    content_blocks: list[dict[str, Any]] = []
    model = ""
    stop_reason: str | None = None
    usage: dict[str, Any] = {}
    msg_id = ""
    role = "assistant"
    current_block: dict[str, Any] | None = None

    for raw in chunks:
        c = _to_dict(raw)
        if not c:
            continue

        evt = c.get("type", "")

        if evt == "message_start":
            msg = c.get("message", {})
            msg_id = msg.get("id", msg_id)
            model = msg.get("model", model)
            role = msg.get("role", role)
            usage = msg.get("usage", usage)

        elif evt == "content_block_start":
            block = c.get("content_block", {})
            current_block = dict(block)
            if current_block.get("type") == "text":
                current_block.setdefault("text", "")
            elif current_block.get("type") == "tool_use":
                current_block.setdefault("input", {})
                current_block["_input_json"] = ""

        elif evt == "content_block_delta":
            delta = c.get("delta", {})
            if current_block is not None:
                if delta.get("type") == "text_delta":
                    current_block["text"] = (
                        current_block.get("text", "") + delta.get("text", "")
                    )
                elif delta.get("type") == "input_json_delta":
                    current_block["_input_json"] = (
                        current_block.get("_input_json", "")
                        + delta.get("partial_json", "")
                    )

        elif evt == "content_block_stop":
            if current_block is not None:
                if (
                    current_block.get("type") == "tool_use"
                    and current_block.get("_input_json")
                ):
                    try:
                        current_block["input"] = json.loads(
                            current_block["_input_json"]
                        )
                    except (json.JSONDecodeError, TypeError):
                        current_block["input"] = {}
                current_block.pop("_input_json", None)
                content_blocks.append(current_block)
                current_block = None

        elif evt == "message_delta":
            delta = c.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)
            if delta.get("usage"):
                usage.update(delta["usage"])

    return {
        "id": msg_id,
        "type": "message",
        "role": role,
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": usage,
    }


# ── Chunk splitting (complete response → chunks) ───────────────────

_CHUNK_TEXT_SIZE = 20  # characters per simulated chunk


def _split_into_chunks(
    response: dict[str, Any], provider: str
) -> list[dict[str, Any]]:
    """Split a complete response into streaming chunks.

    Args:
        response: The complete response dict from the cassette.
        provider: Provider name.

    Returns:
        A list of chunk dicts simulating a streamed response.
    """
    if provider in ("openai", "litellm"):
        return _split_openai_response(response)
    elif provider == "anthropic":
        return _split_anthropic_response(response)
    return [response]


def _split_openai_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Split an OpenAI chat completion into streaming chunks."""
    chunks: list[dict[str, Any]] = []
    choices = response.get("choices", [])
    if not choices:
        return [response]

    message = choices[0].get("message", {})
    content = message.get("content") or ""
    model = response.get("model", "")
    chunk_id = response.get("id", "")
    tool_calls = message.get("tool_calls")
    finish_reason = choices[0].get("finish_reason", "stop")

    # First chunk: role
    chunks.append({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    })

    # Content chunks
    if content:
        for i in range(0, len(content), _CHUNK_TEXT_SIZE):
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content[i : i + _CHUNK_TEXT_SIZE]},
                    "finish_reason": None,
                }],
            })

    # Tool-call chunks
    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            # Header chunk with id + name
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": idx,
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": "",
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            })
            # Argument chunks
            args_str = fn.get("arguments", "")
            for i in range(0, max(1, len(args_str)), _CHUNK_TEXT_SIZE):
                chunks.append({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "function": {
                                    "arguments": args_str[i : i + _CHUNK_TEXT_SIZE],
                                },
                            }],
                        },
                        "finish_reason": None,
                    }],
                })

    # Final chunk: finish_reason
    chunks.append({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    })

    return chunks


def _split_anthropic_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Split an Anthropic message into streaming events."""
    chunks: list[dict[str, Any]] = []

    # message_start
    chunks.append({
        "type": "message_start",
        "message": {
            "id": response.get("id", ""),
            "type": "message",
            "role": response.get("role", "assistant"),
            "model": response.get("model", ""),
            "content": [],
            "usage": response.get("usage", {}),
        },
    })

    for idx, block in enumerate(response.get("content", [])):
        block_type = block.get("type", "text")

        if block_type == "text":
            chunks.append({
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })
            text = block.get("text", "")
            for i in range(0, max(1, len(text)), _CHUNK_TEXT_SIZE):
                chunks.append({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "text_delta",
                        "text": text[i : i + _CHUNK_TEXT_SIZE],
                    },
                })

        elif block_type == "tool_use":
            start_block = {k: v for k, v in block.items() if k != "input"}
            start_block["input"] = {}
            chunks.append({
                "type": "content_block_start",
                "index": idx,
                "content_block": start_block,
            })
            input_json = json.dumps(block.get("input", {}))
            for i in range(0, max(1, len(input_json)), _CHUNK_TEXT_SIZE):
                chunks.append({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": input_json[i : i + _CHUNK_TEXT_SIZE],
                    },
                })

        chunks.append({"type": "content_block_stop", "index": idx})

    # message_delta + message_stop
    chunks.append({
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason", "end_turn")},
        "usage": {},
    })
    chunks.append({"type": "message_stop"})

    return chunks


# ── Helpers ─────────────────────────────────────────────────────────


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert an SDK object or dict to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


def _dict_to_namespace(d: Any) -> Any:
    """Convert a dict to a :class:`~types.SimpleNamespace` recursively."""
    from types import SimpleNamespace

    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    return d
