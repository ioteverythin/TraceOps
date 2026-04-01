"""Fine-tuning dataset exporters for traceops.

Convert recorded cassettes into LLM fine-tuning datasets for
OpenAI and Anthropic supervised fine-tuning APIs.

Usage::

    from trace_ops.export.finetune import to_openai_finetune, to_anthropic_finetune

    to_openai_finetune("cassettes/", output="finetune_openai.jsonl")
    to_anthropic_finetune("cassettes/", output="finetune_anthropic.jsonl")
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any


def _load_traces(
    cassette_dir: str | Path,
    include_patterns: list[str] | None = None,
) -> list[Any]:
    from trace_ops.cassette import load_cassette

    path = Path(cassette_dir)
    patterns = include_patterns or ["*.yaml", "*.yml"]
    traces = []
    for pattern in patterns:
        for cassette_path in sorted(path.glob(pattern)):
            with contextlib.suppress(Exception):
                traces.append((cassette_path, load_cassette(cassette_path)))
    return traces


def to_openai_finetune(
    cassette_dir: str | Path,
    output: str | Path,
    include_patterns: list[str] | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> Path:
    """Export cassettes to OpenAI fine-tuning JSONL format.

    Each cassette becomes one training example with the structure:
    ``{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}``.

    Args:
        cassette_dir: Directory containing cassette YAML files.
        output: Path to write the output JSONL file.
        include_patterns: Glob patterns (default ``["*.yaml"]``).
        system_prompt: Default system message if none recorded.

    Returns:
        Path to the written file.
    """
    from trace_ops._types import EventType

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out, "w", encoding="utf-8") as f:
        for _path, trace in _load_traces(cassette_dir, include_patterns):
            llm_requests = [e for e in trace.events if e.event_type == EventType.LLM_REQUEST]
            llm_responses = [e for e in trace.events if e.event_type == EventType.LLM_RESPONSE]
            if not llm_requests or not llm_responses:
                continue

            messages: list[dict[str, Any]] = list(llm_requests[0].messages or [])
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            resp = llm_responses[0].response or {}
            choices = resp.get("choices") or []
            content = ""
            if choices:
                content = (choices[0].get("message") or {}).get("content", "") or ""
            messages.append({"role": "assistant", "content": content})

            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    return out


def to_anthropic_finetune(
    cassette_dir: str | Path,
    output: str | Path,
    include_patterns: list[str] | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> Path:
    """Export cassettes to Anthropic fine-tuning JSONL format.

    Produces ``{"system": ..., "messages": [...]}`` records compatible
    with the Anthropic Model Fine-tuning API.

    Args:
        cassette_dir: Directory containing cassette YAML files.
        output: Path to write the output JSONL file.
        include_patterns: Glob patterns (default ``["*.yaml"]``).
        system_prompt: Default system message if none recorded.

    Returns:
        Path to the written file.
    """
    from trace_ops._types import EventType

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for _path, trace in _load_traces(cassette_dir, include_patterns):
            llm_requests = [e for e in trace.events if e.event_type == EventType.LLM_REQUEST]
            llm_responses = [e for e in trace.events if e.event_type == EventType.LLM_RESPONSE]
            if not llm_requests or not llm_responses:
                continue

            raw_messages = list(llm_requests[0].messages or [])
            system = system_prompt
            messages: list[dict[str, Any]] = []
            for m in raw_messages:
                if m.get("role") == "system":
                    system = m.get("content", system_prompt) or system_prompt
                else:
                    messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

            # Anthropic response
            resp = llm_responses[0].response or {}
            content_blocks = resp.get("content") or []
            if isinstance(content_blocks, list) and content_blocks:
                reply_text = content_blocks[0].get("text", "") if isinstance(content_blocks[0], dict) else str(content_blocks[0])
            else:
                reply_text = str(resp.get("content", ""))

            messages.append({"role": "assistant", "content": reply_text})

            record: dict[str, Any] = {"system": system, "messages": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out
