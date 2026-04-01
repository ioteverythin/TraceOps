"""Tests for fine-tuning dataset export functions."""

from __future__ import annotations

import json
from pathlib import Path

from trace_ops._types import EventType, Trace, TraceEvent

# ── helpers ───────────────────────────────────────────────────────────────


def _make_cassette_dir(tmp_path: Path) -> Path:
    """Create a cassette directory with sample YAML cassette files."""
    from trace_ops.cassette import save_cassette

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()

    for i in range(3):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is {i} + {i}?"},
            ],
            provider="openai",
            model="gpt-4o-mini",
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            response={
                "choices": [
                    {"message": {"role": "assistant", "content": f"The answer is {i + i}."}}
                ]
            },
            provider="openai",
            model="gpt-4o-mini",
        ))
        save_cassette(trace, cassette_dir / f"trace_{i}.yaml")

    return cassette_dir


def _make_anthropic_cassette_dir(tmp_path: Path) -> Path:
    """Create a cassette directory with Anthropic-style response format."""
    from trace_ops.cassette import save_cassette

    cassette_dir = tmp_path / "anthropic_cassettes"
    cassette_dir.mkdir()

    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        messages=[
            {"role": "user", "content": "Who invented the telephone?"},
        ],
        provider="anthropic",
        model="claude-3-haiku-20240307",
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        response={
            "content": [{"type": "text", "text": "Alexander Graham Bell invented the telephone."}]
        },
        provider="anthropic",
        model="claude-3-haiku-20240307",
    ))
    save_cassette(trace, cassette_dir / "trace_anthropic.yaml")

    return cassette_dir


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ── to_openai_finetune() ──────────────────────────────────────────────────


def test_openai_finetune_returns_path(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "finetune.jsonl"
    result = to_openai_finetune(cassette_dir, out)
    assert isinstance(result, Path)
    assert result == out


def test_openai_finetune_creates_file(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    assert out.exists()


def test_openai_finetune_line_count(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    assert _count_lines(out) == 3  # 3 cassettes


def test_openai_finetune_valid_jsonl(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    for record in lines:
        assert "messages" in record
        assert isinstance(record["messages"], list)


def test_openai_finetune_messages_structure(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    record = records[0]
    roles = [m["role"] for m in record["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_openai_finetune_system_prompt_present(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    record = records[0]
    roles = [m["role"] for m in record["messages"]]
    assert "system" in roles


def test_openai_finetune_custom_system_prompt(tmp_path):
    # Create cassette without a system message
    from trace_ops.cassette import save_cassette
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = tmp_path / "no_sys"
    cassette_dir.mkdir()
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        messages=[{"role": "user", "content": "hello"}],
        provider="openai",
        model="gpt-4o-mini",
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        response={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        provider="openai",
        model="gpt-4o-mini",
    ))
    save_cassette(trace, cassette_dir / "t.yaml")

    out = tmp_path / "custom_sys.jsonl"
    to_openai_finetune(cassette_dir, out, system_prompt="You are a pirate.")
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    system_msgs = [m for m in records[0]["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "pirate" in system_msgs[0]["content"]


def test_openai_finetune_empty_dir(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = tmp_path / "empty"
    cassette_dir.mkdir()
    out = tmp_path / "empty.jsonl"
    to_openai_finetune(cassette_dir, out)
    assert out.exists()
    assert _count_lines(out) == 0


def test_openai_finetune_creates_parent_dirs(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "deep" / "nested" / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    assert out.exists()


def test_openai_finetune_assistant_content_matches(tmp_path):
    from trace_ops.export.finetune import to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out = tmp_path / "ft.jsonl"
    to_openai_finetune(cassette_dir, out)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    for record in records:
        asst_msgs = [m for m in record["messages"] if m["role"] == "assistant"]
        assert len(asst_msgs) == 1
        assert len(asst_msgs[0]["content"]) > 0


# ── to_anthropic_finetune() ───────────────────────────────────────────────


def test_anthropic_finetune_returns_path(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = _make_anthropic_cassette_dir(tmp_path)
    out = tmp_path / "finetune_anthropic.jsonl"
    result = to_anthropic_finetune(cassette_dir, out)
    assert isinstance(result, Path)
    assert result == out


def test_anthropic_finetune_creates_file(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = _make_anthropic_cassette_dir(tmp_path)
    out = tmp_path / "ft.jsonl"
    to_anthropic_finetune(cassette_dir, out)
    assert out.exists()


def test_anthropic_finetune_valid_jsonl(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = _make_anthropic_cassette_dir(tmp_path)
    out = tmp_path / "ft.jsonl"
    to_anthropic_finetune(cassette_dir, out)
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    for record in lines:
        assert "messages" in record
        assert "system" in record


def test_anthropic_finetune_system_field(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = _make_anthropic_cassette_dir(tmp_path)
    out = tmp_path / "ft.jsonl"
    to_anthropic_finetune(cassette_dir, out, system_prompt="Custom system")
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert records[0]["system"] == "Custom system"


def test_anthropic_finetune_extracts_system_from_messages(tmp_path):
    from trace_ops.cassette import save_cassette
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = tmp_path / "sys_in_msgs"
    cassette_dir.mkdir()
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        messages=[
            {"role": "system", "content": "You are a historian."},
            {"role": "user", "content": "Tell me about Rome."},
        ],
        provider="anthropic",
        model="claude-3-haiku-20240307",
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        response={"content": [{"type": "text", "text": "Rome was a great empire."}]},
        provider="anthropic",
        model="claude-3-haiku-20240307",
    ))
    save_cassette(trace, cassette_dir / "t.yaml")

    out = tmp_path / "hist.jsonl"
    to_anthropic_finetune(cassette_dir, out)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert records[0]["system"] == "You are a historian."
    # system msg should NOT appear in messages list
    roles = [m["role"] for m in records[0]["messages"]]
    assert "system" not in roles


def test_anthropic_finetune_assistant_in_messages(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = _make_anthropic_cassette_dir(tmp_path)
    out = tmp_path / "ft.jsonl"
    to_anthropic_finetune(cassette_dir, out)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    roles = [m["role"] for m in records[0]["messages"]]
    assert "assistant" in roles


def test_anthropic_finetune_response_text_content(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = _make_anthropic_cassette_dir(tmp_path)
    out = tmp_path / "ft.jsonl"
    to_anthropic_finetune(cassette_dir, out)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    asst_msgs = [m for m in records[0]["messages"] if m["role"] == "assistant"]
    assert "Alexander Graham Bell" in asst_msgs[0]["content"]


def test_anthropic_finetune_empty_dir(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune

    cassette_dir = tmp_path / "empty"
    cassette_dir.mkdir()
    out = tmp_path / "empty.jsonl"
    to_anthropic_finetune(cassette_dir, out)
    assert out.exists()
    assert _count_lines(out) == 0


# ── cross-format consistency ──────────────────────────────────────────────


def test_both_formats_produce_same_number_of_records(tmp_path):
    from trace_ops.export.finetune import to_anthropic_finetune, to_openai_finetune

    cassette_dir = _make_cassette_dir(tmp_path)
    out_oai = tmp_path / "oai.jsonl"
    out_ant = tmp_path / "ant.jsonl"
    to_openai_finetune(cassette_dir, out_oai)
    to_anthropic_finetune(cassette_dir, out_ant)
    assert _count_lines(out_oai) == _count_lines(out_ant)
