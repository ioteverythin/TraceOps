"""Tests for RAG export — to_ragas_dataset, to_deepeval_dataset, to_csv, to_openai_finetune."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import save_cassette


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_rag_cassette(path: Path, query: str = "What is AI?") -> None:
    """Save a minimal cassette with a retrieval + LLM response."""
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        input_tokens=10,
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.RETRIEVAL,
        query=query,
        chunks=[
            {"id": "c1", "text": "AI stands for artificial intelligence", "score": 0.95},
            {"id": "c2", "text": "Machine learning is a subset of AI", "score": 0.87},
        ],
        vector_store="chromadb",
        duration_ms=45.0,
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model="gpt-4o-mini",
        response={
            "choices": [{"message": {"role": "assistant", "content": "AI is artificial intelligence."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        output_tokens=5,
    ))
    trace.finalize()
    save_cassette(trace, str(path))


# ── to_csv ─────────────────────────────────────────────────────────────────


def test_to_csv_creates_file(tmp_path: Path):
    from trace_ops.rag.export import to_csv

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    _make_rag_cassette(cassette_dir / "test.yaml")
    out = tmp_path / "output.csv"

    to_csv(cassette_dir, out)
    assert out.exists()


def test_to_csv_has_header_and_rows(tmp_path: Path):
    from trace_ops.rag.export import to_csv

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    _make_rag_cassette(cassette_dir / "test.yaml")
    out = tmp_path / "output.csv"
    to_csv(cassette_dir, out)

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 1
    assert "query" in rows[0]
    assert "num_chunks" in rows[0]


def test_to_csv_multiple_cassettes(tmp_path: Path):
    from trace_ops.rag.export import to_csv

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    _make_rag_cassette(cassette_dir / "test1.yaml", query="What is AI?")
    _make_rag_cassette(cassette_dir / "test2.yaml", query="What is ML?")
    out = tmp_path / "output.csv"
    to_csv(cassette_dir, out)

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2


def test_to_csv_empty_directory(tmp_path: Path):
    from trace_ops.rag.export import to_csv

    cassette_dir = tmp_path / "empty"
    cassette_dir.mkdir()
    out = tmp_path / "output.csv"
    to_csv(cassette_dir, out)
    # Should create file with just header
    assert out.exists()


# ── to_ragas_dataset ──────────────────────────────────────────────────────


def test_to_ragas_dataset_raises_without_ragas(tmp_path: Path):
    """Should raise ImportError when ragas is not installed."""
    from trace_ops.rag.export import to_ragas_dataset

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    _make_rag_cassette(cassette_dir / "test.yaml")

    try:
        import ragas  # noqa: F401
        # ragas is installed — skip this test
        pytest.skip("ragas is installed; can't test ImportError path")
    except ImportError:
        with pytest.raises(ImportError, match="ragas"):
            to_ragas_dataset(cassette_dir)


# ── to_deepeval_dataset ──────────────────────────────────────────────────


def test_to_deepeval_dataset_raises_without_deepeval(tmp_path: Path):
    from trace_ops.rag.export import to_deepeval_dataset

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    _make_rag_cassette(cassette_dir / "test.yaml")

    try:
        import deepeval  # noqa: F401
        pytest.skip("deepeval is installed; can't test ImportError path")
    except ImportError:
        with pytest.raises(ImportError, match="deepeval"):
            to_deepeval_dataset(cassette_dir)


# ── to_openai_finetune (from rag/export.py) ──────────────────────────────


def test_to_openai_finetune_from_rag_export(tmp_path: Path):
    from trace_ops.rag.export import to_openai_finetune

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    _make_rag_cassette(cassette_dir / "test.yaml")
    out = tmp_path / "finetune.jsonl"
    to_openai_finetune(cassette_dir, out)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert "messages" in record
