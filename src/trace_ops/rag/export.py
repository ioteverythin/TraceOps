"""Export recorded RAG cassettes to evaluation framework formats.

Supported export targets:

- **RAGAS** ``EvaluationDataset`` — for running offline RAGAS evaluations
- **DeepEval** ``LLMTestCase`` list — for DeepEval batch evaluation
- **CSV** — for spreadsheet / notebook analysis
- **OpenAI fine-tuning JSONL** — for supervised fine-tuning data preparation

Usage::

    from trace_ops.rag.export import to_ragas_dataset, to_csv

    dataset = to_ragas_dataset("cassettes/")
    to_csv("cassettes/", output_path="analysis.csv")
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Any


def _load_traces(
    cassette_dir: str | Path,
    include_patterns: list[str] | None = None,
) -> list[Any]:
    """Load all traces matching the glob patterns in cassette_dir."""
    from trace_ops.cassette import load_cassette

    path = Path(cassette_dir)
    patterns = include_patterns or ["*.yaml", "*.yml"]
    traces = []
    for pattern in patterns:
        for cassette_path in sorted(path.glob(pattern)):
            with contextlib.suppress(Exception):
                traces.append((cassette_path, load_cassette(cassette_path)))
    return traces


def to_ragas_dataset(
    cassette_dir: str | Path,
    include_patterns: list[str] | None = None,
) -> Any:
    """Export cassettes to a RAGAS ``EvaluationDataset``.

    Each cassette contributes one sample per retrieval event, paired with
    the first LLM response in the trace.

    Args:
        cassette_dir: Directory containing cassette YAML files.
        include_patterns: Glob patterns for cassette files (default ``["*.yaml"]``).

    Returns:
        A ``ragas.EvaluationDataset`` instance ready for ``evaluate()``.

    Raises:
        ImportError: If ``ragas`` is not installed.
    """
    try:
        from ragas import EvaluationDataset
    except ImportError as exc:
        raise ImportError("Install ragas: pip install traceops[ragas]") from exc

    from trace_ops._types import EventType

    entries: list[dict[str, Any]] = []
    for _path, trace in _load_traces(cassette_dir, include_patterns):
        llm_responses = [
            e for e in trace.events if e.event_type == EventType.LLM_RESPONSE
        ]
        first_response = ""
        if llm_responses:
            r = llm_responses[0].response or {}
            choices = r.get("choices") or []
            if choices:
                first_response = (choices[0].get("message") or {}).get("content", "") or ""

        for retrieval in trace.retrieval_events:
            chunks = [
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
                for c in (retrieval.chunks or [])
            ]
            entries.append({
                "user_input": retrieval.query or "",
                "retrieved_contexts": chunks,
                "response": first_response,
            })

    return EvaluationDataset.from_list(entries)


def to_deepeval_dataset(
    cassette_dir: str | Path,
    include_patterns: list[str] | None = None,
) -> list[Any]:
    """Export cassettes to a list of DeepEval ``LLMTestCase`` objects.

    Args:
        cassette_dir: Directory containing cassette YAML files.
        include_patterns: Glob patterns for cassette files.

    Returns:
        List of ``deepeval.test_case.LLMTestCase`` instances.

    Raises:
        ImportError: If ``deepeval`` is not installed.
    """
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise ImportError("Install deepeval: pip install traceops[deepeval]") from exc

    from trace_ops._types import EventType

    test_cases: list[Any] = []
    for _path, trace in _load_traces(cassette_dir, include_patterns):
        llm_responses = [
            e for e in trace.events if e.event_type == EventType.LLM_RESPONSE
        ]
        first_response = ""
        if llm_responses:
            r = llm_responses[0].response or {}
            choices = r.get("choices") or []
            if choices:
                first_response = (choices[0].get("message") or {}).get("content", "") or ""

        for retrieval in trace.retrieval_events:
            chunks = [
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
                for c in (retrieval.chunks or [])
            ]
            test_cases.append(LLMTestCase(
                input=retrieval.query or "",
                actual_output=first_response,
                retrieval_context=chunks,
            ))

    return test_cases


def to_csv(
    cassette_dir: str | Path,
    output_path: str | Path,
    include_patterns: list[str] | None = None,
) -> Path:
    """Export cassettes to a CSV file for spreadsheet / notebook analysis.

    Each row represents one retrieval event with the associated LLM response
    and any cached RAG scores.

    Args:
        cassette_dir: Directory containing cassette YAML files.
        output_path: Path to write the output CSV.
        include_patterns: Glob patterns for cassette files.

    Returns:
        Path to the written CSV file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cassette",
            "query",
            "num_chunks",
            "top_score",
            "mean_score",
            "response_preview",
            "faithfulness",
            "context_precision",
            "answer_relevancy",
        ])

        from trace_ops._types import EventType

        for cassette_path, trace in _load_traces(cassette_dir, include_patterns):
            scores = trace.rag_scores or {}

            llm_responses = [
                e for e in trace.events if e.event_type == EventType.LLM_RESPONSE
            ]
            first_response = ""
            if llm_responses:
                r = llm_responses[0].response or {}
                choices = r.get("choices") or []
                if choices:
                    first_response = (
                        (choices[0].get("message") or {}).get("content", "") or ""
                    )[:200]

            for retrieval in trace.retrieval_events:
                chunk_scores = [
                    float(c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0))
                    for c in (retrieval.chunks or [])
                ]
                writer.writerow([
                    cassette_path.name,
                    retrieval.query or "",
                    len(retrieval.chunks or []),
                    max(chunk_scores) if chunk_scores else "",
                    (sum(chunk_scores) / len(chunk_scores)) if chunk_scores else "",
                    first_response,
                    scores.get("faithfulness", ""),
                    scores.get("context_precision", ""),
                    scores.get("answer_relevancy", ""),
                ])

    return out


def to_openai_finetune(
    cassette_dir: str | Path,
    output_path: str | Path,
    include_patterns: list[str] | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> Path:
    """Export cassettes to OpenAI fine-tuning JSONL format.

    Each cassette becomes one training example:
    ``{"messages": [system, user, assistant]}``.

    Args:
        cassette_dir: Directory containing cassette YAML files.
        output_path: Path to write the output JSONL file.
        include_patterns: Glob patterns for cassette files.
        system_prompt: Default system message if none is recorded.

    Returns:
        Path to the written JSONL file.
    """
    import json

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    from trace_ops._types import EventType

    with open(out, "w", encoding="utf-8") as f:
        for _path, trace in _load_traces(cassette_dir, include_patterns):
            llm_requests = [
                e for e in trace.events if e.event_type == EventType.LLM_REQUEST
            ]
            llm_responses = [
                e for e in trace.events if e.event_type == EventType.LLM_RESPONSE
            ]
            if not llm_requests or not llm_responses:
                continue

            messages = list(llm_requests[0].messages or [])
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            # Add assistant response
            resp = llm_responses[0].response or {}
            choices = resp.get("choices") or []
            content = ""
            if choices:
                content = (choices[0].get("message") or {}).get("content", "") or ""
            messages.append({"role": "assistant", "content": content})

            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    return out
