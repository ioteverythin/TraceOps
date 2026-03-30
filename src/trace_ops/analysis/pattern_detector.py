"""Multi-trace behavioral pattern detection.

Finds recurring tool sequences, model usage, error patterns, and cost/token
averages across a collection of cassettes — inspired by agent-pr-replay's
stats module which surfaces how coding agents navigate codebases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._types import EventType, Trace
from ..cassette import load_cassette


@dataclass
class ToolPattern:
    """A recurring tool-call n-gram observed across multiple traces."""

    sequence: tuple[str, ...]
    count: int
    cassettes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": list(self.sequence),
            "count": self.count,
            "cassettes": self.cassettes,
        }


@dataclass
class ModelStat:
    """Per-model usage statistics aggregated across all traces."""

    model: str
    call_count: int
    total_tokens: int
    total_cost_usd: float
    cassettes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cassettes": self.cassettes,
        }


@dataclass
class PatternReport:
    """Results of multi-trace behavioral pattern analysis."""

    cassette_count: int
    total_events: int
    top_tool_sequences: list[ToolPattern]
    model_usage: list[ModelStat]
    most_common_errors: list[tuple[str, int]]
    avg_llm_calls: float
    avg_tokens: float
    avg_cost_usd: float
    tool_frequency: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cassette_count": self.cassette_count,
            "total_events": self.total_events,
            "avg_llm_calls": round(self.avg_llm_calls, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "top_tool_sequences": [t.to_dict() for t in self.top_tool_sequences],
            "model_usage": [m.to_dict() for m in self.model_usage],
            "most_common_errors": [
                {"error": e, "count": c} for e, c in self.most_common_errors
            ],
            "tool_frequency": self.tool_frequency,
        }

    def summary(self) -> str:
        lines = [
            f"Analyzed {self.cassette_count} traces ({self.total_events} events total).",
            (
                f"Avg per trace: {self.avg_llm_calls:.1f} LLM calls, "
                f"{self.avg_tokens:.0f} tokens, ${self.avg_cost_usd:.5f}"
            ),
        ]
        if self.top_tool_sequences:
            top = self.top_tool_sequences[0]
            lines.append(
                f"Most common tool sequence: {' → '.join(top.sequence)} ({top.count}×)"
            )
        if self.most_common_errors:
            err, cnt = self.most_common_errors[0]
            lines.append(f"Most common error: {err} ({cnt}×)")
        return "\n".join(lines)


class PatternDetector:
    """Analyze multiple traces to surface recurring behavioral patterns.

    Inspired by agent-pr-replay's ``stats`` module — adapted for TraceOps
    cassettes instead of Claude Code session files.

    Example::

        detector = PatternDetector(window_size=2)
        report = detector.analyze_dir(Path("cassettes/"))
        print(report.summary())
    """

    def __init__(self, window_size: int = 3, top_n: int = 10) -> None:
        self.window_size = window_size
        self.top_n = top_n

    def analyze(self, traces: list[tuple[str, Trace]]) -> PatternReport:
        """Analyze a list of ``(name, trace)`` pairs and return a PatternReport."""
        if not traces:
            return PatternReport(
                cassette_count=0,
                total_events=0,
                top_tool_sequences=[],
                model_usage=[],
                most_common_errors=[],
                avg_llm_calls=0.0,
                avg_tokens=0.0,
                avg_cost_usd=0.0,
                tool_frequency={},
            )

        seq_counter: Counter[tuple[str, ...]] = Counter()
        seq_cassettes: dict[tuple[str, ...], list[str]] = {}
        model_calls: Counter[str] = Counter()
        model_tokens: Counter[str] = Counter()
        model_costs: dict[str, float] = {}
        model_cassettes: dict[str, list[str]] = {}
        error_counter: Counter[str] = Counter()
        tool_freq: Counter[str] = Counter()

        total_events = 0
        total_llm_calls = 0
        total_tokens = 0
        total_cost = 0.0

        for name, trace in traces:
            total_events += len(trace.events)
            total_llm_calls += trace.total_llm_calls
            total_tokens += trace.total_tokens
            total_cost += trace.total_cost_usd

            # ── Tool call n-grams ────────────────────────────────────────
            tool_calls = [
                e.tool_name or "unknown"
                for e in trace.events
                if e.event_type == EventType.TOOL_CALL and e.tool_name
            ]
            tool_freq.update(tool_calls)

            for i in range(len(tool_calls) - self.window_size + 1):
                seq = tuple(tool_calls[i : i + self.window_size])
                seq_counter[seq] += 1
                seq_cassettes.setdefault(seq, [])
                if name not in seq_cassettes[seq]:
                    seq_cassettes[seq].append(name)

            # ── Model stats ──────────────────────────────────────────────
            for e in trace.events:
                if e.event_type == EventType.LLM_RESPONSE and e.model:
                    model_calls[e.model] += 1
                    model_tokens[e.model] += (e.input_tokens or 0) + (e.output_tokens or 0)
                    model_costs[e.model] = model_costs.get(e.model, 0.0) + (e.cost_usd or 0.0)
                    model_cassettes.setdefault(e.model, [])
                    if name not in model_cassettes[e.model]:
                        model_cassettes[e.model].append(name)
                elif e.event_type == EventType.ERROR and e.error_type:
                    error_counter[e.error_type] += 1

        n = len(traces)
        top_seqs = [
            ToolPattern(
                sequence=seq,
                count=cnt,
                cassettes=seq_cassettes.get(seq, []),
            )
            for seq, cnt in seq_counter.most_common(self.top_n)
        ]
        model_stats = [
            ModelStat(
                model=m,
                call_count=model_calls[m],
                total_tokens=model_tokens[m],
                total_cost_usd=model_costs.get(m, 0.0),
                cassettes=model_cassettes.get(m, []),
            )
            for m in sorted(model_calls, key=lambda x: -model_calls[x])
        ]

        return PatternReport(
            cassette_count=n,
            total_events=total_events,
            top_tool_sequences=top_seqs,
            model_usage=model_stats,
            most_common_errors=error_counter.most_common(self.top_n),
            avg_llm_calls=total_llm_calls / n,
            avg_tokens=total_tokens / n,
            avg_cost_usd=total_cost / n,
            tool_frequency=dict(tool_freq.most_common()),
        )

    def analyze_dir(self, cassette_dir: Path) -> PatternReport:
        """Load all cassettes from a directory and analyze them."""
        cassette_dir = Path(cassette_dir)
        pairs: list[tuple[str, Trace]] = []
        for path in sorted(cassette_dir.rglob("*.yaml")):
            try:
                trace = load_cassette(path)
                pairs.append((path.name, trace))
            except Exception:
                continue
        return self.analyze(pairs)
