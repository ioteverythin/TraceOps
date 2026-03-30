"""Tests for trace_ops.analysis — PatternDetector, GapAnalyzer, SkillsGenerator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_ops._types import EventType, Trace, TraceEvent, TraceMetadata
from trace_ops.analysis import (
    BehavioralGap,
    GapAnalyzer,
    GapReport,
    PatternDetector,
    PatternReport,
    SkillsGenerator,
)
from trace_ops.cassette import save_cassette


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_trace(
    tools: list[str] | None = None,
    model: str = "gpt-4o",
    tokens: int = 100,
    cost: float = 0.001,
    has_error: bool = False,
    llm_calls: int = 1,
) -> Trace:
    trace = Trace(metadata=TraceMetadata(description="test"))
    for _ in range(llm_calls):
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            model=model,
            messages=[{"role": "user", "content": "test"}],
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            model=model,
            input_tokens=tokens // 2,
            output_tokens=tokens // 2,
            cost_usd=cost / llm_calls,
        ))
    for tool in (tools or []):
        trace.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name=tool))
        trace.add_event(TraceEvent(event_type=EventType.TOOL_RESULT, tool_name=tool, tool_output="ok"))
    if has_error:
        trace.add_event(TraceEvent(
            event_type=EventType.ERROR,
            error_type="RateLimitError",
            error_message="quota exceeded",
        ))
    trace.finalize()
    return trace


# ── PatternDetector tests ──────────────────────────────────────────────────────


class TestPatternDetector:
    def test_empty_traces_returns_zero_report(self):
        det = PatternDetector()
        report = det.analyze([])
        assert report.cassette_count == 0
        assert report.total_events == 0
        assert report.top_tool_sequences == []
        assert report.model_usage == []

    def test_single_trace_basic_stats(self):
        t = _make_trace(tools=["search", "read"], tokens=200, cost=0.002)
        det = PatternDetector(window_size=2)
        report = det.analyze([("test.yaml", t)])
        assert report.cassette_count == 1
        assert report.avg_llm_calls == 1.0
        assert report.avg_tokens == 200
        assert report.tool_frequency["search"] == 1
        assert report.tool_frequency["read"] == 1

    def test_tool_sequence_detection(self):
        # Three traces all use search → read → write sequence
        traces = []
        for i in range(3):
            t = _make_trace(tools=["search", "read", "write"])
            traces.append((f"trace_{i}.yaml", t))
        det = PatternDetector(window_size=3)
        report = det.analyze(traces)
        sequences = [tuple(s.sequence) for s in report.top_tool_sequences]
        assert ("search", "read", "write") in sequences
        # Count for that sequence should be 3
        for seq in report.top_tool_sequences:
            if tuple(seq.sequence) == ("search", "read", "write"):
                assert seq.count == 3

    def test_model_stats_aggregation(self):
        traces = [
            ("a.yaml", _make_trace(model="gpt-4o", tokens=100, cost=0.001)),
            ("b.yaml", _make_trace(model="gpt-4o", tokens=200, cost=0.002)),
            ("c.yaml", _make_trace(model="gpt-4o-mini", tokens=50, cost=0.0005)),
        ]
        det = PatternDetector()
        report = det.analyze(traces)
        models = {m.model: m for m in report.model_usage}
        assert "gpt-4o" in models
        assert models["gpt-4o"].call_count == 2
        assert models["gpt-4o"].total_tokens == 300

    def test_error_counting(self):
        traces = [
            ("a.yaml", _make_trace(has_error=True)),
            ("b.yaml", _make_trace(has_error=True)),
            ("c.yaml", _make_trace(has_error=False)),
        ]
        det = PatternDetector()
        report = det.analyze(traces)
        error_names = [e for e, _ in report.most_common_errors]
        assert "RateLimitError" in error_names

    def test_summary_string(self):
        t = _make_trace(tools=["search", "read", "write"])
        det = PatternDetector(window_size=2)
        report = det.analyze([("t.yaml", t)])
        summary = report.summary()
        assert "1 traces" in summary
        assert "Avg per trace" in summary

    def test_to_dict_keys(self):
        t = _make_trace(tools=["search"])
        det = PatternDetector()
        report = det.analyze([("t.yaml", t)])
        d = report.to_dict()
        for key in ["cassette_count", "total_events", "avg_llm_calls",
                    "avg_tokens", "avg_cost_usd", "top_tool_sequences",
                    "model_usage", "most_common_errors", "tool_frequency"]:
            assert key in d

    def test_analyze_dir(self, tmp_path: Path):
        for i in range(3):
            t = _make_trace(tools=["search"], model="gpt-4o", tokens=100)
            save_cassette(t, tmp_path / f"trace_{i}.yaml")
        det = PatternDetector()
        report = det.analyze_dir(tmp_path)
        assert report.cassette_count == 3

    def test_analyze_dir_empty(self, tmp_path: Path):
        det = PatternDetector()
        report = det.analyze_dir(tmp_path)
        assert report.cassette_count == 0

    def test_avg_calculations(self):
        traces = [
            ("a.yaml", _make_trace(tokens=100, cost=0.001, llm_calls=1)),
            ("b.yaml", _make_trace(tokens=100, cost=0.001, llm_calls=3)),
        ]
        det = PatternDetector()
        report = det.analyze(traces)
        # trace a: 1 call × 100 tokens = 100; trace b: 3 calls × 100 tokens = 300; avg = 200
        assert report.avg_tokens == pytest.approx(200, abs=1)
        assert report.avg_llm_calls == pytest.approx(2.0)

    def test_window_size_1_no_sequences(self):
        t = _make_trace(tools=["search"])
        det = PatternDetector(window_size=2)
        report = det.analyze([("t.yaml", t)])
        # Only 1 tool, can't form a window-2 sequence
        assert report.top_tool_sequences == []

    def test_multiple_models_sorted_by_call_count(self):
        traces = [
            ("a.yaml", _make_trace(model="gpt-4o", llm_calls=3)),
            ("b.yaml", _make_trace(model="gpt-4o-mini", llm_calls=1)),
        ]
        det = PatternDetector()
        report = det.analyze(traces)
        assert report.model_usage[0].model == "gpt-4o"


# ── GapAnalyzer tests ──────────────────────────────────────────────────────────


class TestGapAnalyzer:
    def test_empty_inputs_return_empty_report(self):
        analyzer = GapAnalyzer()
        report = analyzer.compare([], [])
        assert len(report.gaps) == 0

    def test_no_gaps_when_identical(self):
        traces = [("a.yaml", _make_trace(tools=["search"], tokens=100, cost=0.001))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(traces, traces)
        # With identical inputs, no inflation gaps should appear
        inflation_gaps = [g for g in report.gaps if "inflation" in g.category]
        assert len(inflation_gaps) == 0

    def test_token_inflation_detected(self):
        golden = [("g.yaml", _make_trace(tokens=100, cost=0.001))]
        agent = [("a.yaml", _make_trace(tokens=500, cost=0.005))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "token_inflation" in cats

    def test_cost_inflation_detected(self):
        golden = [("g.yaml", _make_trace(tokens=50, cost=0.0001))]
        agent = [("a.yaml", _make_trace(tokens=50, cost=0.0005))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "cost_inflation" in cats

    def test_missing_tool_detected(self):
        golden = [
            ("g1.yaml", _make_trace(tools=["search", "read"])),
            ("g2.yaml", _make_trace(tools=["search", "read"])),
        ]
        # Agent never uses "read"
        agent = [
            ("a1.yaml", _make_trace(tools=["search"])),
            ("a2.yaml", _make_trace(tools=["search"])),
        ]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "missing_tool" in cats
        missing = next(g for g in report.gaps if g.category == "missing_tool")
        assert "read" in missing.description

    def test_extra_tool_detected(self):
        golden = [
            ("g1.yaml", _make_trace(tools=["search"])),
            ("g2.yaml", _make_trace(tools=["search"])),
        ]
        # Agent always uses extra tool "write"
        agent = [
            ("a1.yaml", _make_trace(tools=["search", "write"])),
            ("a2.yaml", _make_trace(tools=["search", "write"])),
        ]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "extra_tool" in cats

    def test_model_mismatch_detected(self):
        golden = [("g.yaml", _make_trace(model="gpt-4o"))]
        agent = [("a.yaml", _make_trace(model="gpt-4o-mini"))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "model_mismatch" in cats

    def test_error_rate_gap_detected(self):
        golden = [
            ("g1.yaml", _make_trace(has_error=False)),
            ("g2.yaml", _make_trace(has_error=False)),
        ]
        # Agent has 100% error rate
        agent = [
            ("a1.yaml", _make_trace(has_error=True)),
            ("a2.yaml", _make_trace(has_error=True)),
        ]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "error_rate" in cats

    def test_llm_call_inflation(self):
        golden = [("g.yaml", _make_trace(llm_calls=1))]
        agent = [("a.yaml", _make_trace(llm_calls=5))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        cats = [g.category for g in report.gaps]
        assert "llm_call_inflation" in cats

    def test_gaps_sorted_critical_first(self):
        golden = [("g.yaml", _make_trace(has_error=False, tokens=10, cost=0.00001))]
        agent = [("a.yaml", _make_trace(has_error=True, tokens=1000, cost=0.01))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        if len(report.gaps) >= 2:
            sev_order = {"critical": 0, "warning": 1, "info": 2}
            for i in range(len(report.gaps) - 1):
                assert sev_order[report.gaps[i].severity] <= sev_order[report.gaps[i + 1].severity]

    def test_summary_no_gaps(self):
        traces = [("a.yaml", _make_trace())]
        analyzer = GapAnalyzer()
        report = analyzer.compare(traces, traces)
        # No inflation in identical traces
        assert "gaps" not in report.summary().lower() or "0 behavioral" in report.summary() or "No significant" in report.summary()

    def test_summary_with_gaps(self):
        golden = [("g.yaml", _make_trace(tokens=50))]
        agent = [("a.yaml", _make_trace(tokens=500))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        if report.gaps:
            summary = report.summary()
            assert "gap" in summary.lower()

    def test_to_dict_structure(self):
        golden = [("g.yaml", _make_trace(tokens=50))]
        agent = [("a.yaml", _make_trace(tokens=500))]
        analyzer = GapAnalyzer()
        report = analyzer.compare(golden, agent)
        d = report.to_dict()
        for key in ["golden_count", "agent_count", "gap_count", "critical_count",
                    "warning_count", "gaps"]:
            assert key in d

    def test_behavioral_gap_to_dict(self):
        gap = BehavioralGap(
            category="token_inflation",
            description="test",
            severity="warning",
            frequency=0.8,
            golden_value=100,
            agent_value=500,
        )
        d = gap.to_dict()
        assert d["category"] == "token_inflation"
        assert d["severity"] == "warning"
        assert d["frequency"] == pytest.approx(0.8)

    def test_critical_count_property(self):
        report = GapReport(
            golden_count=1,
            agent_count=1,
            gaps=[
                BehavioralGap("error_rate", "test", "critical", 1.0),
                BehavioralGap("token_inflation", "test", "warning", 1.0),
                BehavioralGap("model_mismatch", "test", "info", 1.0),
            ],
        )
        assert report.critical_count == 1
        assert report.warning_count == 1


# ── SkillsGenerator tests ──────────────────────────────────────────────────────


class TestSkillsGenerator:
    def test_from_gap_report_no_gaps(self):
        report = GapReport(golden_count=3, agent_count=3, gaps=[])
        gen = SkillsGenerator()
        md = gen.from_gap_report(report)
        assert "No Significant Gaps Found" in md
        assert "TraceOps" in md

    def test_from_gap_report_with_gaps(self):
        report = GapReport(
            golden_count=2,
            agent_count=2,
            gaps=[
                BehavioralGap(
                    category="token_inflation",
                    description="Agent uses 3x more tokens",
                    severity="critical",
                    frequency=1.0,
                    golden_value=100,
                    agent_value=300,
                ),
                BehavioralGap(
                    category="missing_tool",
                    description="Agent under-uses 'read'",
                    severity="warning",
                    frequency=0.5,
                ),
            ],
        )
        gen = SkillsGenerator()
        md = gen.from_gap_report(report)
        assert "CRITICAL" in md
        assert "WARNING" in md
        assert "Token Usage" in md or "token_inflation" in md.lower() or "⚠️" in md
        assert "golden_value" not in md  # should be rendered, not raw key
        assert "100" in md  # golden value should appear
        assert "300" in md  # agent value should appear

    def test_from_gap_report_writes_file(self, tmp_path: Path):
        report = GapReport(golden_count=1, agent_count=1, gaps=[])
        gen = SkillsGenerator()
        out = tmp_path / "AGENTS.md"
        gen.from_gap_report(report, output_path=out)
        assert out.exists()
        content = out.read_text()
        assert "TraceOps" in content

    def test_from_pattern_report(self):
        t = _make_trace(tools=["search", "read", "write"], model="gpt-4o", tokens=150, cost=0.002)
        det = PatternDetector(window_size=2)
        report = det.analyze([("t.yaml", t)])
        gen = SkillsGenerator()
        md = gen.from_pattern_report(report)
        assert "Pattern Summary" in md
        assert "gpt-4o" in md
        assert "LLM calls" in md

    def test_from_pattern_report_with_errors(self):
        traces = [
            ("a.yaml", _make_trace(has_error=True)),
            ("b.yaml", _make_trace(has_error=True)),
        ]
        det = PatternDetector()
        report = det.analyze(traces)
        gen = SkillsGenerator()
        md = gen.from_pattern_report(report)
        assert "Errors" in md or "RateLimitError" in md

    def test_from_pattern_report_writes_file(self, tmp_path: Path):
        t = _make_trace()
        det = PatternDetector()
        report = det.analyze([("t.yaml", t)])
        gen = SkillsGenerator()
        out = tmp_path / "SUMMARY.md"
        gen.from_pattern_report(report, output_path=out)
        assert out.exists()

    def test_custom_title(self):
        report = GapReport(golden_count=1, agent_count=1, gaps=[])
        gen = SkillsGenerator()
        md = gen.from_gap_report(report, title="My Custom Guidance")
        assert "My Custom Guidance" in md

    def test_frequency_shown_when_partial(self):
        report = GapReport(
            golden_count=1,
            agent_count=1,
            gaps=[
                BehavioralGap(
                    category="missing_tool",
                    description="test",
                    severity="warning",
                    frequency=0.6,  # partial — should be shown
                ),
            ],
        )
        gen = SkillsGenerator()
        md = gen.from_gap_report(report)
        assert "60%" in md

    def test_full_frequency_not_shown(self):
        report = GapReport(
            golden_count=1,
            agent_count=1,
            gaps=[
                BehavioralGap(
                    category="token_inflation",
                    description="test",
                    severity="warning",
                    frequency=1.0,  # full — should NOT show "Affects 100%"
                ),
            ],
        )
        gen = SkillsGenerator()
        md = gen.from_gap_report(report)
        assert "Affects 100%" not in md
