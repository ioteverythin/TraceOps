"""Tests for cost dashboard reporter and CLI costs command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import save_cassette
from trace_ops.cli import main
from trace_ops.reporters.cost_dashboard import (
    CassetteCost,
    CostDashboard,
    CostSummary,
    ModelCost,
)

# ── Helpers ─────────────────────────────────────────────────────────


def _make_cassette(
    tmp_path: Path,
    name: str,
    *,
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost: float = 0.01,
    num_calls: int = 1,
) -> Path:
    trace = Trace()
    for _ in range(num_calls):
        trace.add_event(
            TraceEvent(
                event_type=EventType.LLM_REQUEST,
                provider="openai",
                model=model,
                messages=[{"role": "user", "content": "hello"}],
            )
        )
        trace.add_event(
            TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                provider="openai",
                model=model,
                response={"choices": [{"message": {"content": "hi"}}]},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        )
    trace.finalize()
    path = tmp_path / name
    save_cassette(trace, path)
    return path


# ── CostSummary dataclass ──────────────────────────────────────────


class TestCostSummary:
    def test_to_dict_empty(self):
        s = CostSummary()
        d = s.to_dict()
        assert d["cassette_count"] == 0
        assert d["total_cost_usd"] == 0.0
        assert d["by_model"] == []
        assert d["by_cassette"] == []
        assert d["errors"] == []

    def test_to_dict_with_data(self):
        s = CostSummary(
            cassette_count=2,
            total_llm_calls=5,
            total_tokens=1000,
            total_cost_usd=0.05,
            by_model=[
                ModelCost(model="gpt-4o", calls=3, input_tokens=600, output_tokens=200, total_tokens=800, cost_usd=0.04),
                ModelCost(model="claude-3", calls=2, input_tokens=100, output_tokens=100, total_tokens=200, cost_usd=0.01),
            ],
            by_cassette=[
                CassetteCost(path="a.yaml", llm_calls=3, total_tokens=800, cost_usd=0.04, models=["gpt-4o"]),
                CassetteCost(path="b.yaml", llm_calls=2, total_tokens=200, cost_usd=0.01, models=["claude-3"]),
            ],
        )
        d = s.to_dict()
        assert d["cassette_count"] == 2
        assert len(d["by_model"]) == 2
        assert d["by_model"][0]["model"] == "gpt-4o"
        assert len(d["by_cassette"]) == 2


# ── CostDashboard ──────────────────────────────────────────────────


class TestCostDashboard:
    def test_empty_directory(self, tmp_path: Path):
        dashboard = CostDashboard(tmp_path)
        s = dashboard.data
        assert s.cassette_count == 0
        assert s.total_cost_usd == 0.0

    def test_single_cassette(self, tmp_path: Path):
        _make_cassette(tmp_path, "test.yaml", cost=0.05, input_tokens=200, output_tokens=100)
        dashboard = CostDashboard(tmp_path)
        s = dashboard.data
        assert s.cassette_count == 1
        assert s.total_llm_calls == 1
        assert s.total_cost_usd == pytest.approx(0.05, abs=0.001)
        assert len(s.by_model) == 1
        assert s.by_model[0].model == "gpt-4o"

    def test_multiple_cassettes(self, tmp_path: Path):
        _make_cassette(tmp_path, "a.yaml", model="gpt-4o", cost=0.02, num_calls=2)
        _make_cassette(tmp_path, "b.yaml", model="claude-3", cost=0.03, num_calls=1)
        dashboard = CostDashboard(tmp_path)
        s = dashboard.data
        assert s.cassette_count == 2
        assert s.total_llm_calls == 3
        assert len(s.by_model) == 2
        # Sorted by cost descending
        assert s.by_model[0].cost_usd >= s.by_model[1].cost_usd

    def test_nested_cassettes(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _make_cassette(tmp_path, "root.yaml")
        _make_cassette(sub, "nested.yaml")
        dashboard = CostDashboard(tmp_path)
        assert dashboard.data.cassette_count == 2

    def test_broken_cassette(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("{invalid yaml: [unclosed", encoding="utf-8")
        _make_cassette(tmp_path, "good.yaml")
        dashboard = CostDashboard(tmp_path)
        s = dashboard.data
        assert s.cassette_count == 1
        assert len(s.errors) == 1

    def test_caches_result(self, tmp_path: Path):
        _make_cassette(tmp_path, "test.yaml")
        dashboard = CostDashboard(tmp_path)
        s1 = dashboard.data
        s2 = dashboard.data
        assert s1 is s2

    def test_yml_extension(self, tmp_path: Path):
        trace = Trace()
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        ))
        trace.finalize()
        save_cassette(trace, tmp_path / "test.yml")
        dashboard = CostDashboard(tmp_path)
        assert dashboard.data.cassette_count == 1


class TestCostDashboardPrint:
    def test_print_no_errors(self, tmp_path: Path):
        _make_cassette(tmp_path, "test.yaml", cost=0.02)
        dashboard = CostDashboard(tmp_path)
        con = Console(force_terminal=True, no_color=True, width=120)
        # Should not raise
        dashboard.print(console=con)

    def test_print_with_errors(self, tmp_path: Path):
        (tmp_path / "bad.yaml").write_text(":::", encoding="utf-8")
        dashboard = CostDashboard(tmp_path)
        con = Console(force_terminal=True, no_color=True, width=120)
        dashboard.print(console=con)

    def test_print_empty(self, tmp_path: Path):
        dashboard = CostDashboard(tmp_path)
        con = Console(force_terminal=True, no_color=True, width=120)
        dashboard.print(console=con)

    def test_print_custom_top(self, tmp_path: Path):
        for i in range(5):
            _make_cassette(tmp_path, f"test_{i}.yaml", cost=0.01 * (i + 1))
        dashboard = CostDashboard(tmp_path)
        con = Console(force_terminal=True, no_color=True, width=120)
        dashboard.print(top=2, console=con)


# ── CLI costs command ───────────────────────────────────────────────


class TestCLICosts:
    def test_costs_basic(self, tmp_path: Path):
        _make_cassette(tmp_path, "a.yaml", cost=0.02)
        _make_cassette(tmp_path, "b.yaml", cost=0.03)
        runner = CliRunner()
        result = runner.invoke(main, ["costs", str(tmp_path)])
        assert result.exit_code == 0
        assert "Cost Dashboard" in result.output

    def test_costs_json(self, tmp_path: Path):
        _make_cassette(tmp_path, "a.yaml", cost=0.05)
        runner = CliRunner()
        result = runner.invoke(main, ["costs", str(tmp_path), "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data["cassette_count"] == 1
        assert data["total_cost_usd"] == pytest.approx(0.05, abs=0.001)

    def test_costs_sort_tokens(self, tmp_path: Path):
        _make_cassette(tmp_path, "a.yaml", input_tokens=500, output_tokens=200)
        _make_cassette(tmp_path, "b.yaml", input_tokens=100, output_tokens=50)
        runner = CliRunner()
        result = runner.invoke(main, ["costs", str(tmp_path), "--sort", "tokens"])
        assert result.exit_code == 0

    def test_costs_sort_calls(self, tmp_path: Path):
        _make_cassette(tmp_path, "a.yaml", num_calls=3)
        _make_cassette(tmp_path, "b.yaml", num_calls=1)
        runner = CliRunner()
        result = runner.invoke(main, ["costs", str(tmp_path), "--sort", "calls"])
        assert result.exit_code == 0

    def test_costs_top(self, tmp_path: Path):
        for i in range(5):
            _make_cassette(tmp_path, f"c_{i}.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["costs", str(tmp_path), "--top", "2"])
        assert result.exit_code == 0

    def test_costs_missing_dir(self):
        runner = CliRunner()
        result = runner.invoke(main, ["costs", "nonexistent_dir_xyz"])
        assert result.exit_code != 0

    def test_costs_empty_dir(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["costs", str(tmp_path)])
        assert result.exit_code == 0


# ── CLI diff enhancements ──────────────────────────────────────────


class TestCLIDiffEnhanced:
    def test_diff_detailed(self, tmp_path: Path):
        p1 = _make_cassette(tmp_path, "a.yaml", model="gpt-4o")
        p2 = _make_cassette(tmp_path, "b.yaml", model="claude-3")
        runner = CliRunner()
        result = runner.invoke(main, ["diff", str(p1), str(p2), "--detailed"])
        assert result.exit_code == 0
        assert "Per-Event" in result.output or "Trace Diff" in result.output

    def test_diff_output_html(self, tmp_path: Path):
        p1 = _make_cassette(tmp_path, "a.yaml")
        p2 = _make_cassette(tmp_path, "b.yaml", model="claude-3")
        out = tmp_path / "diff.html"
        runner = CliRunner()
        result = runner.invoke(main, ["diff", str(p1), str(p2), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "<html" in out.read_text(encoding="utf-8")

    def test_diff_identical_output(self, tmp_path: Path):
        p1 = _make_cassette(tmp_path, "a.yaml")
        p2 = _make_cassette(tmp_path, "b.yaml")
        out = tmp_path / "diff.html"
        runner = CliRunner()
        result = runner.invoke(main, ["diff", str(p1), str(p2), "-o", str(out)])
        assert result.exit_code == 0
        assert "identical" in result.output.lower()
        assert out.exists()

    def test_diff_missing_file(self, tmp_path: Path):
        p1 = _make_cassette(tmp_path, "a.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["diff", str(p1), str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_diff_detailed_added_events(self, tmp_path: Path):
        """When new trace has more events."""
        t1 = Trace()
        t1.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t1.finalize()
        save_cassette(t1, tmp_path / "a.yaml")

        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t2.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))
        t2.finalize()
        save_cassette(t2, tmp_path / "b.yaml")

        runner = CliRunner()
        result = runner.invoke(
            main, ["diff", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"), "--detailed"]
        )
        assert result.exit_code == 0

    def test_diff_detailed_removed_events(self, tmp_path: Path):
        """When new trace has fewer events."""
        t1 = Trace()
        t1.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t1.add_event(TraceEvent(event_type=EventType.TOOL_CALL, tool_name="search"))
        t1.finalize()
        save_cassette(t1, tmp_path / "a.yaml")

        t2 = Trace()
        t2.add_event(TraceEvent(event_type=EventType.LLM_REQUEST, model="gpt-4o"))
        t2.finalize()
        save_cassette(t2, tmp_path / "b.yaml")

        runner = CliRunner()
        result = runner.invoke(
            main, ["diff", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"), "--detailed"]
        )
        assert result.exit_code == 0
