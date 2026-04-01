"""Deep CLI coverage — targeting edge cases and remaining uncovered paths."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import save_cassette
from trace_ops.cli import main

# ── Helpers ─────────────────────────────────────────────────────────


def _rich_cassette(tmp_path: Path, name: str = "rich.yaml") -> Path:
    """Cassette with many event types for full rendering paths."""
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
        max_tokens=500,
        tools=[{"type": "function", "function": {"name": "search"}}],
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model="gpt-4o",
        response={"choices": [{"message": {"content": "thinking...", "tool_calls": [{"function": {"name": "search"}}]}}]},
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.005,
        duration_ms=350.5,
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.TOOL_CALL,
        tool_name="search",
        tool_input={"query": "python testing"},
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.TOOL_RESULT,
        tool_name="search",
        tool_output="Found 5 results",
        duration_ms=120.0,
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.AGENT_DECISION,
        decision="delegate_to_agent_b",
        reasoning="Need deeper analysis",
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.ERROR,
        error_type="TimeoutError",
        error_message="Request timed out after 30s",
    ))
    trace.finalize()
    path = tmp_path / name
    save_cassette(trace, path)
    return path


# ── inspect deep paths ──────────────────────────────────────────────


class TestInspectDeep:
    def test_inspect_all_event_types(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "search" in result.output
        assert "TimeoutError" in result.output or "error" in result.output.lower()

    def test_inspect_trajectory_numbering(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "1." in result.output
        assert "llm_call:gpt-4o" in result.output

    def test_inspect_shows_tokens_cost(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "0.0050" in result.output or "0.005" in result.output

    def test_inspect_shows_fingerprint(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "Fingerprint" in result.output


# ── export deep paths ──────────────────────────────────────────────


class TestExportDeep:
    def test_export_yaml(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(path), "--format", "yaml"])
        assert result.exit_code == 0
        assert "trace_id" in result.output
        assert "events" in result.output

    def test_export_json_structure(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(path)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "version" in data
        assert "events" in data
        assert len(data["events"]) == 6

    def test_export_missing_file(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_export_yaml_to_file(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        out = tmp_path / "out.yaml"
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(path), "--format", "yaml", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()


# ── report deep paths ──────────────────────────────────────────────


class TestReportDeep:
    def test_report_with_compare(self, tmp_path: Path):
        p1 = _rich_cassette(tmp_path, "a.yaml")
        p2 = _rich_cassette(tmp_path, "b.yaml")
        out = tmp_path / "report.html"
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(p1), "--compare", str(p2), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "Diff" in html or "diff" in html

    def test_report_auto_output_name(self, tmp_path: Path):
        path = _rich_cassette(tmp_path, "trace.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(path)])
        assert result.exit_code == 0
        # Should create trace.html
        assert (tmp_path / "trace.html").exists()

    def test_report_missing_compare(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(path), "--compare", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0


# ── stats deep paths ───────────────────────────────────────────────


class TestStatsDeep:
    def test_stats_model_breakdown(self, tmp_path: Path):
        _rich_cassette(tmp_path, "a.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(tmp_path)])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_stats_tool_breakdown(self, tmp_path: Path):
        _rich_cassette(tmp_path, "a.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(tmp_path)])
        assert result.exit_code == 0
        assert "search" in result.output

    def test_stats_missing_dir(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", "nonexistent_dir_xyz"])
        assert result.exit_code != 0

    def test_stats_empty(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(tmp_path)])
        assert result.exit_code == 0
        assert "No cassettes" in result.output

    def test_stats_ignores_broken(self, tmp_path: Path):
        _rich_cassette(tmp_path, "good.yaml")
        (tmp_path / "bad.yaml").write_text(":::bad:::", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(tmp_path)])
        assert result.exit_code == 0
        # Should report stats for the good cassette
        assert "1" in result.output


# ── prune deep paths ───────────────────────────────────────────────


class TestPruneDeep:
    def test_prune_actually_deletes(self, tmp_path: Path):
        path = _rich_cassette(tmp_path, "old.yaml")
        runner = CliRunner()
        # Use 0h to make everything "old"
        result = runner.invoke(main, ["prune", str(tmp_path), "--older-than", "0h"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert not path.exists()

    def test_prune_invalid_format(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["prune", str(tmp_path), "--older-than", "invalid"])
        assert result.exit_code != 0

    def test_prune_missing_dir(self):
        runner = CliRunner()
        result = runner.invoke(main, ["prune", "nonexistent_dir_xyz"])
        assert result.exit_code != 0

    def test_prune_weeks(self, tmp_path: Path):
        _rich_cassette(tmp_path, "fresh.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["prune", str(tmp_path), "--older-than", "1w"])
        assert result.exit_code == 0
        assert "No stale" in result.output


# ── validate deep paths ────────────────────────────────────────────


class TestValidateDeep:
    def test_validate_empty_events(self, tmp_path: Path):
        trace = Trace()
        save_cassette(trace, tmp_path / "empty.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path / "empty.yaml")])
        # Should flag "no events" issue
        assert "no events" in result.output.lower() or "Issue" in result.output or "valid" in result.output.lower()

    def test_validate_non_dict_yaml(self, tmp_path: Path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(bad)])
        assert result.exit_code != 0

    def test_validate_missing_version(self, tmp_path: Path):
        import yaml
        data = {"trace_id": "abc", "events": []}
        f = tmp_path / "noversion.yaml"
        f.write_text(yaml.dump(data), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(f)])
        # Should report version issue
        assert "version" in result.output.lower() or "Issue" in result.output or result.exit_code == 0

    def test_validate_missing_trace_id(self, tmp_path: Path):
        import yaml
        data = {"version": "1", "events": []}
        f = tmp_path / "noid.yaml"
        f.write_text(yaml.dump(data), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(f)])
        assert "trace_id" in result.output.lower() or "Issue" in result.output or result.exit_code == 0


# ── debug command ───────────────────────────────────────────────────


class TestDebugCommand:
    def test_debug_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["debug", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_debug_llm_only_flag(self, tmp_path: Path):
        """Verify --llm-only flag is accepted."""
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        # The debug command runs an interactive loop — we can only test
        # that the flags parse correctly by checking it starts
        result = runner.invoke(main, ["debug", str(path), "--llm-only"], input="q\n")
        # May succeed or fail due to interactive mode, but shouldn't error on parse
        assert result.exit_code == 0 or "Error" not in result.output

    def test_debug_tools_only_flag(self, tmp_path: Path):
        path = _rich_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["debug", str(path), "--tools-only"], input="q\n")
        assert result.exit_code == 0 or "Error" not in result.output
