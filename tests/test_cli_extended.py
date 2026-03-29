"""Tests for new v0.2 CLI commands."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import save_cassette
from trace_ops.cli import main


# ── Helpers ─────────────────────────────────────────────────────────


def _save_sample_cassette(tmp_path: Path, name: str = "test.yaml") -> Path:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        provider="openai",
        model="gpt-4o",
        response={"choices": [{"message": {"content": "hi"}}]},
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.TOOL_CALL,
        tool_name="search",
        tool_input={"q": "test"},
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.TOOL_RESULT,
        tool_name="search",
        tool_output="found it",
    ))
    trace.finalize()
    path = tmp_path / name
    save_cassette(trace, path)
    return path


# ── Version ─────────────────────────────────────────────────────────


class TestCLIVersion:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.5.0" in result.output


# ── Inspect ─────────────────────────────────────────────────────────


class TestCLIInspect:
    def test_inspect(self, tmp_path: Path):
        path = _save_sample_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_inspect_missing(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0


# ── Diff ────────────────────────────────────────────────────────────


class TestCLIDiff:
    def test_identical(self, tmp_path: Path):
        p1 = _save_sample_cassette(tmp_path, "a.yaml")
        p2 = _save_sample_cassette(tmp_path, "b.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["diff", str(p1), str(p2)])
        assert result.exit_code == 0
        assert "identical" in result.output.lower()


# ── Export ──────────────────────────────────────────────────────────


class TestCLIExport:
    def test_export_json(self, tmp_path: Path):
        path = _save_sample_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(path)])
        assert result.exit_code == 0
        assert "trace_id" in result.output

    def test_export_json_to_file(self, tmp_path: Path):
        path = _save_sample_cassette(tmp_path)
        out = tmp_path / "export.json"
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(path), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()


# ── Report ──────────────────────────────────────────────────────────


class TestCLIReport:
    def test_report(self, tmp_path: Path):
        path = _save_sample_cassette(tmp_path)
        out = tmp_path / "report.html"
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(path), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "<html" in html


# ── List ────────────────────────────────────────────────────────────


class TestCLIList:
    def test_ls(self, tmp_path: Path):
        _save_sample_cassette(tmp_path, "one.yaml")
        _save_sample_cassette(tmp_path, "two.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["ls", str(tmp_path)])
        assert result.exit_code == 0
        assert "one.yaml" in result.output
        assert "two.yaml" in result.output

    def test_ls_empty(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["ls", str(tmp_path)])
        assert result.exit_code == 0
        assert "No cassettes" in result.output

    def test_ls_missing_dir(self):
        runner = CliRunner()
        result = runner.invoke(main, ["ls", "nonexistent_dir_12345"])
        assert result.exit_code != 0


# ── Stats ───────────────────────────────────────────────────────────


class TestCLIStats:
    def test_stats(self, tmp_path: Path):
        _save_sample_cassette(tmp_path, "a.yaml")
        _save_sample_cassette(tmp_path, "b.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["stats", str(tmp_path)])
        assert result.exit_code == 0
        assert "Cassettes" in result.output
        assert "2" in result.output


# ── Prune ───────────────────────────────────────────────────────────


class TestCLIPrune:
    def test_prune_dry_run(self, tmp_path: Path):
        _save_sample_cassette(tmp_path, "old.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["prune", str(tmp_path), "--older-than", "0h", "--dry-run"])
        # 0h means everything is "old"
        assert result.exit_code == 0
        assert "Would delete" in result.output
        # File should still exist
        assert (tmp_path / "old.yaml").exists()

    def test_prune_nothing_old(self, tmp_path: Path):
        _save_sample_cassette(tmp_path, "fresh.yaml")
        runner = CliRunner()
        result = runner.invoke(main, ["prune", str(tmp_path), "--older-than", "30d"])
        assert result.exit_code == 0
        assert "No stale cassettes" in result.output


# ── Validate ────────────────────────────────────────────────────────


class TestCLIValidate:
    def test_validate_good(self, tmp_path: Path):
        path = _save_sample_cassette(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_missing_file(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_validate_invalid_yaml(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(":::not valid yaml:::", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(bad)])
        # Should detect issue
        assert result.exit_code != 0 or "Invalid" in result.output or "Issue" in result.output
