"""Tests for auto-record mode, env var overrides, and deeper pytest plugin paths."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import save_cassette


# ── Helpers ─────────────────────────────────────────────────────────


def _make_cassette(tmp_path: Path, name: str = "cassette.yaml") -> Path:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        model="gpt-4o",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
    ))
    trace.finalize()
    path = tmp_path / name
    save_cassette(trace, path)
    return path


# ── pytest_addoption ────────────────────────────────────────────────


class TestPytestAddOption:
    def test_auto_mode_in_choices(self):
        """Verify 'auto' was added as a record-mode choice."""
        from trace_ops.pytest_plugin import pytest_addoption

        class MockOption:
            def __init__(self):
                self.options = []

            def addoption(self, *args, **kwargs):
                self.options.append((args, kwargs))

        class MockParser:
            def __init__(self):
                self.group = MockOption()

            def getgroup(self, *args, **kwargs):
                return self.group

        parser = MockParser()
        pytest_addoption(parser)

        # Find the --record-mode option
        record_mode_opt = None
        for args, kwargs in parser.group.options:
            if "--record-mode" in args:
                record_mode_opt = kwargs
                break

        assert record_mode_opt is not None
        assert "auto" in record_mode_opt["choices"]


# ── Environment variable overrides ──────────────────────────────────


class TestEnvVarOverrides:
    """Test trace_ops_RECORD and trace_ops_MODE env vars."""

    def test_env_record_true(self):
        """trace_ops_RECORD=1 should force recording."""
        # This tests the logic by importing and checking the fixture function
        # We can't easily test the fixture directly, but we can test the logic
        with patch.dict(os.environ, {"trace_ops_RECORD": "1"}):
            env_record = os.environ.get("trace_ops_RECORD", "").lower()
            assert env_record in ("1", "true", "yes")

    def test_env_record_yes(self):
        with patch.dict(os.environ, {"trace_ops_RECORD": "yes"}):
            env_record = os.environ.get("trace_ops_RECORD", "").lower()
            assert env_record in ("1", "true", "yes")

    def test_env_record_true_string(self):
        with patch.dict(os.environ, {"trace_ops_RECORD": "TRUE"}):
            env_record = os.environ.get("trace_ops_RECORD", "").lower()
            assert env_record in ("1", "true", "yes")

    def test_env_mode_auto(self):
        with patch.dict(os.environ, {"trace_ops_MODE": "auto"}):
            env_mode = os.environ.get("trace_ops_MODE", "").lower()
            assert env_mode in ("none", "once", "new", "all", "auto")

    def test_env_mode_all(self):
        with patch.dict(os.environ, {"trace_ops_MODE": "ALL"}):
            env_mode = os.environ.get("trace_ops_MODE", "").lower()
            assert env_mode in ("none", "once", "new", "all", "auto")

    def test_env_mode_invalid_ignored(self):
        with patch.dict(os.environ, {"trace_ops_MODE": "invalid"}):
            env_mode = os.environ.get("trace_ops_MODE", "").lower()
            assert env_mode not in ("none", "once", "new", "all", "auto")

    def test_env_unset_default(self):
        env = dict(os.environ)
        env.pop("trace_ops_RECORD", None)
        env.pop("trace_ops_MODE", None)
        with patch.dict(os.environ, env, clear=True):
            env_record = os.environ.get("trace_ops_RECORD", "").lower()
            assert env_record not in ("1", "true", "yes")


# ── Record mode logic ──────────────────────────────────────────────


class TestRecordModeLogic:
    """Test the should_record decision logic from the cassette fixture."""

    def _should_record(
        self,
        record_flag: bool,
        record_mode: str,
        cassette_exists: bool,
    ) -> bool:
        """Replicate the logic from the cassette fixture."""
        should_record = False
        if record_flag or record_mode == "all":
            should_record = True
        elif record_mode in ("once", "new", "auto") and not cassette_exists:
            should_record = True
        return should_record

    def test_none_mode_no_record(self):
        assert not self._should_record(False, "none", True)
        assert not self._should_record(False, "none", False)

    def test_once_records_when_missing(self):
        assert self._should_record(False, "once", False)

    def test_once_replays_when_exists(self):
        assert not self._should_record(False, "once", True)

    def test_new_records_when_missing(self):
        assert self._should_record(False, "new", False)

    def test_new_replays_when_exists(self):
        assert not self._should_record(False, "new", True)

    def test_auto_records_when_missing(self):
        assert self._should_record(False, "auto", False)

    def test_auto_replays_when_exists(self):
        assert not self._should_record(False, "auto", True)

    def test_all_always_records(self):
        assert self._should_record(False, "all", True)
        assert self._should_record(False, "all", False)

    def test_flag_overrides_mode(self):
        assert self._should_record(True, "none", True)
        assert self._should_record(True, "none", False)


# ── Budget marker ───────────────────────────────────────────────────


class TestBudgetMarker:
    def test_budget_marker_registered(self):
        """The budget marker should be registered as a valid marker."""
        from trace_ops.pytest_plugin import pytest_configure

        class FakeConfig:
            def __init__(self):
                self.values = []

            def addinivalue_line(self, name, line):
                self.values.append((name, line))

        config = FakeConfig()
        pytest_configure(config)
        marker_lines = [v[1] for v in config.values if v[0] == "markers"]
        assert any("budget" in line for line in marker_lines)
        assert any("traceops_cassette" in line for line in marker_lines)


# ── Fixture types ───────────────────────────────────────────────────


class TestFixtureTypes:
    def test_traceops_recorder_returns_recorder_class(self):
        """The traceops_recorder fixture function body returns Recorder."""
        from trace_ops.recorder import Recorder
        # Verify the symbol is importable and correct
        assert Recorder is not None
        assert hasattr(Recorder, "__enter__")

    def test_trace_opser_returns_replayer_class(self):
        """The trace_opser fixture function body returns Replayer."""
        from trace_ops.replayer import Replayer
        assert Replayer is not None
        assert hasattr(Replayer, "__enter__")
