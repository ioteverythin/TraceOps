"""Tests for the agent-replay pytest plugin.

Uses ``pytester`` to spin up isolated pytest runs and verify that
the ``cassette`` fixture, ``--record`` / ``--record-mode`` CLI flags,
``@pytest.mark.traceops_cassette``, ``@pytest.mark.budget``, and the
``trace_snapshot`` fixture all work correctly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import save_cassette
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer

# ── enable pytester ──────────────────────────────────────────────────

pytest_plugins = ["pytester"]


# ── helpers ──────────────────────────────────────────────────────────


def _make_cassette(path: Path, n_llm_calls: int = 1) -> None:
    """Write a minimal cassette file to *path*."""
    trace = Trace()
    for i in range(n_llm_calls):
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"q{i}"}],
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o-mini",
            response={"choices": [{"message": {"content": f"a{i}"}}]},
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        ))
    trace.finalize()
    save_cassette(trace, path)


# ── Test: traceops_recorder / trace_opser fixtures ───────────────────


class TestSimpleFixtures:
    """``traceops_recorder`` and ``trace_opser`` provide the class objects."""

    def test_traceops_recorder_fixture(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            def test_fixture(traceops_recorder):
                assert traceops_recorder is not None
                rec = traceops_recorder()
                with rec:
                    rec.record_decision("test")
                assert len(rec.trace.events) == 1
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_trace_opser_fixture(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            from trace_ops.replayer import Replayer
            def test_fixture(trace_opser):
                assert trace_opser is Replayer
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


# ── Test: cassette fixture – record mode ─────────────────────────────


class TestCassetteFixtureRecord:
    """``cassette`` fixture in various record modes."""

    def test_record_flag_creates_cassette(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            from trace_ops.recorder import Recorder
            def test_rec(cassette):
                # cassette should be a Recorder in --record mode
                assert isinstance(cassette, Recorder)
                cassette.record_decision("hello")
        """)
        result = pytester.runpytest("--record", "-v")
        result.assert_outcomes(passed=1)

    def test_record_mode_all(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            from trace_ops.recorder import Recorder
            def test_rec(cassette):
                assert isinstance(cassette, Recorder)
        """)
        result = pytester.runpytest("--record-mode=all", "-v")
        result.assert_outcomes(passed=1)

    def test_record_mode_once_creates_when_missing(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            from trace_ops.recorder import Recorder
            def test_rec(cassette):
                assert isinstance(cassette, Recorder)
        """)
        result = pytester.runpytest("--record-mode=once", "-v")
        result.assert_outcomes(passed=1)


# ── Test: cassette fixture – replay mode ─────────────────────────────


class TestCassetteFixtureReplay:
    """``cassette`` fixture in default (replay) mode."""

    def test_skips_when_cassette_missing(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            def test_rep(cassette):
                pass
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(skipped=1)

    def test_replays_existing_cassette(self, pytester: pytest.Pytester) -> None:
        # Pre-create the cassette in the expected location
        pytester.makepyfile("""
            import pytest
            from pathlib import Path
            from trace_ops.replayer import Replayer

            @pytest.mark.traceops_cassette("my_cassette.yaml")
            def test_rep(cassette):
                assert isinstance(cassette, Replayer)
        """)
        # Create the cassette file
        cass = pytester.path / "my_cassette.yaml"
        _make_cassette(cass, n_llm_calls=0)

        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


# ── Test: @pytest.mark.traceops_cassette ────────────────────────────────


class TestAgentCassetteMarker:
    """``@pytest.mark.traceops_cassette(path)`` controls cassette path."""

    def test_marker_with_explicit_path(self, pytester: pytest.Pytester) -> None:
        cass = pytester.path / "custom" / "my.yaml"
        _make_cassette(cass, n_llm_calls=0)

        pytester.makepyfile("""
            import pytest
            from trace_ops.replayer import Replayer

            @pytest.mark.traceops_cassette("custom/my.yaml")
            def test_custom(cassette):
                assert isinstance(cassette, Replayer)
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_marker_record_with_path(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest
            from trace_ops.recorder import Recorder

            @pytest.mark.traceops_cassette("explicit_rec.yaml")
            def test_custom(cassette):
                assert isinstance(cassette, Recorder)
        """)
        result = pytester.runpytest("--record", "-v")
        result.assert_outcomes(passed=1)
        assert (pytester.path / "explicit_rec.yaml").exists()


# ── Test: trace_snapshot fixture ─────────────────────────────────────


class TestTraceSnapshotFixture:
    """``trace_snapshot`` fixture for snapshot testing."""

    def test_first_run_records_snapshot(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            from trace_ops._types import Trace, TraceEvent, EventType

            def test_snap(trace_snapshot):
                trace = Trace()
                trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="openai", model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                ))
                trace.finalize()
                # First run should just save the snapshot (no comparison)
                trace_snapshot.assert_unchanged(trace)
        """)
        result = pytester.runpytest("--record", "-v")
        result.assert_outcomes(passed=1)

    def test_second_run_compares(self, pytester: pytest.Pytester) -> None:
        test_code = textwrap.dedent("""\
            from trace_ops._types import Trace, TraceEvent, EventType

            def test_snap(trace_snapshot):
                trace = Trace()
                trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="openai", model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                ))
                trace.finalize()
                trace_snapshot.assert_unchanged(trace)
        """)
        pytester.makepyfile(**{"test_snap": test_code})
        # First run: record
        r1 = pytester.runpytest("--record", "-v")
        r1.assert_outcomes(passed=1)

        # Second run: replay (compares against the snapshot)
        r2 = pytester.runpytest("-v")
        r2.assert_outcomes(passed=1)


# ── Test: @pytest.mark.budget ────────────────────────────────────────


class TestBudgetMarker:
    """Verify budget constraints are enforced via the marker."""

    def test_budget_marker_registered(self, pytester: pytest.Pytester) -> None:
        """Budget marker should be registered without warnings."""
        pytester.makepyfile("""
            import pytest

            @pytest.mark.budget(max_llm_calls=10)
            def test_budget(cassette):
                pass
        """)
        result = pytester.runpytest("--record", "-v", "-W", "error::pytest.PytestUnknownMarkWarning")
        # Should not fail with "Unknown marker" warning
        result.assert_outcomes(passed=1)


# ── Test: CLI options registered ─────────────────────────────────────


class TestCLIOptions:
    """Verify that --record, --record-mode, --replay-strict are registered."""

    def test_help_includes_options(self, pytester: pytest.Pytester) -> None:
        result = pytester.runpytest("--help")
        output = "\n".join(result.outlines)
        assert "--record" in output
        assert "--record-mode" in output
        assert "--replay-strict" in output


# ── Direct unit tests (no pytester needed) ───────────────────────────


class TestPluginDirect:
    """Test plugin components directly without full pytest subprocess."""

    def test_traceops_recorder_fixture_param(self, traceops_recorder) -> None:
        """Fixture should provide the Recorder class."""
        assert traceops_recorder is Recorder

    def test_trace_opser_fixture_param(self, trace_opser) -> None:
        """Fixture should provide the Replayer class."""
        assert trace_opser is Replayer
