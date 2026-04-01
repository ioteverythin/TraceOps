"""pytest plugin for traceops.

Auto-registered via entry_points in pyproject.toml.
Provides fixtures and markers for agent trace recording/replay.

Usage:
    @pytest.mark.traceops_cassette("cassettes/test_math.yaml")
    def test_my_agent():
        agent.run("What is 2+2?")

    # Or with fixtures:
    def test_my_agent(traceops_recorder):
        with traceops_recorder("cassettes/test_math.yaml") as rec:
            agent.run("What is 2+2?")
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from trace_ops.cassette import cassette_path_for_test, load_cassette, save_cassette
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add traceops CLI options to pytest."""
    group = parser.getgroup("traceops", "Agent trace recording and replay")
    group.addoption(
        "--record",
        action="store_true",
        default=False,
        help="Record new cassettes instead of replaying existing ones.",
    )
    group.addoption(
        "--record-mode",
        choices=["none", "once", "new", "all", "auto"],
        default="none",
        help=(
            "Recording mode: 'none' (replay only), 'once' (record if cassette missing), "
            "'new' (record only new tests), 'all' (always re-record), "
            "'auto' (record if missing, replay if exists)."
        ),
    )
    group.addoption(
        "--replay-strict",
        action="store_true",
        default=True,
        help="Fail on any divergence from the recorded cassette.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers."""
    config.addinivalue_line(
        "markers",
        "traceops_cassette(path): Mark test to use a specific cassette file for recording/replay.",
    )
    config.addinivalue_line(
        "markers",
        "budget(max_usd=None, max_tokens=None, max_llm_calls=None, max_consecutive_same_tool=None): "
        "Enforce cost, token, and behavioural budgets on the trace recorded during the test.",
    )
    config.addinivalue_line(
        "markers",
        "rag_budget("
        "max_chunks=None, "
        "max_retrieval_ms=None, "
        "max_context_percent=None, "
        "min_faithfulness=None, "
        "min_context_relevance=None"
        "): Enforce RAG-specific budget guards on the trace recorded during the test.",
    )
    config.addinivalue_line(
        "markers",
        "retriever_snapshot(path, threshold=0.8): "
        "Check that retriever output matches the saved snapshot file.",
    )


@pytest.fixture
def traceops_recorder() -> type[Recorder]:
    """Provide the Recorder class for manual recording in tests.

    Usage:
        def test_something(traceops_recorder):
            with traceops_recorder(save_to="cassettes/test.yaml") as rec:
                agent.run("Hello")
            assert rec.trace.total_llm_calls > 0
    """
    return Recorder


@pytest.fixture
def trace_opser() -> type[Replayer]:
    """Provide the Replayer class for manual replay in tests.

    Usage:
        def test_something(trace_opser):
            with trace_opser("cassettes/test.yaml") as rep:
                result = agent.run("Hello")
    """
    return Replayer


@pytest.fixture
def cassette(request: pytest.FixtureRequest) -> Generator[Recorder | Replayer, None, None]:
    """Auto-detect record/replay mode based on CLI flags and cassette existence.

    Uses the test name to derive the cassette path automatically.

    Usage:
        def test_something(cassette):
            # Automatically records on first run, replays on subsequent runs
            agent.run("Hello")
    """
    record_flag = request.config.getoption("--record", default=False)
    record_mode = request.config.getoption("--record-mode", default="none")
    strict = request.config.getoption("--replay-strict", default=True)

    # Environment variable override: TRACEOPS_RECORD=1 or TRACEOPS_MODE=auto
    import os

    env_record = os.environ.get("TRACEOPS_RECORD", "").lower()
    if env_record in ("1", "true", "yes"):
        record_flag = True

    env_mode = os.environ.get("TRACEOPS_MODE", "").lower()
    if env_mode in ("none", "once", "new", "all", "auto"):
        record_mode = env_mode

    # Check for marker with explicit path
    marker = request.node.get_closest_marker("traceops_cassette")
    if marker and marker.args:
        cass_path = Path(marker.args[0])
    else:
        cass_path = cassette_path_for_test(
            str(request.fspath), request.node.name
        )

    # Determine mode
    should_record = False
    if record_flag or record_mode == "all" or record_mode in ("once", "new", "auto") and not cass_path.exists():
        should_record = True

    if should_record:
        rec = Recorder(
            save_to=str(cass_path),
            description=f"Auto-recorded for {request.node.nodeid}",
        )
        with rec:
            yield rec
    else:
        if not cass_path.exists():
            pytest.skip(
                f"Cassette not found: {cass_path}. "
                f"Run with --record to create it."
            )
        rep = Replayer(str(cass_path), strict=strict)
        with rep:
            yield rep


@pytest.fixture
def trace_snapshot(request: pytest.FixtureRequest):
    """Provide trace comparison utilities for snapshot-style testing.

    Usage:
        def test_something(trace_snapshot):
            with Recorder() as rec:
                agent.run("Hello")
            trace_snapshot.assert_unchanged(rec.trace)
    """
    from trace_ops.diff import assert_trace_unchanged

    update = request.config.getoption("--record", default=False)
    test_file = Path(str(request.fspath))
    test_name = request.node.name
    snap_path = cassette_path_for_test(str(test_file), f"{test_name}_snapshot")

    class _TraceSnapshot:
        def assert_unchanged(self, trace, **kwargs):
            if update or not snap_path.exists():
                save_cassette(trace, str(snap_path))
                return

            old_trace = load_cassette(str(snap_path))
            assert_trace_unchanged(old_trace, trace, **kwargs)

    return _TraceSnapshot()


# ── Budget marker support ───────────────────────────────────────────


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[type-arg]
    """After a test runs, enforce @pytest.mark.budget constraints on recorded traces."""
    if call.when != "call" or call.excinfo is not None:
        return

    marker = item.get_closest_marker("budget")
    if marker is None:
        return

    # Find the trace from the cassette fixture (if used)
    from trace_ops.assertions import (
        assert_cost_under,
        assert_max_llm_calls,
        assert_no_loops,
        assert_tokens_under,
    )

    # Look for a Recorder in the fixture values
    trace = None
    for fixture_name in ("cassette",):
        if fixture_name in item.funcargs:
            ctx = item.funcargs[fixture_name]
            if hasattr(ctx, "trace"):
                trace = ctx.trace
            elif hasattr(ctx, "_trace"):
                trace = ctx._trace
            break

    if trace is None:
        return  # No trace to check

    max_usd = marker.kwargs.get("max_usd")
    max_tokens = marker.kwargs.get("max_tokens")
    max_llm_calls = marker.kwargs.get("max_llm_calls")
    max_consecutive_same_tool = marker.kwargs.get("max_consecutive_same_tool")

    if max_usd is not None:
        assert_cost_under(trace, max_usd=max_usd)
    if max_tokens is not None:
        assert_tokens_under(trace, max_tokens=max_tokens)
    if max_llm_calls is not None:
        assert_max_llm_calls(trace, max_calls=max_llm_calls)
    if max_consecutive_same_tool is not None:
        assert_no_loops(trace, max_consecutive_same_tool=max_consecutive_same_tool)

    # ── rag_budget marker ───────────────────────────────────────────
    rag_marker = item.get_closest_marker("rag_budget")
    if rag_marker is not None and trace is not None:
        try:
            from trace_ops.rag.assertions import (
                assert_chunk_count,
                assert_context_window_usage,
                assert_rag_scores,
                assert_retrieval_latency,
            )
        except ImportError:
            pass
        else:
            max_chunks = rag_marker.kwargs.get("max_chunks")
            max_retrieval_ms = rag_marker.kwargs.get("max_retrieval_ms")
            max_context_percent = rag_marker.kwargs.get("max_context_percent")
            min_faithfulness = rag_marker.kwargs.get("min_faithfulness")
            min_context_relevance = rag_marker.kwargs.get("min_context_relevance")

            if max_chunks is not None:
                assert_chunk_count(trace, max_chunks=max_chunks)
            if max_retrieval_ms is not None:
                assert_retrieval_latency(trace, max_latency_ms=max_retrieval_ms)
            if max_context_percent is not None:
                assert_context_window_usage(trace, max_percent=max_context_percent)

            rag_score_constraints: dict[str, float] = {}
            if min_faithfulness is not None:
                rag_score_constraints["faithfulness"] = float(min_faithfulness)
            if min_context_relevance is not None:
                rag_score_constraints["context_relevance"] = float(min_context_relevance)
            if rag_score_constraints:
                assert_rag_scores(trace, min_scores=rag_score_constraints)

    # ── retriever_snapshot marker ────────────────────────────────────
    snap_marker = item.get_closest_marker("retriever_snapshot")
    if snap_marker is not None:
        try:
            from trace_ops.rag.snapshot import RetrieverSnapshot
        except ImportError:
            return

        snap_path = snap_marker.args[0] if snap_marker.args else snap_marker.kwargs.get("path")
        threshold = snap_marker.kwargs.get("threshold", 0.8)

        if snap_path is None:
            return

        # Find cassette/recorder from fixtures
        ctx = item.funcargs.get("cassette")
        if ctx is None:
            return

        # If we have a trace with retrieval events, check against snapshot
        check_trace = getattr(ctx, "_trace", None) or getattr(ctx, "trace", None)
        if check_trace is None:
            return

        try:
            snap = RetrieverSnapshot.load(snap_path)
            results = snap.check_trace(check_trace, threshold=threshold)
            failed = [r for r in results if not r.passed]
            if failed:
                msgs = "; ".join(f"query '{r.query[:40]}' score={r.score:.2f}" for r in failed)
                raise AssertionError(
                    f"Retriever snapshot check failed for {len(failed)} query(s): {msgs}"
                )
        except FileNotFoundError:
            pass  # No snapshot yet — first run silently passes
