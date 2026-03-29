"""pytest-based testing with agent-replay.

Using the @pytest.mark.traceops_cassette marker and cassette fixture
for automatic record/replay with pytest.

Run:
    # Record all cassettes
    pytest tests/ --record

    # Replay (default)
    pytest tests/

    # Specific record mode
    pytest tests/ --record-mode=once
    pytest tests/ --record-mode=new
    pytest tests/ --record-mode=all
"""

from __future__ import annotations

import pytest

from trace_ops import Recorder, Replayer


@pytest.mark.traceops_cassette("cassettes/test_math.yaml")
def test_agent_math(cassette):
    """Test that uses auto-record/replay via the cassette fixture.
    
    First run (with --record):
        cassette is a Recorder, saves to cassettes/test_math.yaml
    
    Subsequent runs (no --record):
        cassette is a Replayer, reads from cassettes/test_math.yaml
    """
    assert isinstance(cassette, (Recorder, Replayer))
    # Your agent code here
    # cassette will auto-record or auto-replay based on CLI flags


def test_budget(cassette):
    """Test with a cost budget constraint.
    
    Run with:
        pytest tests/ --record -v
    """
    assert isinstance(cassette, (Recorder, Replayer))
    # Agent code that stays under $0.10
    # cassette.assert_cost_under(0.10)


def test_with_fixture(trace_snapshot):
    """Use trace_snapshot for snapshot-style testing.
    
    First run: records the trace to cassettes/test_with_fixture_snapshot.yaml
    Subsequent runs: compares against the snapshot
    """
    from trace_ops import Trace, TraceEvent, EventType

    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.LLM_REQUEST,
        provider="test",
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
    ))
    trace.finalize()

    # This will save (--record) or compare (normal run)
    trace_snapshot.assert_unchanged(trace)


if __name__ == "__main__":
    print(__doc__)
