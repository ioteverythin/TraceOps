"""Tests for the Replayer — v0.2 features.

These tests focus on the Replayer's internal logic (loading, response
queueing, mismatch detection, thread safety) without requiring actual
LLM SDK packages installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import CassetteMismatchError, save_cassette
from trace_ops.replayer import Replayer

# ── Helpers ─────────────────────────────────────────────────────────


def _save_cassette(tmp_path: Path, responses: int = 2) -> Path:
    """Save a simple cassette with N LLM response events."""
    trace = Trace()
    for i in range(responses):
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": f"msg-{i}"}],
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            response={"choices": [{"message": {"content": f"reply-{i}"}}]},
            input_tokens=10,
            output_tokens=5,
        ))
    trace.finalize()
    path = tmp_path / "test.yaml"
    save_cassette(trace, path)
    return path


# ── Tests ───────────────────────────────────────────────────────────


class TestReplayerInit:
    def test_basic_init(self, tmp_path: Path):
        path = _save_cassette(tmp_path)
        replayer = Replayer(str(path))
        assert replayer.cassette_path == str(path)
        assert replayer.strict is True
        assert replayer.allow_new_calls is False

    def test_non_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path)
        replayer = Replayer(str(path), strict=False)
        assert replayer.strict is False


class TestReplayerLoad:
    def test_load_prepares_queue(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=3)
        replayer = Replayer(str(path))
        replayer._load()
        assert len(replayer._response_queue) == 3
        assert replayer._call_index == 0

    def test_get_next_response(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=2)
        replayer = Replayer(str(path))
        replayer._load()

        resp0 = replayer._get_next_response("openai", "gpt-4o")
        assert resp0["choices"][0]["message"]["content"] == "reply-0"

        resp1 = replayer._get_next_response("openai", "gpt-4o")
        assert resp1["choices"][0]["message"]["content"] == "reply-1"

    def test_exhausted_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=1)
        replayer = Replayer(str(path))
        replayer._load()

        replayer._get_next_response("openai", "gpt-4o")  # consume the only one
        with pytest.raises(CassetteMismatchError, match="more LLM calls"):
            replayer._get_next_response("openai", "gpt-4o")

    def test_exhausted_non_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=1)
        replayer = Replayer(str(path), strict=False)
        replayer._load()

        replayer._get_next_response("openai", "gpt-4o")
        # Non-strict should return empty dict
        resp = replayer._get_next_response("openai", "gpt-4o")
        assert resp == {}


class TestReplayerMismatch:
    def test_provider_mismatch_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=1)
        replayer = Replayer(str(path))
        replayer._load()

        with pytest.raises(CassetteMismatchError, match="provider"):
            replayer._get_next_response("anthropic", "gpt-4o")

    def test_model_mismatch_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=1)
        replayer = Replayer(str(path))
        replayer._load()

        with pytest.raises(CassetteMismatchError, match="model"):
            replayer._get_next_response("openai", "gpt-3.5-turbo")


class TestReplayerContextManager:
    def test_enter_exit_no_sdks(self, tmp_path: Path):
        """Entering/exiting should work even without LLM SDKs installed."""
        path = _save_cassette(tmp_path, responses=0)
        # The Replayer context manager should not raise even if no SDK patches apply
        with Replayer(str(path), strict=False) as replayer:
            assert replayer.recorded_trace is not None

    def test_unconsumed_responses_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=2)
        with pytest.raises(CassetteMismatchError, match="fewer LLM calls"), Replayer(str(path)) as replayer:
            # consume only 1 of 2
            replayer._get_next_response("openai", "gpt-4o")

    def test_unconsumed_responses_non_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=2)
        # Non-strict should not raise
        with Replayer(str(path), strict=False) as replayer:
            replayer._get_next_response("openai", "gpt-4o")


class TestReplayerAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_enter_exit(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=0)
        async with Replayer(str(path), strict=False) as replayer:
            assert replayer.recorded_trace is not None

    @pytest.mark.asyncio
    async def test_async_unconsumed_strict(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=2)
        with pytest.raises(CassetteMismatchError, match="fewer LLM calls"):
            async with Replayer(str(path)) as replayer:
                replayer._get_next_response("openai", "gpt-4o")


class TestReplayerDecorator:
    def test_decorator_basic(self, tmp_path: Path):
        path = _save_cassette(tmp_path, responses=0)

        @Replayer.replay(str(path), strict=False)
        def my_func():
            return "ran"

        result = my_func()
        assert result == "ran"


class TestReplayerThreadSafety:
    def test_concurrent_access(self, tmp_path: Path):
        """Ensure _get_next_response is thread-safe."""
        import threading

        path = _save_cassette(tmp_path, responses=100)
        replayer = Replayer(str(path), strict=False)
        replayer._load()

        results = []
        errors = []

        def consume():
            try:
                for _ in range(10):
                    resp = replayer._get_next_response("openai", "gpt-4o")
                    results.append(resp)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=consume) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 100
