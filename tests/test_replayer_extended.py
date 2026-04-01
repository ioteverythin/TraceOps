"""Extended replayer tests — LiteLLM patching, __exit__ validation,
langchain/langgraph/crewai delegation, allow_new_calls, model mismatch.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import CassetteMismatchError
from trace_ops.replayer import Replayer

# ── Helpers ─────────────────────────────────────────────────────────


def _build_cassette(tmp_path, responses, provider="openai"):
    """Build a cassette YAML file with the given LLM response dicts."""
    trace = Trace()
    for resp in responses:
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider=provider,
            model=resp.get("model", "gpt-4o"),
        ))
        trace.add_event(TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider=provider,
            model=resp.get("model", "gpt-4o"),
            response=resp,
            input_tokens=10,
            output_tokens=5,
        ))
    trace.finalize()
    path = tmp_path / "test.yaml"
    data = trace.to_dict()
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


def _openai_resp(content="Hello", model="gpt-4o"):
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


# ── LiteLLM replay patching ────────────────────────────────────────


class _FakeLiteLLM:
    """Fake litellm module for testing."""
    _original_completion = None
    _original_acompletion = None

    def __init__(self):
        self.completion = self._real_completion
        self.acompletion = self._real_acompletion

    def _real_completion(self, *args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="real", role="assistant"),
                finish_reason="stop",
            )],
            model="gpt-4o",
        )

    async def _real_acompletion(self, *args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="real-async", role="assistant"),
                finish_reason="stop",
            )],
            model="gpt-4o",
        )


@pytest.fixture()
def fake_litellm():
    fake = _FakeLiteLLM()
    old = sys.modules.get("litellm")
    sys.modules["litellm"] = fake  # type: ignore[assignment]
    yield fake
    if old is None:
        sys.modules.pop("litellm", None)
    else:
        sys.modules["litellm"] = old


class TestReplayerLiteLLMSync:
    def test_replays_recorded_response(self, tmp_path, fake_litellm):
        cassette = _build_cassette(tmp_path, [_openai_resp("replayed")], provider="litellm")
        replayer = Replayer(
            cassette,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=True,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        with replayer:
            import litellm
            result = litellm.completion(model="gpt-4o", messages=[])
        assert result.choices[0].message.content == "replayed"

    def test_litellm_model_from_positional_arg(self, tmp_path, fake_litellm):
        cassette = _build_cassette(tmp_path, [_openai_resp("ok")], provider="litellm")
        replayer = Replayer(
            cassette,
            strict=False,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=True,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        with replayer:
            import litellm
            result = litellm.completion("gpt-4o", messages=[])
        assert result is not None


class TestReplayerLiteLLMAsync:
    @pytest.mark.asyncio
    async def test_async_replays(self, tmp_path, fake_litellm):
        cassette = _build_cassette(tmp_path, [_openai_resp("async-replayed")], provider="litellm")
        replayer = Replayer(
            cassette,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=True,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        async with replayer:
            import litellm
            result = await litellm.acompletion(model="gpt-4o", messages=[])
        assert result.choices[0].message.content == "async-replayed"


# ── Replayer __exit__ validation ───────────────────────────────────


class TestReplayerExitValidation:
    def test_unconsumed_responses_strict_raises(self, tmp_path):
        cassette = _build_cassette(tmp_path, [
            _openai_resp("first"),
            _openai_resp("second"),
        ])
        with pytest.raises(CassetteMismatchError, match="fewer LLM calls"), Replayer(
            cassette,
            intercept_openai=True,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        ):
            from openai.resources.chat.completions import Completions
            Completions.create(MagicMock(), model="gpt-4o", messages=[])
                # Only consumed 1 of 2 responses

    @pytest.mark.asyncio
    async def test_async_unconsumed_raises(self, tmp_path):
        cassette = _build_cassette(tmp_path, [
            _openai_resp("first"),
            _openai_resp("second"),
        ])
        with pytest.raises(CassetteMismatchError, match="fewer LLM calls"):
            async with Replayer(
                cassette,
                intercept_openai=True,
                intercept_anthropic=False,
                intercept_litellm=False,
                intercept_langchain=False,
                intercept_langgraph=False,
                intercept_crewai=False,
            ):
                from openai.resources.chat.completions import Completions
                Completions.create(MagicMock(), model="gpt-4o", messages=[])


# ── Replayer model mismatch ────────────────────────────────────────


class TestReplayerModelMismatch:
    def test_strict_model_mismatch_raises(self, tmp_path):
        cassette = _build_cassette(tmp_path, [_openai_resp("ok", model="gpt-4o")])
        with pytest.raises(CassetteMismatchError, match="model"), Replayer(
            cassette,
            strict=True,
            intercept_openai=True,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        ):
            from openai.resources.chat.completions import Completions
            Completions.create(MagicMock(), model="gpt-4o-mini", messages=[])


# ── Replayer delegation ────────────────────────────────────────────


class TestReplayerDelegation:
    def test_langchain_delegation(self, tmp_path):
        cassette = _build_cassette(tmp_path, [])  # empty — no responses to consume
        r = Replayer(
            cassette,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=True,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        with r:
            pass  # Should not crash

    def test_langgraph_delegation(self, tmp_path):
        cassette = _build_cassette(tmp_path, [])
        r = Replayer(
            cassette,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=True,
            intercept_crewai=False,
        )
        with r:
            pass

    def test_crewai_delegation(self, tmp_path):
        cassette = _build_cassette(tmp_path, [])
        r = Replayer(
            cassette,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=True,
        )
        with r:
            pass


# ── Replayer properties ────────────────────────────────────────────


class TestReplayerProperties:
    def test_recorded_trace(self, tmp_path):
        cassette = _build_cassette(tmp_path, [_openai_resp("ok")])
        r = Replayer(
            cassette,
            strict=False,  # Don't fail on unconsumed responses
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        assert r.recorded_trace is None  # Not loaded yet
        with r:
            assert r.recorded_trace is not None
            assert len(r.recorded_trace.events) > 0

    def test_replay_trace(self, tmp_path):
        cassette = _build_cassette(tmp_path, [_openai_resp("ok")])
        r = Replayer(
            cassette,
            strict=False,
            intercept_openai=False,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        assert r.replay_trace is None
        with r:
            assert r.replay_trace is not None

    def test_non_strict_extra_calls_return_empty(self, tmp_path):
        cassette = _build_cassette(tmp_path, [])  # No responses
        r = Replayer(
            cassette,
            strict=False,
            intercept_openai=True,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        with r:
            from openai.resources.chat.completions import Completions
            result = Completions.create(MagicMock(), model="gpt-4o", messages=[])
            # Non-strict returns empty namespace from empty dict
            assert result is not None


# ── Replayer.replay decorator ──────────────────────────────────────


class TestReplayerDecorator:
    def test_decorator(self, tmp_path):
        _build_cassette(tmp_path, [_openai_resp("decorated")])

        @Replayer.replay(
            str(tmp_path / "test.yaml"),
            intercept_openai=True,
            intercept_anthropic=False,
            intercept_litellm=False,
            intercept_langchain=False,
            intercept_langgraph=False,
            intercept_crewai=False,
        )
        def my_test():
            from openai.resources.chat.completions import Completions
            result = Completions.create(MagicMock(), model="gpt-4o", messages=[])
            return result.choices[0].message.content

        assert my_test() == "decorated"
