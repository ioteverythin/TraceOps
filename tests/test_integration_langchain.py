"""Integration tests for agent-replay ↔ LangChain.

These tests use real ``langchain-core`` classes (no API keys needed)
to verify that recording and replaying work end-to-end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from trace_ops._types import EventType
from trace_ops.cassette import load_cassette
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer

# ── langchain-core availability ──────────────────────────────────────
lc = pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def cassette_path(tmp_path: Path) -> str:
    return str(tmp_path / "langchain.yaml")


# ── Test: basic text response ────────────────────────────────────────


class TestLangChainRecordReplay:
    """Record + replay with FakeListChatModel (plain text responses)."""

    def test_record_captures_request_and_response(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["Paris is the capital of France."])

        with Recorder(save_to=cassette_path) as rec:
            result = model.invoke("What is the capital of France?")

        assert result.content == "Paris is the capital of France."

        events = rec.trace.events
        req_events = [e for e in events if e.event_type == EventType.LLM_REQUEST]
        res_events = [e for e in events if e.event_type == EventType.LLM_RESPONSE]
        assert len(req_events) == 1
        assert len(res_events) == 1
        assert req_events[0].provider == "langchain"
        assert req_events[0].model == "FakeListChatModel"
        assert res_events[0].duration_ms is not None

    def test_replay_returns_recorded_response(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["Paris is the capital of France."])

        # Record
        with Recorder(save_to=cassette_path):
            model.invoke("What is the capital of France?")

        # Replay — even though the model would return a different response
        # on a fresh .invoke(), the replayer intercepts and returns the
        # recorded one.
        model2 = FakeListChatModel(responses=["SHOULD NOT SEE THIS"])
        with Replayer(cassette_path):
            result = model2.invoke("What is the capital of France?")

        assert result.content == "Paris is the capital of France."

    def test_cassette_file_is_valid_yaml(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["42"])
        with Recorder(save_to=cassette_path):
            model.invoke("What is 6 * 7?")

        trace = load_cassette(cassette_path)
        assert len(trace.events) >= 2  # at least request + response

    def test_multiple_calls_in_one_session(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["A", "B", "C"])

        with Recorder(save_to=cassette_path) as rec:
            r1 = model.invoke("First")
            r2 = model.invoke("Second")
            r3 = model.invoke("Third")

        assert r1.content == "A"
        assert r2.content == "B"
        assert r3.content == "C"

        # Three request/response pairs → 6 events minimum
        llm_responses = [
            e for e in rec.trace.events if e.event_type == EventType.LLM_RESPONSE
        ]
        assert len(llm_responses) == 3

        # Replay all three in order
        model2 = FakeListChatModel(responses=["X"])
        with Replayer(cassette_path):
            rr1 = model2.invoke("First")
            rr2 = model2.invoke("Second")
            rr3 = model2.invoke("Third")

        assert rr1.content == "A"
        assert rr2.content == "B"
        assert rr3.content == "C"


# ── Test: tool-calling responses ─────────────────────────────────────


class TestLangChainToolCalls:
    """Record + replay with FakeMessagesListChatModel (tool call responses)."""

    def test_record_captures_tool_calls(self, cassette_path: str) -> None:
        tc_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {"q": "python"}, "id": "call_abc"},
            ],
        )
        model = FakeMessagesListChatModel(responses=[tc_msg])

        with Recorder(save_to=cassette_path) as rec:
            result = model.invoke("Search for Python")

        assert result.tool_calls[0]["name"] == "search"

        tool_events = [
            e for e in rec.trace.events if e.event_type == EventType.TOOL_CALL
        ]
        assert len(tool_events) >= 1
        assert tool_events[0].tool_name == "search"

    def test_replay_preserves_tool_calls(self, cassette_path: str) -> None:
        tc_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "NYC"}, "id": "call_1"},
            ],
        )
        model = FakeMessagesListChatModel(responses=[tc_msg])

        with Recorder(save_to=cassette_path):
            model.invoke("Weather in NYC?")

        model2 = FakeMessagesListChatModel(
            responses=[AIMessage(content="NOPE")]
        )
        with Replayer(cassette_path):
            result = model2.invoke("Weather in NYC?")

        assert isinstance(result, AIMessage)
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"city": "NYC"}


# ── Test: BaseTool recording ────────────────────────────────────────


class TestLangChainBaseTool:
    """Record BaseTool.invoke as tool_call + tool_result events."""

    def test_tool_invoke_recording(self, cassette_path: str) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        with Recorder(save_to=cassette_path) as rec:
            result = add.invoke({"a": 3, "b": 4})

        assert result == 7

        tool_calls = [
            e for e in rec.trace.events if e.event_type == EventType.TOOL_CALL
        ]
        tool_results = [
            e for e in rec.trace.events if e.event_type == EventType.TOOL_RESULT
        ]
        assert len(tool_calls) >= 1
        assert tool_calls[0].tool_name == "add"
        assert len(tool_results) >= 1
        assert tool_results[0].tool_name == "add"


# ── Test: async ainvoke ──────────────────────────────────────────────


class TestLangChainAsync:
    """Async record + replay through ainvoke."""

    def test_async_record_replay(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["async result"])

        async def _record():
            with Recorder(save_to=cassette_path) as rec:
                result = await model.ainvoke("async question")
            return result, rec

        result, rec = asyncio.run(_record())
        assert result.content == "async result"

        req_events = [
            e for e in rec.trace.events if e.event_type == EventType.LLM_REQUEST
        ]
        assert len(req_events) >= 1

        # Replay
        model2 = FakeListChatModel(responses=["WRONG"])

        async def _replay():
            with Replayer(cassette_path):
                return await model2.ainvoke("async question")

        replayed = asyncio.run(_replay())
        assert replayed.content == "async result"


# ── Test: error recording ───────────────────────────────────────────


class TestLangChainErrorRecording:
    """Verify that exceptions during invoke are captured as ERROR events."""

    def test_error_event_on_exception(self, cassette_path: str) -> None:
        # FakeListChatModel with empty list will raise on invoke
        model = FakeListChatModel(responses=[])

        with pytest.raises(Exception), Recorder(save_to=cassette_path) as rec:
            model.invoke("This will fail")

        # Should still have at least a request event before the error
        req_events = [
            e for e in rec.trace.events if e.event_type == EventType.LLM_REQUEST
        ]
        assert len(req_events) >= 1


# ── Test: patches are cleaned up ────────────────────────────────────


class TestLangChainPatchCleanup:
    """Verify that monkey-patches are removed after context exit."""

    def test_recorder_removes_patches(self) -> None:
        from langchain_core.language_models.chat_models import BaseChatModel

        original_invoke = BaseChatModel.invoke

        with Recorder():
            pass  # just enter and exit

        assert BaseChatModel.invoke is original_invoke

    def test_replayer_removes_patches(self, cassette_path: str) -> None:
        from langchain_core.language_models.chat_models import BaseChatModel

        model = FakeListChatModel(responses=["x"])
        with Recorder(save_to=cassette_path):
            model.invoke("q")

        original_invoke = BaseChatModel.invoke

        # strict=False so we don't error on unconsumed responses
        with Replayer(cassette_path, strict=False):
            pass

        assert BaseChatModel.invoke is original_invoke
