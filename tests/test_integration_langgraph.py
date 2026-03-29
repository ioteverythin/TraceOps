"""Integration tests for agent-replay ↔ LangGraph.

Uses real ``langgraph`` + ``langchain-core`` classes with
``FakeListChatModel`` / ``FakeMessagesListChatModel`` so no API keys
are needed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Annotated

import pytest

from trace_ops._types import EventType
from trace_ops.cassette import load_cassette
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer

# ── availability guards ──────────────────────────────────────────────
pytest.importorskip("langchain_core", reason="langchain-core not installed")
pytest.importorskip("langgraph", reason="langgraph not installed")

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── Shared state definition ─────────────────────────────────────────


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def cassette_path(tmp_path: Path) -> str:
    return str(tmp_path / "langgraph.yaml")


# ── Test: simple chatbot graph ───────────────────────────────────────


class TestSimpleChatbot:
    """Single-node graph: START → chatbot → END."""

    def _build_graph(self, model):
        def chatbot(state: State):
            return {"messages": [model.invoke(state["messages"])]}

        g = StateGraph(State)
        g.add_node("chatbot", chatbot)
        g.add_edge(START, "chatbot")
        g.add_edge("chatbot", END)
        return g.compile()

    def test_record_and_replay(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["Hello from the chatbot!"])
        app = self._build_graph(model)

        # Record
        with Recorder(save_to=cassette_path) as rec:
            result = app.invoke({"messages": [HumanMessage(content="Hi")]})

        assert result["messages"][-1].content == "Hello from the chatbot!"
        assert any(
            e.event_type == EventType.LLM_RESPONSE for e in rec.trace.events
        )

        # Graph-level events from LangGraph interceptor
        decisions = [
            e for e in rec.trace.events
            if e.event_type == EventType.AGENT_DECISION
        ]
        decision_names = [e.decision for e in decisions]
        assert "graph_start" in decision_names
        assert "graph_end" in decision_names

        # Replay — a dummy model is used; the interceptor overrides it
        model2 = FakeListChatModel(responses=["WRONG ANSWER"])
        app2 = self._build_graph(model2)

        with Replayer(cassette_path):
            result2 = app2.invoke({"messages": [HumanMessage(content="Hi")]})

        assert result2["messages"][-1].content == "Hello from the chatbot!"

    def test_multi_turn(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["Answer 1", "Answer 2"])

        def chatbot(state: State):
            return {"messages": [model.invoke(state["messages"])]}

        g = StateGraph(State)
        g.add_node("chatbot", chatbot)
        g.add_edge(START, "chatbot")
        g.add_edge("chatbot", END)
        app = g.compile()

        with Recorder(save_to=cassette_path) as rec:
            r1 = app.invoke({"messages": [HumanMessage(content="Q1")]})
            r2 = app.invoke({"messages": [HumanMessage(content="Q2")]})

        assert r1["messages"][-1].content == "Answer 1"
        assert r2["messages"][-1].content == "Answer 2"

        llm_responses = [
            e for e in rec.trace.events if e.event_type == EventType.LLM_RESPONSE
        ]
        assert len(llm_responses) == 2

        # Replay both turns
        model2 = FakeListChatModel(responses=["X"])
        app2 = self._build_graph(model2)

        with Replayer(cassette_path):
            rr1 = app2.invoke({"messages": [HumanMessage(content="Q1")]})
            rr2 = app2.invoke({"messages": [HumanMessage(content="Q2")]})

        assert rr1["messages"][-1].content == "Answer 1"
        assert rr2["messages"][-1].content == "Answer 2"


# ── Test: ReAct agent with tool calls ────────────────────────────────


class TestReActAgent:
    """Multi-step: agent → tool call → tool execution → agent → final answer."""

    @staticmethod
    @tool
    def search(query: str) -> str:
        """Search the web for information."""
        return f"Result for {query}: Python was created by Guido."

    def _build_react_graph(self, model):
        search_tool = self.search

        def should_continue(state: State):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        def call_model(state: State):
            return {"messages": [model.invoke(state["messages"])]}

        def call_tools(state: State):
            last = state["messages"][-1]
            results = []
            for tc in last.tool_calls:
                if tc["name"] == "search":
                    output = search_tool.invoke(tc["args"])
                    results.append(
                        ToolMessage(content=output, tool_call_id=tc["id"])
                    )
            return {"messages": results}

        graph = StateGraph(State)
        graph.add_node("agent", call_model)
        graph.add_node("tools", call_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    def test_record_react_flow(self, cassette_path: str) -> None:
        """Record a full ReAct loop: LLM→tool call→tool result→LLM→final."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"query": "who created python"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Python was created by Guido van Rossum."),
        ]
        model = FakeMessagesListChatModel(responses=responses)
        app = self._build_react_graph(model)

        with Recorder(save_to=cassette_path) as rec:
            result = app.invoke(
                {"messages": [HumanMessage(content="Who created Python?")]}
            )

        assert result["messages"][-1].content == "Python was created by Guido van Rossum."

        events = rec.trace.events
        # Expect: llm_request, llm_response, tool_call (from AIMessage),
        #         tool_call (BaseTool.invoke), tool_result, llm_request, llm_response
        assert any(e.event_type == EventType.TOOL_CALL for e in events)
        assert any(e.event_type == EventType.TOOL_RESULT for e in events)
        llm_responses = [
            e for e in events if e.event_type == EventType.LLM_RESPONSE
        ]
        assert len(llm_responses) == 2  # two model invocations

    def test_replay_react_flow(self, cassette_path: str) -> None:
        """Replay should produce identical final answer without real LLM."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"query": "who created python"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Python was created by Guido van Rossum."),
        ]
        model = FakeMessagesListChatModel(responses=responses)
        app = self._build_react_graph(model)

        # Record
        with Recorder(save_to=cassette_path):
            original_result = app.invoke(
                {"messages": [HumanMessage(content="Who created Python?")]}
            )

        # Replay with a dummy model that should never be reached
        dummy = FakeMessagesListChatModel(
            responses=[AIMessage(content="SHOULD NOT SEE")]
        )
        app2 = self._build_react_graph(dummy)

        with Replayer(cassette_path):
            replayed_result = app2.invoke(
                {"messages": [HumanMessage(content="Who created Python?")]}
            )

        assert (
            replayed_result["messages"][-1].content
            == original_result["messages"][-1].content
        )

    def test_cassette_persisted_correctly(self, cassette_path: str) -> None:
        """Cassette file roundtrips through YAML correctly."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "test"}, "id": "c1"}
                ],
            ),
            AIMessage(content="Done."),
        ]
        model = FakeMessagesListChatModel(responses=responses)
        app = self._build_react_graph(model)

        with Recorder(save_to=cassette_path):
            app.invoke(
                {"messages": [HumanMessage(content="Test")]}
            )

        trace = load_cassette(cassette_path)
        providers = {e.provider for e in trace.events if e.provider}
        assert "langchain" in providers


# ── Test: conditional branching graph ────────────────────────────────


class TestConditionalGraph:
    """Graph with conditional edges — verifies replay follows same path."""

    def test_record_replay_with_branch(self, cassette_path: str) -> None:
        model = FakeListChatModel(responses=["positive sentiment"])

        def classify(state: State):
            return {"messages": [model.invoke(state["messages"])]}

        def route(state: State):
            last = state["messages"][-1]
            if "positive" in last.content.lower():
                return "happy_path"
            return "sad_path"

        def happy(state: State):
            return state

        def sad(state: State):
            return state

        g = StateGraph(State)
        g.add_node("classify", classify)
        g.add_node("happy_path", happy)
        g.add_node("sad_path", sad)
        g.add_edge(START, "classify")
        g.add_conditional_edges(
            "classify", route, {"happy_path": "happy_path", "sad_path": "sad_path"}
        )
        g.add_edge("happy_path", END)
        g.add_edge("sad_path", END)
        app = g.compile()

        with Recorder(save_to=cassette_path) as rec:
            result = app.invoke(
                {"messages": [HumanMessage(content="I love this!")]}
            )

        assert "positive" in result["messages"][-1].content.lower()

        # Replay
        model2 = FakeListChatModel(responses=["WRONG"])
        g2 = StateGraph(State)

        def classify2(state: State):
            return {"messages": [model2.invoke(state["messages"])]}

        g2.add_node("classify", classify2)
        g2.add_node("happy_path", happy)
        g2.add_node("sad_path", sad)
        g2.add_edge(START, "classify")
        g2.add_conditional_edges(
            "classify", route, {"happy_path": "happy_path", "sad_path": "sad_path"}
        )
        g2.add_edge("happy_path", END)
        g2.add_edge("sad_path", END)
        app2 = g2.compile()

        with Replayer(cassette_path):
            result2 = app2.invoke(
                {"messages": [HumanMessage(content="I love this!")]}
            )

        assert result2["messages"][-1].content == result["messages"][-1].content


# ── Test: multiple tools ─────────────────────────────────────────────


class TestMultipleTools:
    """ReAct agent with more than one tool available."""

    @staticmethod
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))

    @staticmethod
    @tool
    def dictionary(word: str) -> str:
        """Look up a word definition."""
        return f"{word}: a common English word."

    def test_multi_tool_record_replay(self, cassette_path: str) -> None:
        calculator_tool = self.calculator
        dictionary_tool = self.dictionary

        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "dictionary", "args": {"word": "python"}, "id": "c2"},
                ],
            ),
            AIMessage(content="2+2=4 and python is a common English word."),
        ]
        model = FakeMessagesListChatModel(responses=responses)

        def should_continue(state: State):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        def call_model(state: State):
            return {"messages": [model.invoke(state["messages"])]}

        def call_tools(state: State):
            last = state["messages"][-1]
            results = []
            for tc in last.tool_calls:
                if tc["name"] == "calculator":
                    out = calculator_tool.invoke(tc["args"])
                elif tc["name"] == "dictionary":
                    out = dictionary_tool.invoke(tc["args"])
                else:
                    out = "unknown tool"
                results.append(ToolMessage(content=out, tool_call_id=tc["id"]))
            return {"messages": results}

        graph = StateGraph(State)
        graph.add_node("agent", call_model)
        graph.add_node("tools", call_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        graph.add_edge("tools", "agent")
        app = graph.compile()

        # Record
        with Recorder(save_to=cassette_path) as rec:
            result = app.invoke(
                {"messages": [HumanMessage(content="What is 2+2 and define python?")]}
            )

        final = result["messages"][-1].content
        assert "4" in final and "python" in final

        llm_responses = [
            e for e in rec.trace.events if e.event_type == EventType.LLM_RESPONSE
        ]
        assert len(llm_responses) == 3  # three model calls

        # Replay
        dummy = FakeMessagesListChatModel(
            responses=[AIMessage(content="NO")]
        )

        def call_model2(state: State):
            return {"messages": [dummy.invoke(state["messages"])]}

        graph2 = StateGraph(State)
        graph2.add_node("agent", call_model2)
        graph2.add_node("tools", call_tools)
        graph2.add_edge(START, "agent")
        graph2.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        graph2.add_edge("tools", "agent")
        app2 = graph2.compile()

        with Replayer(cassette_path):
            result2 = app2.invoke(
                {"messages": [HumanMessage(content="What is 2+2 and define python?")]}
            )

        assert result2["messages"][-1].content == final
