"""LangGraph ReAct agent example with recording and replay.

A multi-step agent that calls tools and reasons about results.

Records:
    python examples/langgraph_react.py

Replays:
    python examples/langgraph_react.py --replay
"""

from __future__ import annotations

import sys

from trace_ops import Recorder, Replayer


def create_react_graph():
    """Build a simple LangGraph ReAct agent.
    
    In a real scenario:
        from langgraph.graph import StateGraph, START, END
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage
        ...
    """
    pass


def main(replay: bool = False) -> None:
    """Run LangGraph agent with recording or replay."""
    cassette = "cassettes/langgraph_react.yaml"

    if replay:
        print("🔄 REPLAY MODE — LangGraph ReAct\n")
        try:
            with Replayer(cassette):
                app = create_react_graph()
                # result = app.invoke({"messages": [HumanMessage(...)]})
                print(f"Replayed from: {cassette}")
        except FileNotFoundError:
            print(f"Cassette not found: {cassette}")
            sys.exit(1)
    else:
        print("🎙️  RECORD MODE — LangGraph ReAct\n")
        with Recorder(save_to=cassette, description="LangGraph ReAct agent"):
            app = create_react_graph()
            # result = app.invoke({"messages": [HumanMessage(...)]})
            print(f"Cassette saved to: {cassette}")


if __name__ == "__main__":
    replay = "--replay" in sys.argv
    main(replay=replay)
