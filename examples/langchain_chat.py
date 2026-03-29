"""LangChain chat agent example with recording and replay.

Records:
    python examples/langchain_chat.py

Replays:
    python examples/langchain_chat.py --replay
"""

from __future__ import annotations

import sys

from trace_ops import Recorder, Replayer


def create_agent():
    """Build a simple LangChain agent.
    
    In a real scenario, you'd import and configure:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
    """
    pass


def main(replay: bool = False) -> None:
    """Run LangChain agent with recording or replay."""
    cassette = "cassettes/langchain_agent.yaml"

    if replay:
        print("🔄 REPLAY MODE\n")
        try:
            with Replayer(cassette):
                agent = create_agent()
                # agent.run("What is 2+2?")
                print(f"Replayed from: {cassette}")
        except FileNotFoundError:
            print(f"Cassette not found: {cassette}")
            print("Run without --replay to record first.")
            sys.exit(1)
    else:
        print("🎙️  RECORD MODE\n")
        with Recorder(save_to=cassette, description="LangChain ReAct agent"):
            agent = create_agent()
            # agent.run("What is 2+2?")
            print(f"Cassette saved to: {cassette}")


if __name__ == "__main__":
    replay = "--replay" in sys.argv
    main(replay=replay)
