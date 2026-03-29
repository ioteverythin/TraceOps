"""Basic OpenAI chat example with recording and replay.

Records:
    python examples/openai_basic.py

Replays:
    python examples/openai_basic.py --replay
"""

from __future__ import annotations

import sys

from trace_ops import Recorder, Replayer


def main(replay: bool = False) -> None:
    """Simple chat loop using OpenAI."""
    # This is a demo — replace with real OpenAI calls
    if replay:
        print("🔄 REPLAY MODE — using recorded cassette\n")
        cassette = "cassettes/openai_chat.yaml"
        try:
            with Replayer(cassette) as rep:
                # Your agent code would go here
                # Replayer will intercept calls and return recorded responses
                print(f"Replayed from: {cassette}")
        except FileNotFoundError:
            print(f"Cassette not found: {cassette}")
            print("Run without --replay to record first.")
            sys.exit(1)
    else:
        print("🎙️  RECORD MODE — saving responses to cassette\n")
        cassette = "cassettes/openai_chat.yaml"
        with Recorder(save_to=cassette, description="OpenAI chat example"):
            # Your agent code would go here
            # Recorder will intercept OpenAI calls and save responses
            print(f"Cassette saved to: {cassette}")


if __name__ == "__main__":
    replay = "--replay" in sys.argv
    main(replay=replay)
