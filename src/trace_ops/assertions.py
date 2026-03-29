"""Budget and behavioural assertions for agent traces.

These helpers let you guard against cost overruns, token bloat,
excessive LLM round-trips, and infinite tool-call loops directly
inside your test suite.

Usage::

    from trace_ops.assertions import assert_cost_under, assert_no_loops

    with Recorder() as rec:
        agent.run("Summarize the report")

    assert_cost_under(rec.trace, max_usd=0.50)
    assert_no_loops(rec.trace)
"""

from __future__ import annotations

from trace_ops._types import EventType, Trace


class BudgetExceededError(AssertionError):
    """Raised when an agent trace exceeds a defined budget."""


class AgentLoopError(AssertionError):
    """Raised when an agent trace exhibits loop-like behaviour."""


# ── Public assertion functions ──────────────────────────────────────


def assert_cost_under(trace: Trace, *, max_usd: float) -> None:
    """Assert that total cost of a trace is within budget.

    Args:
        trace: The recorded agent trace.
        max_usd: Maximum allowed cost in US dollars.

    Raises:
        BudgetExceededError: If ``trace.total_cost_usd`` exceeds *max_usd*.
    """
    if trace.total_cost_usd > max_usd:
        raise BudgetExceededError(
            f"Trace cost ${trace.total_cost_usd:.4f} exceeds budget of ${max_usd:.4f}.\n"
            f"The agent made {trace.total_llm_calls} LLM calls "
            f"using {trace.total_tokens:,} tokens.\n"
            f"Optimise prompts or reduce tool-call loops to lower cost."
        )


def assert_tokens_under(trace: Trace, *, max_tokens: int) -> None:
    """Assert that total token usage is within a limit.

    Args:
        trace: The recorded agent trace.
        max_tokens: Maximum allowed token count (input + output).

    Raises:
        BudgetExceededError: If ``trace.total_tokens`` exceeds *max_tokens*.
    """
    if trace.total_tokens > max_tokens:
        raise BudgetExceededError(
            f"Trace used {trace.total_tokens:,} tokens, "
            f"exceeding limit of {max_tokens:,}.\n"
            f"The agent made {trace.total_llm_calls} LLM calls.\n"
            f"Reduce prompt size or limit tool-call depth."
        )


def assert_max_llm_calls(trace: Trace, *, max_calls: int) -> None:
    """Assert that the agent didn't make too many LLM round-trips.

    Args:
        trace: The recorded agent trace.
        max_calls: Maximum allowed LLM calls.

    Raises:
        BudgetExceededError: If ``trace.total_llm_calls`` exceeds *max_calls*.
    """
    if trace.total_llm_calls > max_calls:
        raise BudgetExceededError(
            f"Trace made {trace.total_llm_calls} LLM calls, "
            f"exceeding limit of {max_calls}.\n"
            f"Trajectory: {' → '.join(trace.trajectory)}\n"
            f"The agent may be stuck in a loop or using an inefficient strategy."
        )


def assert_no_loops(
    trace: Trace,
    *,
    max_consecutive_same_tool: int = 3,
) -> None:
    """Assert that the trace doesn't contain tool-call loops.

    Scans for *N* consecutive ``TOOL_CALL`` events with the same
    ``tool_name``.  Such runs typically indicate the agent is stuck
    retrying the same action.

    Args:
        trace: The recorded agent trace.
        max_consecutive_same_tool: Maximum allowed consecutive calls
            to the same tool before raising.

    Raises:
        AgentLoopError: If a run of same-tool calls exceeds the limit.
    """
    tool_events = [
        e for e in trace.events if e.event_type == EventType.TOOL_CALL
    ]
    if not tool_events:
        return

    consecutive = 1
    for i in range(1, len(tool_events)):
        if tool_events[i].tool_name == tool_events[i - 1].tool_name:
            consecutive += 1
            if consecutive > max_consecutive_same_tool:
                raise AgentLoopError(
                    f"Detected {consecutive} consecutive calls to tool "
                    f"'{tool_events[i].tool_name}' "
                    f"(limit: {max_consecutive_same_tool}).\n"
                    f"The agent may be stuck in an infinite loop.\n"
                    f"Check the agent's exit conditions or add loop guards."
                )
        else:
            consecutive = 1
