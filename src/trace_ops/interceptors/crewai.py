"""CrewAI interceptor for traceops.

Captures crew-level execution metadata, inter-agent delegation, and
per-agent LLM / tool activity.  The underlying LLM calls are still
captured by the OpenAI / Anthropic / LiteLLM interceptors; this module
adds the *coordination layer* on top.

This module is **optional** — it only activates when ``crewai`` is
installed.
"""

from __future__ import annotations

import time
from typing import Any


def install_crewai_record_patches(
    recorder: Any,
    patches: list[Any],
) -> None:
    """Install CrewAI recording interceptors.

    Args:
        recorder: The :class:`~trace_ops.recorder.Recorder` instance.
        patches: The recorder's ``_patches`` list to append to.
    """
    _patch_crew_kickoff(recorder, patches)
    _patch_agent_execute_task(recorder, patches)


def install_crewai_replay_patches(
    replayer: Any,
    patches: list[Any],
) -> None:
    """Install CrewAI replay interceptors.

    CrewAI replay is primarily handled by the underlying LLM interceptors.
    This function patches crew-level entry points so that the execution
    flow is consistent during replay.

    Args:
        replayer: The :class:`~trace_ops.replayer.Replayer` instance.
        patches: The replayer's ``_patches`` list to append to.
    """
    # CrewAI replay relies on the underlying LLM interceptors.
    # We only need to capture crew-level metadata during recording.
    pass


# ── Recording patches ───────────────────────────────────────────────


def _patch_crew_kickoff(recorder: Any, patches: list[Any]) -> None:
    """Patch ``Crew.kickoff()`` to record crew execution metadata."""
    try:
        from crewai import Crew

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch, _safe_serialize

        original = Crew.kickoff
        rec = recorder

        def patched_kickoff(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
            # Record crew-level metadata
            agents_info = []
            for agent in getattr(self_inner, "agents", []):
                agents_info.append({
                    "name": getattr(agent, "name", "unknown"),
                    "role": getattr(agent, "role", "unknown"),
                    "goal": getattr(agent, "goal", ""),
                })

            tasks_info = []
            for task in getattr(self_inner, "tasks", []):
                tasks_info.append({
                    "description": getattr(task, "description", ""),
                    "agent": getattr(
                        getattr(task, "agent", None), "name", "unassigned"
                    ),
                })

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="crew_kickoff",
                reasoning=f"Starting crew with {len(agents_info)} agents",
                metadata={
                    "framework": "crewai",
                    "agents": agents_info,
                    "tasks": tasks_info,
                    "process": str(getattr(self_inner, "process", "sequential")),
                },
            ))

            start = time.monotonic()
            try:
                result = original(self_inner, *args, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"framework": "crewai", "phase": "crew_kickoff"},
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="crew_complete",
                reasoning=f"Crew finished in {elapsed:.0f}ms",
                duration_ms=elapsed,
                metadata={
                    "framework": "crewai",
                    "result_preview": str(_safe_serialize(result))[:500],
                },
            ))

            return result

        patches.append(_Patch(Crew, "kickoff", original, patched_kickoff))
        Crew.kickoff = patched_kickoff  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_agent_execute_task(recorder: Any, patches: list[Any]) -> None:
    """Patch CrewAI's agent task execution to capture delegation events."""
    try:
        from crewai import Agent

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch, _safe_serialize

        original = Agent.execute_task
        rec = recorder

        def patched_execute_task(
            self_inner: Any,
            task: Any,
            context: Any = None,
            tools: Any = None,
        ) -> Any:
            agent_name = getattr(self_inner, "name", "unknown")
            agent_role = getattr(self_inner, "role", "unknown")
            task_desc = getattr(task, "description", "")[:200]

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision=f"agent_execute:{agent_name}",
                reasoning=f"Agent '{agent_role}' executing task: {task_desc}",
                metadata={
                    "framework": "crewai",
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "task_description": task_desc,
                },
            ))

            start = time.monotonic()
            try:
                result = original(self_inner, task, context, tools)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={
                        "framework": "crewai",
                        "agent_name": agent_name,
                        "phase": "execute_task",
                    },
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision=f"agent_complete:{agent_name}",
                reasoning=f"Agent '{agent_role}' finished in {elapsed:.0f}ms",
                duration_ms=elapsed,
                metadata={
                    "framework": "crewai",
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "result_preview": str(_safe_serialize(result))[:500],
                },
            ))

            return result

        patches.append(_Patch(Agent, "execute_task", original, patched_execute_task))
        Agent.execute_task = patched_execute_task  # type: ignore[assignment]

    except ImportError:
        pass
