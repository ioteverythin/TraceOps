"""LangGraph interceptor for traceops.

Patches ``Pregel.invoke()`` / ``ainvoke()`` / ``stream()`` / ``astream()``
to record **graph-level** events:

* ``AGENT_DECISION`` with ``decision="graph_start"`` — captures graph name,
  node topology, and input state.
* ``AGENT_DECISION`` with ``decision="graph_end"`` — captures the output
  state and total duration.

This complements the LangChain interceptor which captures individual
LLM calls *inside* graph nodes.  Together they give a complete picture:
LangGraph events show *which graph ran and for how long*, while LangChain
events show *every LLM call that happened inside*.

This module is **optional** — it only activates when ``langgraph`` is
installed.
"""

from __future__ import annotations

import time
from typing import Any, Iterator


# ── Public API ───────────────────────────────────────────────────────


def install_langgraph_record_patches(
    recorder: Any,
    patches: list[Any],
) -> None:
    """Install LangGraph recording interceptors on ``Pregel``."""
    _patch_pregel_invoke(recorder, patches)
    _patch_pregel_ainvoke(recorder, patches)
    _patch_pregel_stream(recorder, patches)
    _patch_pregel_astream(recorder, patches)


def install_langgraph_replay_patches(
    replayer: Any,
    patches: list[Any],
) -> None:
    """LangGraph replay is a no-op.

    Replay of individual LLM calls is handled by the LangChain
    interceptor (``BaseChatModel.invoke`` patches).  The graph-level
    events are purely observational and don't need to be replayed.
    """
    pass


# ── Helpers ──────────────────────────────────────────────────────────


def _graph_metadata(graph: Any) -> dict[str, Any]:
    """Extract useful metadata from a compiled LangGraph ``Pregel`` instance."""
    meta: dict[str, Any] = {
        "graph_type": type(graph).__name__,
        "graph_name": getattr(graph, "name", None) or "LangGraph",
    }
    try:
        graph_def = graph.get_graph()
        meta["nodes"] = [n.id for n in graph_def.nodes.values()]
        meta["edges"] = [
            {"source": e.source, "target": e.target}
            for e in graph_def.edges
        ]
    except Exception:
        pass
    return meta


def _safe_input(inp: Any) -> Any:
    """Best-effort serialisable snapshot of the graph input."""
    try:
        from trace_ops.recorder import _safe_serialize
        return _safe_serialize(inp)
    except Exception:
        return str(inp)


# ── Recording patches ───────────────────────────────────────────────


def _patch_pregel_invoke(recorder: Any, patches: list[Any]) -> None:
    try:
        from langgraph.pregel.main import Pregel

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch

        original = Pregel.invoke
        rec = recorder

        def patched_invoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            meta = _graph_metadata(self_inner)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_start",
                reasoning=f"LangGraph invoke: {meta.get('graph_name', 'unknown')}",
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "input": _safe_input(input),
                },
            ))

            start = time.monotonic()
            try:
                result = original(self_inner, input, config, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langgraph",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"framework": "langgraph", **meta},
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_end",
                reasoning=f"LangGraph completed in {elapsed:.1f}ms",
                duration_ms=elapsed,
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "output": _safe_input(result),
                },
            ))
            return result

        patches.append(_Patch(Pregel, "invoke", original, patched_invoke))
        Pregel.invoke = patched_invoke  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_pregel_ainvoke(recorder: Any, patches: list[Any]) -> None:
    try:
        from langgraph.pregel.main import Pregel

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch

        original = Pregel.ainvoke
        rec = recorder

        async def patched_ainvoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            meta = _graph_metadata(self_inner)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_start",
                reasoning=f"LangGraph ainvoke: {meta.get('graph_name', 'unknown')}",
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "input": _safe_input(input),
                },
            ))

            start = time.monotonic()
            try:
                result = await original(self_inner, input, config, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langgraph",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"framework": "langgraph", **meta},
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_end",
                reasoning=f"LangGraph completed in {elapsed:.1f}ms",
                duration_ms=elapsed,
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "output": _safe_input(result),
                },
            ))
            return result

        patches.append(_Patch(Pregel, "ainvoke", original, patched_ainvoke))
        Pregel.ainvoke = patched_ainvoke  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_pregel_stream(recorder: Any, patches: list[Any]) -> None:
    try:
        from langgraph.pregel.main import Pregel

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch

        original = Pregel.stream
        rec = recorder

        def patched_stream(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Iterator:
            meta = _graph_metadata(self_inner)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_stream_start",
                reasoning=f"LangGraph stream: {meta.get('graph_name', 'unknown')}",
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "input": _safe_input(input),
                },
            ))

            start = time.monotonic()
            chunk_count = 0
            try:
                for chunk in original(self_inner, input, config, **kwargs):
                    chunk_count += 1
                    yield chunk
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langgraph",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"framework": "langgraph", **meta},
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_stream_end",
                reasoning=f"LangGraph stream completed: {chunk_count} chunks in {elapsed:.1f}ms",
                duration_ms=elapsed,
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "chunk_count": chunk_count,
                },
            ))

        patches.append(_Patch(Pregel, "stream", original, patched_stream))
        Pregel.stream = patched_stream  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_pregel_astream(recorder: Any, patches: list[Any]) -> None:
    try:
        from langgraph.pregel.main import Pregel

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch

        original = Pregel.astream
        rec = recorder

        async def patched_astream(self_inner: Any, input: Any, config: Any = None, **kwargs: Any):
            meta = _graph_metadata(self_inner)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_stream_start",
                reasoning=f"LangGraph astream: {meta.get('graph_name', 'unknown')}",
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "input": _safe_input(input),
                },
            ))

            start = time.monotonic()
            chunk_count = 0
            try:
                async for chunk in original(self_inner, input, config, **kwargs):
                    chunk_count += 1
                    yield chunk
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langgraph",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"framework": "langgraph", **meta},
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            rec._trace.add_event(TraceEvent(
                event_type=EventType.AGENT_DECISION,
                decision="graph_stream_end",
                reasoning=f"LangGraph astream completed: {chunk_count} chunks in {elapsed:.1f}ms",
                duration_ms=elapsed,
                metadata={
                    "framework": "langgraph",
                    **meta,
                    "chunk_count": chunk_count,
                },
            ))

        patches.append(_Patch(Pregel, "astream", original, patched_astream))
        Pregel.astream = patched_astream  # type: ignore[assignment]

    except ImportError:
        pass
