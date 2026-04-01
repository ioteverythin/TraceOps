"""MCP protocol interceptor.

Patches ``mcp.ClientSession.call_tool()`` (and its async counterpart) so
that every MCP tool call is automatically recorded as ``mcp_tool_call`` /
``mcp_tool_result`` events in the trace.

Silently does nothing if the ``mcp`` package is not installed.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_mcp(recorder: Recorder) -> None:
    """Install MCP interceptor on ``mcp.ClientSession``."""
    try:
        from mcp import ClientSession
    except ImportError:
        return

    _patch_sync(recorder, ClientSession)
    _patch_async(recorder, ClientSession)


def _patch_sync(recorder: Recorder, ClientSession: Any) -> None:
    if not hasattr(ClientSession, "call_tool"):
        return

    original = ClientSession.call_tool

    @functools.wraps(original)
    def patched(self_inner: Any, tool_name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        from trace_ops._types import EventType, TraceEvent

        server_name = getattr(self_inner, "_server_name", "") or "mcp"
        recorder._trace.add_event(TraceEvent(
            event_type=EventType.MCP_TOOL_CALL,
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments or {},
        ))

        t0 = time.perf_counter()
        try:
            result = original(self_inner, tool_name, arguments, **kwargs)
        except Exception as exc:
            recorder._trace.add_event(TraceEvent(
                event_type=EventType.MCP_TOOL_RESULT,
                server_name=server_name,
                tool_name=tool_name,
                result=str(exc),
                is_error=True,
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
            raise

        duration_ms = (time.perf_counter() - t0) * 1000
        recorder._trace.add_event(TraceEvent(
            event_type=EventType.MCP_TOOL_RESULT,
            server_name=server_name,
            tool_name=tool_name,
            result=_serialize_result(result),
            is_error=False,
            duration_ms=duration_ms,
        ))
        return result

    ClientSession.call_tool = patched  # type: ignore[method-assign]
    recorder._mcp_patches.append(("mcp.ClientSession.call_tool", original, ClientSession, "call_tool"))


def _patch_async(recorder: Recorder, ClientSession: Any) -> None:
    """Patch the async variant if it exists."""
    if not hasattr(ClientSession, "call_tool_async"):
        return

    original_async = ClientSession.call_tool_async

    @functools.wraps(original_async)
    async def patched_async(
        self_inner: Any, tool_name: str, arguments: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        from trace_ops._types import EventType, TraceEvent

        server_name = getattr(self_inner, "_server_name", "") or "mcp"
        recorder._trace.add_event(TraceEvent(
            event_type=EventType.MCP_TOOL_CALL,
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments or {},
        ))

        t0 = time.perf_counter()
        try:
            result = await original_async(self_inner, tool_name, arguments, **kwargs)
        except Exception as exc:
            recorder._trace.add_event(TraceEvent(
                event_type=EventType.MCP_TOOL_RESULT,
                server_name=server_name,
                tool_name=tool_name,
                result=str(exc),
                is_error=True,
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
            raise

        duration_ms = (time.perf_counter() - t0) * 1000
        recorder._trace.add_event(TraceEvent(
            event_type=EventType.MCP_TOOL_RESULT,
            server_name=server_name,
            tool_name=tool_name,
            result=_serialize_result(result),
            is_error=False,
            duration_ms=duration_ms,
        ))
        return result

    ClientSession.call_tool_async = patched_async  # type: ignore[method-assign]
    recorder._mcp_patches.append(("mcp.ClientSession.call_tool_async", original_async, ClientSession, "call_tool_async"))


def _serialize_result(result: Any) -> Any:
    """Convert MCP result to a JSON-serialisable value."""
    if result is None:
        return None
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)
    try:
        return str(result)
    except Exception:
        return "<unserializable>"
