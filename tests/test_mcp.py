"""Tests for MCP events, interceptor, and diff."""

from __future__ import annotations

from trace_ops._types import EventType, Trace, TraceEvent

# ── MCPServerConnect / MCPToolCall / MCPToolResult dataclasses ─────────────


def test_mcp_server_connect_fields():
    from trace_ops.mcp.events import MCPServerConnect
    ev = MCPServerConnect(
        server_name="my-mcp-server",
        server_url="stdio://myserver",
        capabilities={"tools": True},
    )
    assert ev.server_name == "my-mcp-server"
    assert ev.capabilities["tools"] is True


def test_mcp_tool_call_fields():
    from trace_ops.mcp.events import MCPToolCall
    ev = MCPToolCall(
        server_name="my-mcp-server",
        tool_name="search",
        arguments={"query": "latest news"},
    )
    assert ev.tool_name == "search"
    assert ev.arguments["query"] == "latest news"


def test_mcp_tool_result_fields():
    from trace_ops.mcp.events import MCPToolResult
    ev = MCPToolResult(
        server_name="my-mcp-server",
        tool_name="search",
        result=[{"type": "text", "text": "Result: ..."}],
        is_error=False,
        duration_ms=120.0,
    )
    assert ev.result[0]["type"] == "text"
    assert not ev.is_error
    assert ev.duration_ms == 120.0


# ── Trace.mcp_events property ──────────────────────────────────────────────


def _trace_with_mcp_events() -> Trace:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.MCP_SERVER_CONNECT,
        server_name="srv1",
        server_url="stdio://srv1",
        capabilities={"tools": True},
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "hello"},
    ))
    trace.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_RESULT,
        server_name="srv1",
        tool_name="search",
        result=[{"type": "text", "text": "hello world"}],
        is_error=False,
        duration_ms=50.0,
    ))
    return trace


def test_trace_mcp_events_includes_all_mcp_types():
    trace = _trace_with_mcp_events()
    mcp_events = trace.mcp_events
    assert len(mcp_events) == 3
    types = {e.event_type for e in mcp_events}
    assert EventType.MCP_SERVER_CONNECT in types
    assert EventType.MCP_TOOL_CALL in types
    assert EventType.MCP_TOOL_RESULT in types


# ── diff_mcp() ─────────────────────────────────────────────────────────────


def test_diff_mcp_identical_traces():
    from trace_ops.mcp.diff import diff_mcp

    old = _trace_with_mcp_events()
    new = _trace_with_mcp_events()
    result = diff_mcp(old, new)
    assert not result.sequence_changed


def test_diff_mcp_different_tool_name():
    from trace_ops.mcp.diff import diff_mcp

    old = Trace()
    old.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "hello"},
    ))
    new = Trace()
    new.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="lookup",  # different tool
        arguments={"q": "hello"},
    ))
    result = diff_mcp(old, new)
    assert result.sequence_changed


def test_diff_mcp_different_arguments():
    from trace_ops.mcp.diff import diff_mcp

    old = Trace()
    old.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "hello"},
    ))
    new = Trace()
    new.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "world"},  # different args
    ))
    result = diff_mcp(old, new)
    assert result.sequence_changed


def test_diff_mcp_different_result():
    from trace_ops.mcp.diff import diff_mcp

    old = Trace()
    old.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "v1"},
    ))
    new = Trace()
    new.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "v2"},
    ))
    result = diff_mcp(old, new)
    assert result.sequence_changed


def test_diff_mcp_more_calls_in_new():
    from trace_ops.mcp.diff import diff_mcp

    old = Trace()
    new = Trace()
    new.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={},
    ))
    result = diff_mcp(old, new)
    assert result.sequence_changed


def test_diff_mcp_empty_traces():
    from trace_ops.mcp.diff import diff_mcp

    old = Trace()
    new = Trace()
    result = diff_mcp(old, new)
    assert not result.sequence_changed


def test_diff_mcp_summary():
    from trace_ops.mcp.diff import diff_mcp

    old = Trace()
    old.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="search",
        arguments={"q": "old"},
    ))
    new = Trace()
    new.add_event(TraceEvent(
        event_type=EventType.MCP_TOOL_CALL,
        server_name="srv1",
        tool_name="lookup",
        arguments={"q": "old"},
    ))
    result = diff_mcp(old, new)
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


# ── MCPDiffResult dataclass ────────────────────────────────────────────────


def test_mcp_diff_result_tool_diffs():
    from trace_ops.mcp.diff import MCPDiffResult, MCPToolDiff

    td = MCPToolDiff(
        index=0,
        status="changed",
        old_call={"server_name": "srv1", "tool_name": "search", "arguments": {}},
        new_call={"server_name": "srv1", "tool_name": "lookup", "arguments": {}},
    )
    result = MCPDiffResult(tool_diffs=[td], sequence_changed=True)
    assert len(result.tool_diffs) == 1
    assert result.tool_diffs[0].old_call["tool_name"] == "search"
