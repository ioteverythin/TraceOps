"""MCP (Model Context Protocol) testing add-on for traceops.

Provides automatic recording and replay of MCP tool calls so tests
run without a live MCP server.

Usage::

    from trace_ops import Recorder, Replayer

    # Record — MCP tool calls captured automatically
    with Recorder(save_to="cassettes/mcp_test.yaml", intercept_mcp=True) as rec:
        result = agent.run("Read the report and summarise it")

    # Replay — no MCP server needed
    with Replayer("cassettes/mcp_test.yaml"):
        result = agent.run("Read the report and summarise it")
"""

from trace_ops.mcp.diff import MCPDiffResult, MCPToolDiff, diff_mcp
from trace_ops.mcp.events import MCPServerConnect, MCPToolCall, MCPToolResult
from trace_ops.mcp.interceptor import patch_mcp

__all__ = [
    "MCPServerConnect",
    "MCPToolCall",
    "MCPToolResult",
    "patch_mcp",
    "diff_mcp",
    "MCPDiffResult",
    "MCPToolDiff",
]
