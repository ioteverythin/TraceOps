"""MCP event dataclasses for traceops.

These mirror the cassette YAML schema for MCP events:
  - ``mcp_server_connect``
  - ``mcp_tool_call``
  - ``mcp_tool_result``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPServerConnect:
    """Recorded MCP server connection event."""

    server_name: str
    server_url: str = ""
    capabilities: list[str] = field(default_factory=list)


@dataclass
class MCPToolCall:
    """Recorded MCP tool invocation."""

    server_name: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class MCPToolResult:
    """Recorded MCP tool result."""

    server_name: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    duration_ms: float = 0.0
