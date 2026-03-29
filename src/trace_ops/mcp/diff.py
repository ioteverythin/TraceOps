"""MCP-aware trace diffing.

Compares MCP tool sequences between two traces and reports changes in
server usage, tool call order, argument changes, and schema diffs.

Usage::

    from trace_ops import load_cassette
    from trace_ops.mcp.diff import diff_mcp

    old = load_cassette("cassettes/v1.yaml")
    new = load_cassette("cassettes/v2.yaml")
    result = diff_mcp(old, new)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops._types import Trace


@dataclass
class MCPToolDiff:
    """Diff of a single MCP tool call."""

    index: int
    status: str  # "added" | "removed" | "changed" | "unchanged"
    old_call: dict[str, Any] | None = None
    new_call: dict[str, Any] | None = None


@dataclass
class MCPDiffResult:
    """Complete MCP diff result."""

    tool_diffs: list[MCPToolDiff] = field(default_factory=list)
    new_servers: list[str] = field(default_factory=list)
    removed_servers: list[str] = field(default_factory=list)
    sequence_changed: bool = False

    def summary(self) -> str:
        lines = ["MCP Changes:"]

        if self.new_servers:
            for s in self.new_servers:
                lines.append(f"  ⚠ New MCP server used: '{s}'")
        if self.removed_servers:
            for s in self.removed_servers:
                lines.append(f"  ⚠ MCP server removed: '{s}'")

        changed = [d for d in self.tool_diffs if d.status != "unchanged"]
        if changed:
            for d in changed:
                old_name = (d.old_call or {}).get("tool_name", "?")
                new_name = (d.new_call or {}).get("tool_name", "?")
                if d.status == "added":
                    lines.append(f"  + Added tool call: {new_name}")
                elif d.status == "removed":
                    lines.append(f"  - Removed tool call: {old_name}")
                else:
                    old_args = (d.old_call or {}).get("arguments", {})
                    new_args = (d.new_call or {}).get("arguments", {})
                    if old_args != new_args:
                        lines.append(
                            f"  ⚠ Tool call changed: {old_name}\n"
                            f"    Old args: {old_args}\n"
                            f"    New args: {new_args}"
                        )
        else:
            lines.append("  ✅ Tool call sequence unchanged")

        return "\n".join(lines)


def diff_mcp(old_trace: "Trace", new_trace: "Trace") -> MCPDiffResult:
    """Compare MCP tool sequences between two traces.

    Args:
        old_trace: Baseline trace.
        new_trace: Current trace.

    Returns:
        :class:`MCPDiffResult` describing all MCP-level changes.
    """
    from trace_ops._types import EventType

    def _tool_calls(trace: "Trace") -> list[dict[str, Any]]:
        return [
            {
                "server_name": e.server_name or "",
                "tool_name": e.tool_name or "",
                "arguments": e.arguments or {},
            }
            for e in trace.events
            if e.event_type == EventType.MCP_TOOL_CALL
        ]

    old_calls = _tool_calls(old_trace)
    new_calls = _tool_calls(new_trace)

    old_servers = {c["server_name"] for c in old_calls}
    new_servers = {c["server_name"] for c in new_calls}

    tool_diffs: list[MCPToolDiff] = []
    max_len = max(len(old_calls), len(new_calls))

    for i in range(max_len):
        old_c = old_calls[i] if i < len(old_calls) else None
        new_c = new_calls[i] if i < len(new_calls) else None

        if old_c is None:
            status = "added"
        elif new_c is None:
            status = "removed"
        elif old_c == new_c:
            status = "unchanged"
        else:
            status = "changed"

        tool_diffs.append(MCPToolDiff(index=i, status=status, old_call=old_c, new_call=new_c))

    sequence_changed = any(d.status != "unchanged" for d in tool_diffs)

    return MCPDiffResult(
        tool_diffs=tool_diffs,
        new_servers=sorted(new_servers - old_servers),
        removed_servers=sorted(old_servers - new_servers),
        sequence_changed=sequence_changed,
    )
