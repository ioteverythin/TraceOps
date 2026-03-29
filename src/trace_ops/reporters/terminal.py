"""Time-travel debugger — step through a recorded trace interactively.

Uses Rich to render a terminal UI that lets you navigate forward and
backward through events, inspect prompts, view tool I/O, and
optionally compare against a reference trace.

Launch via CLI::

    replay debug cassettes/test_math.yaml
    replay debug cassettes/v1.yaml --compare cassettes/v2.yaml
    replay debug cassettes/test.yaml --tools-only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import load_cassette

# Event-type → (label, Rich colour)
_EVENT_STYLES: dict[EventType, tuple[str, str]] = {
    EventType.LLM_REQUEST: ("LLM Request", "blue"),
    EventType.LLM_RESPONSE: ("LLM Response", "green"),
    EventType.TOOL_CALL: ("Tool Call", "yellow"),
    EventType.TOOL_RESULT: ("Tool Result", "yellow"),
    EventType.AGENT_DECISION: ("Decision", "magenta"),
    EventType.ERROR: ("Error", "red"),
}


class TraceDebugger:
    """Interactive trace debugger.

    Args:
        trace: The trace to step through.
        compare_trace: Optional reference trace for side-by-side comparison.
        event_filter: If set, only show events of these types.
    """

    def __init__(
        self,
        trace: Trace,
        *,
        compare_trace: Trace | None = None,
        event_filter: set[EventType] | None = None,
    ) -> None:
        self.trace = trace
        self.compare_trace = compare_trace
        self._events = self._filter(trace.events, event_filter)
        self._compare_events = (
            self._filter(compare_trace.events, event_filter)
            if compare_trace
            else []
        )
        self._index = 0
        self.console = Console()

    # ── public API ──

    def run(self) -> None:
        """Enter the interactive stepping loop."""
        if not self._events:
            self.console.print("[dim]No events to display.[/dim]")
            return

        self.console.print(
            Panel(
                self._trace_summary(),
                title="🔁 traceops debugger",
                border_style="cyan",
            )
        )
        self.console.print(
            "[dim]  n/Enter = next · p = prev · q = quit · "
            "g <N> = go to event N · ? = help[/dim]\n"
        )

        while True:
            self._render_current()
            try:
                cmd = input("\n[step] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("n", "", "next"):
                self._step(1)
            elif cmd in ("p", "prev", "back"):
                self._step(-1)
            elif cmd.startswith("g "):
                self._goto(cmd)
            elif cmd == "?":
                self._help()
            else:
                self.console.print(f"[red]Unknown command: {cmd}[/red]")

    # ── rendering ──

    def _render_current(self) -> None:
        event = self._events[self._index]
        label, colour = _EVENT_STYLES.get(
            event.event_type, ("Unknown", "white")
        )
        header = (
            f"[{colour} bold]{label}[/{colour} bold]  "
            f"[dim]#{self._index + 1}/{len(self._events)}[/dim]"
        )
        if event.provider:
            header += f"  provider=[cyan]{event.provider}[/cyan]"
        if event.model:
            header += f"  model=[cyan]{event.model}[/cyan]"
        if event.duration_ms:
            header += f"  [dim]{event.duration_ms:.0f}ms[/dim]"

        self.console.print(header)

        # -- body --
        if event.event_type == EventType.LLM_REQUEST:
            self._render_messages(event)
        elif event.event_type == EventType.LLM_RESPONSE:
            self._render_response(event)
        elif event.event_type == EventType.TOOL_CALL:
            self._render_tool_call(event)
        elif event.event_type == EventType.TOOL_RESULT:
            self._render_tool_result(event)
        elif event.event_type == EventType.AGENT_DECISION:
            self._render_decision(event)
        elif event.event_type == EventType.ERROR:
            self._render_error(event)

        # Token / cost summary
        tokens_parts: list[str] = []
        if event.input_tokens:
            tokens_parts.append(f"in={event.input_tokens}")
        if event.output_tokens:
            tokens_parts.append(f"out={event.output_tokens}")
        if event.cost_usd:
            tokens_parts.append(f"${event.cost_usd:.4f}")
        if tokens_parts:
            self.console.print(f"  [dim]Tokens: {', '.join(tokens_parts)}[/dim]")

        # Compare
        if self._compare_events and self._index < len(self._compare_events):
            comp = self._compare_events[self._index]
            if comp.event_type != event.event_type:
                self.console.print(
                    f"  [red]⚠  Reference has {comp.event_type.value} here[/red]"
                )
            elif comp.model != event.model:
                self.console.print(
                    f"  [yellow]⚠  Reference used model {comp.model}[/yellow]"
                )

    def _render_messages(self, event: TraceEvent) -> None:
        if not event.messages:
            return
        for msg in event.messages[-3:]:  # show last 3 messages
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:300]
            self.console.print(f"  [{_role_colour(role)}]{role}[/{_role_colour(role)}]: {escape(content)}")

    def _render_response(self, event: TraceEvent) -> None:
        if not event.response:
            return
        # Extract content from response
        content = _extract_content(event.response)
        if content:
            self.console.print(Panel(escape(content[:500]), title="Response", border_style="green"))

    def _render_tool_call(self, event: TraceEvent) -> None:
        self.console.print(f"  Tool: [yellow bold]{event.tool_name}[/yellow bold]")
        if event.tool_input:
            import json
            try:
                formatted = json.dumps(event.tool_input, indent=2)[:400]
            except (TypeError, ValueError):
                formatted = str(event.tool_input)[:400]
            self.console.print(f"  Input: {escape(formatted)}")

    def _render_tool_result(self, event: TraceEvent) -> None:
        self.console.print(f"  Tool: [yellow bold]{event.tool_name}[/yellow bold]")
        output = str(event.tool_output)[:400] if event.tool_output else "(empty)"
        self.console.print(f"  Output: {escape(output)}")

    def _render_decision(self, event: TraceEvent) -> None:
        self.console.print(f"  Decision: [magenta bold]{event.decision}[/magenta bold]")
        if event.reasoning:
            self.console.print(f"  Reasoning: {escape(event.reasoning[:300])}")

    def _render_error(self, event: TraceEvent) -> None:
        self.console.print(f"  [red bold]{event.error_type}: {escape(str(event.error_message)[:300])}[/red bold]")

    # ── navigation ──

    def _step(self, delta: int) -> None:
        new = self._index + delta
        if 0 <= new < len(self._events):
            self._index = new
        else:
            self.console.print("[dim]Already at the boundary.[/dim]")

    def _goto(self, cmd: str) -> None:
        try:
            n = int(cmd.split()[1]) - 1
            if 0 <= n < len(self._events):
                self._index = n
            else:
                self.console.print("[red]Event number out of range.[/red]")
        except (IndexError, ValueError):
            self.console.print("[red]Usage: g <event-number>[/red]")

    def _help(self) -> None:
        self.console.print(
            "\n[bold]Commands:[/bold]\n"
            "  n / Enter  — next event\n"
            "  p          — previous event\n"
            "  g <N>      — go to event N\n"
            "  q          — quit\n"
        )

    # ── helpers ──

    def _trace_summary(self) -> str:
        t = self.trace
        lines = [
            f"Trace ID: {t.trace_id}",
            f"Events: {len(self._events)}  "
            f"(LLM: {t.total_llm_calls}, Tools: {t.total_tool_calls})",
            f"Tokens: {t.total_tokens:,}  Cost: ${t.total_cost_usd:.4f}  "
            f"Duration: {t.total_duration_ms:.0f}ms",
            f"Fingerprint: {t.fingerprint()}",
        ]
        if self.compare_trace:
            lines.append(
                f"Comparing against: {self.compare_trace.trace_id} "
                f"({len(self._compare_events)} events)"
            )
        return "\n".join(lines)

    @staticmethod
    def _filter(
        events: list[TraceEvent],
        event_filter: set[EventType] | None,
    ) -> list[TraceEvent]:
        if event_filter is None:
            return list(events)
        return [e for e in events if e.event_type in event_filter]


# ── Module-level helpers ────────────────────────────────────────────


def run_debugger(
    cassette_path: str,
    *,
    compare_path: str | None = None,
    event_filter: set[EventType] | None = None,
) -> None:
    """Load a cassette and launch the interactive debugger.

    Args:
        cassette_path: Path to the YAML cassette file.
        compare_path: Optional second cassette for comparison.
        event_filter: If set, only show these event types.
    """
    trace = load_cassette(cassette_path)
    compare_trace = load_cassette(compare_path) if compare_path else None

    debugger = TraceDebugger(
        trace, compare_trace=compare_trace, event_filter=event_filter
    )
    debugger.run()


def _role_colour(role: str) -> str:
    return {
        "system": "dim",
        "user": "cyan",
        "assistant": "green",
        "tool": "yellow",
    }.get(role, "white")


def _extract_content(response: dict[str, Any]) -> str | None:
    """Extract text content from a provider response dict."""
    # OpenAI-style
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return msg.get("content")
    # Anthropic-style
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text")
    # Generic
    return response.get("content")
