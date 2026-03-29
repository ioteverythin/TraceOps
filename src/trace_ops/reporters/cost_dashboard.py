"""Cost dashboard — aggregate cost analysis across cassette collections.

Provides both a Rich terminal view and a summary data structure for
programmatic access.  Useful for tracking spend across test suites
or CI runs.

Usage::

    from trace_ops.reporters.cost_dashboard import CostDashboard

    dashboard = CostDashboard("cassettes/")
    dashboard.print()          # Rich terminal output
    summary = dashboard.data   # dict for programmatic use

CLI::

    replay costs cassettes/
    replay costs cassettes/ --top 5 --sort cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelCost:
    """Aggregate cost data for a single model."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CassetteCost:
    """Cost data for a single cassette file."""

    path: str
    llm_calls: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    models: list[str] = field(default_factory=list)


@dataclass
class CostSummary:
    """Complete cost dashboard data."""

    cassette_count: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    by_model: list[ModelCost] = field(default_factory=list)
    by_cassette: list[CassetteCost] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "cassette_count": self.cassette_count,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_model": [
                {
                    "model": m.model,
                    "calls": m.calls,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_tokens": m.total_tokens,
                    "cost_usd": m.cost_usd,
                }
                for m in self.by_model
            ],
            "by_cassette": [
                {
                    "path": c.path,
                    "llm_calls": c.llm_calls,
                    "total_tokens": c.total_tokens,
                    "cost_usd": c.cost_usd,
                    "models": c.models,
                }
                for c in self.by_cassette
            ],
            "errors": self.errors,
        }


class CostDashboard:
    """Aggregate cost analysis across a collection of cassettes.

    Args:
        directory: Path to a directory containing cassette YAML files.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._summary: CostSummary | None = None

    @property
    def data(self) -> CostSummary:
        """Compute (or return cached) cost summary."""
        if self._summary is None:
            self._summary = self._analyse()
        return self._summary

    def _analyse(self) -> CostSummary:
        """Scan cassettes and build the summary."""
        from trace_ops._types import EventType
        from trace_ops.cassette import load_cassette

        summary = CostSummary()
        model_map: dict[str, ModelCost] = {}

        yamls = sorted(self.directory.rglob("*.yaml")) + sorted(
            self.directory.rglob("*.yml")
        )

        for path in yamls:
            try:
                trace = load_cassette(path)
            except Exception as exc:
                summary.errors.append(f"{path}: {exc}")
                continue

            summary.cassette_count += 1
            summary.total_llm_calls += trace.total_llm_calls
            summary.total_tokens += trace.total_tokens
            summary.total_cost_usd += trace.total_cost_usd

            cassette_models: set[str] = set()
            cc = CassetteCost(
                path=str(path.relative_to(self.directory)),
                llm_calls=trace.total_llm_calls,
                total_tokens=trace.total_tokens,
                cost_usd=trace.total_cost_usd,
            )

            for event in trace.events:
                if event.event_type == EventType.LLM_RESPONSE:
                    model_name = event.model or "unknown"
                    cassette_models.add(model_name)

                    if model_name not in model_map:
                        model_map[model_name] = ModelCost(model=model_name)
                    mc = model_map[model_name]
                    mc.calls += 1
                    mc.input_tokens += event.input_tokens or 0
                    mc.output_tokens += event.output_tokens or 0
                    mc.total_tokens += (event.input_tokens or 0) + (
                        event.output_tokens or 0
                    )
                    mc.cost_usd += event.cost_usd or 0.0

            cc.models = sorted(cassette_models)
            summary.by_cassette.append(cc)

        summary.by_model = sorted(
            model_map.values(), key=lambda m: m.cost_usd, reverse=True
        )
        summary.by_cassette.sort(key=lambda c: c.cost_usd, reverse=True)

        return summary

    def print(self, *, top: int = 10, console: Any | None = None) -> None:
        """Print a rich cost dashboard to the terminal.

        Args:
            top: Number of top cassettes to show.
            console: Optional Rich Console instance (uses default if None).
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        con = console or Console()
        s = self.data

        # ── Overview panel ──────────────────────────────────────────
        con.print(
            Panel(
                f"[bold]Cassettes scanned:[/bold] {s.cassette_count}\n"
                f"[bold]Total LLM calls:[/bold]  {s.total_llm_calls}\n"
                f"[bold]Total tokens:[/bold]     {s.total_tokens:,}\n"
                f"[bold]Total cost:[/bold]       ${s.total_cost_usd:.4f}",
                title="💰 Cost Dashboard",
                border_style="green",
            )
        )

        # ── Per-model table ─────────────────────────────────────────
        if s.by_model:
            table = Table(title="Cost by Model")
            table.add_column("Model", style="cyan")
            table.add_column("Calls", justify="right")
            table.add_column("Input Tok", justify="right")
            table.add_column("Output Tok", justify="right")
            table.add_column("Total Tok", justify="right")
            table.add_column("Cost", justify="right", style="green")

            for mc in s.by_model:
                table.add_row(
                    mc.model,
                    str(mc.calls),
                    f"{mc.input_tokens:,}",
                    f"{mc.output_tokens:,}",
                    f"{mc.total_tokens:,}",
                    f"${mc.cost_usd:.4f}",
                )

            con.print(table)

        # ── Top cassettes by cost ───────────────────────────────────
        if s.by_cassette:
            table = Table(title=f"Top {top} Cassettes by Cost")
            table.add_column("Cassette", style="cyan")
            table.add_column("LLM Calls", justify="right")
            table.add_column("Tokens", justify="right")
            table.add_column("Cost", justify="right", style="green")
            table.add_column("Models")

            for cc in s.by_cassette[:top]:
                table.add_row(
                    cc.path,
                    str(cc.llm_calls),
                    f"{cc.total_tokens:,}",
                    f"${cc.cost_usd:.4f}",
                    ", ".join(cc.models),
                )

            con.print(table)

        # ── Errors ──────────────────────────────────────────────────
        if s.errors:
            con.print(f"\n[yellow]⚠ {len(s.errors)} file(s) could not be loaded.[/yellow]")
            for err in s.errors[:5]:
                con.print(f"  [dim]{err}[/dim]")
