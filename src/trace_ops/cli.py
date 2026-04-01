"""CLI entry point for traceops.

Provides the `replay` command with subcommands:
  replay diff <old> <new>    — Compare two cassette files
  replay inspect <cassette>  — Show trace summary
  replay export <cassette>   — Export to JSON/HTML
  replay debug <cassette>    — Interactive time-travel debugger
  replay report <cassette>   — Generate HTML trace report
  replay ls [dir]            — List cassettes with summary stats
  replay prune [dir]         — Delete stale cassettes
  replay stats [dir]         — Aggregate stats across cassettes
  replay validate <cassette> — Check cassette integrity
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.5.0", prog_name="traceops")
def main():
    """🔁 traceops — record and replay LLM agent traces."""
    pass


@main.command()
@click.argument("cassette_path")
def inspect(cassette_path: str):
    """Inspect a cassette file and show a summary."""
    from trace_ops._types import EventType
    from trace_ops.cassette import load_cassette

    try:
        trace = load_cassette(cassette_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    # Summary panel
    console.print(Panel(
        f"[bold]Trace ID:[/bold] {trace.trace_id}\n"
        f"[bold]LLM calls:[/bold] {trace.total_llm_calls}\n"
        f"[bold]Tool calls:[/bold] {trace.total_tool_calls}\n"
        f"[bold]Total tokens:[/bold] {trace.total_tokens:,}\n"
        f"[bold]Total cost:[/bold] ${trace.total_cost_usd:.4f}\n"
        f"[bold]Duration:[/bold] {trace.total_duration_ms:.0f}ms\n"
        f"[bold]Fingerprint:[/bold] {trace.fingerprint()}",
        title=f"Cassette: {cassette_path}",
    ))

    # Trajectory
    console.print("\n[bold]Trajectory:[/bold]")
    for i, step in enumerate(trace.trajectory, 1):
        console.print(f"  {i}. {step}")

    # Events table
    console.print()
    table = Table(title="Events")
    table.add_column("#", style="dim")
    table.add_column("Type")
    table.add_column("Provider")
    table.add_column("Model / Tool")
    table.add_column("Tokens")
    table.add_column("Duration")

    for i, event in enumerate(trace.events, 1):
        tokens = ""
        if event.input_tokens or event.output_tokens:
            tokens = f"{event.input_tokens or 0}/{event.output_tokens or 0}"

        duration = f"{event.duration_ms:.0f}ms" if event.duration_ms else ""

        name = event.model or event.tool_name or event.decision or ""

        type_style = {
            EventType.LLM_REQUEST: "blue",
            EventType.LLM_RESPONSE: "green",
            EventType.TOOL_CALL: "yellow",
            EventType.TOOL_RESULT: "yellow",
            EventType.AGENT_DECISION: "magenta",
            EventType.ERROR: "red",
        }.get(event.event_type, "white")

        table.add_row(
            str(i),
            f"[{type_style}]{event.event_type.value}[/{type_style}]",
            event.provider or "",
            name,
            tokens,
            duration,
        )

    console.print(table)


@main.command()
@click.argument("old_path")
@click.argument("new_path")
@click.option("--detailed", is_flag=True, default=False, help="Show per-event diffs.")
@click.option("-o", "--output", "diff_output", default=None, help="Write HTML diff report.")
def diff(old_path: str, new_path: str, detailed: bool, diff_output: str | None):
    """Compare two cassette files and show differences."""
    from trace_ops.cassette import load_cassette
    from trace_ops.diff import diff_traces

    try:
        old_trace = load_cassette(old_path)
        new_trace = load_cassette(new_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    result = diff_traces(old_trace, new_trace)

    if not result.has_changes:
        console.print("[green]✅ Traces are identical.[/green]")
        if diff_output:
            _write_diff_html(old_trace, new_trace, result, diff_output)
        return

    console.print(Panel(
        result.summary(),
        title="Trace Diff",
        border_style="yellow" if result.trajectory_changed else "cyan",
    ))

    if detailed:
        _print_detailed_diff(old_trace, new_trace, result)

    if diff_output:
        _write_diff_html(old_trace, new_trace, result, diff_output)
        console.print(f"[green]Diff report written to {diff_output}[/green]")


def _print_detailed_diff(old_trace, new_trace, result):
    """Print detailed per-event diffs to terminal."""

    old_events = old_trace.events
    new_events = new_trace.events

    max_len = max(len(old_events), len(new_events))
    table = Table(title="Per-Event Comparison")
    table.add_column("#", style="dim", width=4)
    table.add_column("Old Event", min_width=30)
    table.add_column("New Event", min_width=30)
    table.add_column("Status")

    for i in range(max_len):
        old_e = old_events[i] if i < len(old_events) else None
        new_e = new_events[i] if i < len(new_events) else None

        old_desc = ""
        new_desc = ""
        status = "[green]Match[/green]"

        if old_e:
            name = old_e.model or old_e.tool_name or old_e.decision or ""
            old_desc = f"{old_e.event_type.value} {name}"
        if new_e:
            name = new_e.model or new_e.tool_name or new_e.decision or ""
            new_desc = f"{new_e.event_type.value} {name}"

        if old_e is None:
            status = "[green]+Added[/green]"
        elif new_e is None:
            status = "[red]-Removed[/red]"
        elif old_e.event_type != new_e.event_type or (old_e.model != new_e.model) or (old_e.tool_name != new_e.tool_name):
            status = "[yellow]~Changed[/yellow]"

        table.add_row(str(i + 1), old_desc, new_desc, status)

    console.print(table)


def _write_diff_html(old_trace, new_trace, result, output_path: str):
    """Write an HTML diff report."""
    from trace_ops.reporters.html import generate_html_report

    generate_html_report(
        old_trace,
        output_path,
        compare_trace=new_trace,
        title="Diff Report",
    )


@main.command()
@click.argument("cassette_path")
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "yaml"]),
    default="json",
)
@click.option("-o", "--output", default=None, help="Output file path.")
def export(cassette_path: str, fmt: str, output: str | None):
    """Export a cassette to JSON or YAML."""
    from trace_ops.cassette import load_cassette

    try:
        trace = load_cassette(cassette_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    data = trace.to_dict()

    if fmt == "json":
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    else:
        import yaml as _yaml
        text = _yaml.dump(data, default_flow_style=False, allow_unicode=True)

    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Exported to {output}[/green]")
    else:
        print(text)


# ── New v0.2 commands ───────────────────────────────────────────────


@main.command()
@click.argument("cassette_path")
@click.option("--compare", default=None, help="Compare against a second cassette.")
@click.option(
    "--tools-only", is_flag=True, default=False,
    help="Only show tool call events.",
)
@click.option(
    "--llm-only", is_flag=True, default=False,
    help="Only show LLM events.",
)
def debug(cassette_path: str, compare: str | None, tools_only: bool, llm_only: bool):
    """Interactive time-travel debugger for a cassette."""
    from trace_ops._types import EventType
    from trace_ops.reporters.terminal import run_debugger

    event_filter: set[EventType] | None = None
    if tools_only:
        event_filter = {EventType.TOOL_CALL, EventType.TOOL_RESULT}
    elif llm_only:
        event_filter = {EventType.LLM_REQUEST, EventType.LLM_RESPONSE}

    try:
        run_debugger(cassette_path, compare_path=compare, event_filter=event_filter)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


@main.command()
@click.argument("cassette_path")
@click.option("-o", "--output", default=None, help="Output HTML file path.")
@click.option("--compare", default=None, help="Compare against a second cassette.")
def report(cassette_path: str, output: str | None, compare: str | None):
    """Generate an HTML trace report from a cassette."""
    from trace_ops.cassette import load_cassette
    from trace_ops.reporters.html import generate_html_report

    try:
        trace = load_cassette(cassette_path)
        compare_trace = load_cassette(compare) if compare else None
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    out_path = output or cassette_path.replace(".yaml", ".html").replace(".yml", ".html")
    generate_html_report(trace, out_path, compare_trace=compare_trace)
    console.print(f"[green]Report written to {out_path}[/green]")


@main.command("ls")
@click.argument("directory", default="cassettes")
def list_cassettes(directory: str):
    """List all cassettes with summary stats."""
    from trace_ops.cassette import load_cassette

    cassette_dir = Path(directory)
    if not cassette_dir.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        sys.exit(1)

    yamls = sorted(cassette_dir.rglob("*.yaml")) + sorted(cassette_dir.rglob("*.yml"))
    if not yamls:
        console.print(f"[dim]No cassettes found in {directory}[/dim]")
        return

    table = Table(title=f"Cassettes in {directory}")
    table.add_column("File", style="cyan")
    table.add_column("LLM Calls", justify="right")
    table.add_column("Tool Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Size", justify="right")

    for path in yamls:
        try:
            trace = load_cassette(path)
            size_kb = path.stat().st_size / 1024
            table.add_row(
                str(path.relative_to(cassette_dir)),
                str(trace.total_llm_calls),
                str(trace.total_tool_calls),
                f"{trace.total_tokens:,}",
                f"${trace.total_cost_usd:.4f}",
                f"{size_kb:.1f}KB",
            )
        except Exception as exc:
            table.add_row(str(path.relative_to(cassette_dir)), "[red]ERROR[/red]", "", "", "", str(exc)[:30])

    console.print(table)


@main.command()
@click.argument("directory", default="cassettes")
@click.option("--older-than", default="30d", help="Delete cassettes older than this (e.g. 30d, 1w).")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be deleted.")
def prune(directory: str, older_than: str, dry_run: bool):
    """Delete stale cassettes older than a given age."""
    import re
    import time as _time

    cassette_dir = Path(directory)
    if not cassette_dir.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        sys.exit(1)

    # Parse duration string
    match = re.match(r"(\d+)([dwh])", older_than)
    if not match:
        console.print("[red]Invalid --older-than format. Use e.g. 30d, 1w, 24h.[/red]")
        sys.exit(1)
    amount, unit = int(match.group(1)), match.group(2)
    seconds = amount * {"d": 86400, "w": 604800, "h": 3600}[unit]
    cutoff = _time.time() - seconds

    yamls = list(cassette_dir.rglob("*.yaml")) + list(cassette_dir.rglob("*.yml"))
    to_delete = [p for p in yamls if p.stat().st_mtime < cutoff]

    if not to_delete:
        console.print("[green]No stale cassettes found.[/green]")
        return

    for path in to_delete:
        if dry_run:
            console.print(f"  [dim]Would delete: {path}[/dim]")
        else:
            path.unlink()
            console.print(f"  Deleted: {path}")

    verb = "Would delete" if dry_run else "Deleted"
    console.print(f"\n[bold]{verb} {len(to_delete)} cassette(s).[/bold]")


@main.command()
@click.argument("directory", default="cassettes")
def stats(directory: str):
    """Aggregate stats across all cassettes in a directory."""
    from trace_ops.cassette import load_cassette

    cassette_dir = Path(directory)
    if not cassette_dir.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        sys.exit(1)

    yamls = list(cassette_dir.rglob("*.yaml")) + list(cassette_dir.rglob("*.yml"))
    if not yamls:
        console.print(f"[dim]No cassettes found in {directory}[/dim]")
        return

    total_files = 0
    total_llm = 0
    total_tools = 0
    total_tokens = 0
    total_cost = 0.0
    model_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}

    for path in yamls:
        try:
            trace = load_cassette(path)
            total_files += 1
            total_llm += trace.total_llm_calls
            total_tools += trace.total_tool_calls
            total_tokens += trace.total_tokens
            total_cost += trace.total_cost_usd

            from trace_ops._types import EventType
            for e in trace.events:
                if e.event_type == EventType.LLM_REQUEST and e.model:
                    model_counts[e.model] = model_counts.get(e.model, 0) + 1
                if e.event_type == EventType.TOOL_CALL and e.tool_name:
                    tool_counts[e.tool_name] = tool_counts.get(e.tool_name, 0) + 1
        except Exception:
            pass

    console.print(Panel(
        f"[bold]Cassettes:[/bold] {total_files}\n"
        f"[bold]Total LLM calls:[/bold] {total_llm}\n"
        f"[bold]Total tool calls:[/bold] {total_tools}\n"
        f"[bold]Total tokens:[/bold] {total_tokens:,}\n"
        f"[bold]Total cost:[/bold] ${total_cost:.4f}",
        title=f"Stats for {directory}",
    ))

    if model_counts:
        console.print("\n[bold]Most-used models:[/bold]")
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  {model}: {count}")

    if tool_counts:
        console.print("\n[bold]Most-called tools:[/bold]")
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  {tool}: {count}")


@main.command()
@click.argument("directory", default="cassettes")
@click.option("--top", default=10, help="Number of top cassettes to show.")
@click.option(
    "--sort",
    type=click.Choice(["cost", "tokens", "calls"]),
    default="cost",
    help="Sort order for cassette ranking.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def costs(directory: str, top: int, sort: str, as_json: bool):
    """Show a cost dashboard across all cassettes in a directory."""
    from trace_ops.reporters.cost_dashboard import CostDashboard

    cassette_dir = Path(directory)
    if not cassette_dir.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        sys.exit(1)

    dashboard = CostDashboard(directory)

    if as_json:
        print(json.dumps(dashboard.data.to_dict(), indent=2, default=str))
        return

    # Re-sort if requested
    s = dashboard.data
    if sort == "tokens":
        s.by_cassette.sort(key=lambda c: c.total_tokens, reverse=True)
    elif sort == "calls":
        s.by_cassette.sort(key=lambda c: c.llm_calls, reverse=True)
    # "cost" is default sort

    dashboard.print(top=top, console=console)


@main.command()
@click.argument("cassette_path")
def validate(cassette_path: str):
    """Validate a cassette file for integrity."""
    from trace_ops.cassette import load_cassette

    path = Path(cassette_path)
    if not path.exists():
        console.print(f"[red]File not found: {cassette_path}[/red]")
        sys.exit(1)

    issues: list[str] = []

    # Check YAML parseable
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        console.print(f"[red]Invalid YAML: {exc}[/red]")
        sys.exit(1)

    if not isinstance(raw, dict):
        console.print("[red]Cassette root must be a YAML mapping.[/red]")
        sys.exit(1)

    # Check schema version
    version = raw.get("version")
    if version not in ("1", 1):
        issues.append(f"Unknown schema version: {version}")

    # Check required keys
    if "events" not in raw:
        issues.append("Missing 'events' key")
    if "trace_id" not in raw:
        issues.append("Missing 'trace_id' key")

    # Try full deserialization
    try:
        trace = load_cassette(cassette_path)
        if not trace.events:
            issues.append("Trace contains no events")
    except Exception as exc:
        issues.append(f"Deserialization error: {exc}")

    if issues:
        console.print("[yellow]Issues found:[/yellow]")
        for issue in issues:
            console.print(f"  ⚠  {issue}")
    else:
        console.print(f"[green]✅ Cassette is valid ({len(trace.events)} events).[/green]")


# ── v0.5.0 RAG / MCP / Semantic commands ───────────────────────────


@main.command("context")
@click.argument("cassette_path")
@click.option("--visual", is_flag=True, default=False, help="Show chunk-level breakdown.")
def context(cassette_path: str, visual: bool):
    """Analyse retrieval context usage from a cassette."""
    from trace_ops.cassette import load_cassette

    try:
        trace = load_cassette(cassette_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    try:
        from trace_ops.rag.context_analysis import analyze_context_usage
    except ImportError:
        console.print("[red]Install trace_ops[rag] to use context analysis.[/red]")
        sys.exit(1)

    analysis = analyze_context_usage(trace)

    console.print(Panel(
        f"[bold]Context percent:[/bold] {analysis.context_percent:.1%}\n"
        f"[bold]Retrieved context tokens:[/bold] {analysis.retrieved_context_tokens}\n"
        f"[bold]Total input tokens:[/bold] {analysis.total_input_tokens}\n"
        f"[bold]System prompt tokens:[/bold] {analysis.system_prompt_tokens}\n"
        f"[bold]User query tokens:[/bold] {analysis.user_query_tokens}\n"
        f"[bold]Unused chunk tokens:[/bold] {analysis.unused_chunk_tokens}",
        title=f"Context Analysis: {cassette_path}",
    ))

    if visual and analysis.chunk_usages:
        table = Table(title="Chunk Usage")
        table.add_column("Chunk ID", style="cyan")
        table.add_column("Content (preview)")
        table.add_column("Query Count", justify="right")
        table.add_column("Avg Score", justify="right")
        for cu in analysis.chunk_usages[:20]:
            table.add_row(
                cu.chunk_id[:20],
                cu.text_preview[:60],
                "1" if cu.used_in_response else "0",
                f"{cu.relevance_score:.2f}",
            )
        console.print(table)


@main.group("snapshot")
def snapshot_group():
    """Manage retriever snapshots for regression testing."""
    pass


@snapshot_group.command("record")
@click.option("--queries", "-q", multiple=True, required=True, help="Queries to snapshot.")
@click.option("--retriever", required=True, help="Python import path to retriever factory, e.g. myapp.get_retriever.")
@click.option("-o", "--output", required=True, help="Output snapshot file path.")
def snapshot_record(queries: tuple[str, ...], retriever: str, output: str):
    """Record a retriever snapshot for the given queries."""
    import importlib

    try:
        from trace_ops.rag.snapshot import RetrieverSnapshot
    except ImportError:
        console.print("[red]Install trace_ops[rag] for snapshot support.[/red]")
        sys.exit(1)

    # Load retriever factory
    module_path, _, factory_name = retriever.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
        retriever_obj = getattr(mod, factory_name)()
    except Exception as exc:
        console.print(f"[red]Failed to load retriever '{retriever}': {exc}[/red]")
        sys.exit(1)

    snap = RetrieverSnapshot.record(retriever_obj, list(queries), save_to=output)
    console.print(f"[green]Snapshot saved to {output} ({len(snap.data.get('queries', {}))} queries recorded).[/green]")


@snapshot_group.command("check")
@click.argument("snapshot_path")
@click.option("--retriever", required=True, help="Python import path to retriever factory.")
@click.option("--threshold", default=0.8, show_default=True, help="Minimum Jaccard overlap score.")
def snapshot_check(snapshot_path: str, retriever: str, threshold: float):
    """Check current retriever output against a saved snapshot."""
    import importlib

    try:
        from trace_ops.rag.snapshot import RetrieverSnapshot
    except ImportError:
        console.print("[red]Install trace_ops[rag] for snapshot support.[/red]")
        sys.exit(1)

    module_path, _, factory_name = retriever.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
        retriever_obj = getattr(mod, factory_name)()
    except Exception as exc:
        console.print(f"[red]Failed to load retriever: {exc}[/red]")
        sys.exit(1)

    snap = RetrieverSnapshot.load(snapshot_path)
    results = snap.check(retriever_obj, threshold=threshold)

    all_pass = all(r.passed for r in results.queries)
    status = "[green]✅ All queries passed.[/green]" if all_pass else "[red]❌ Some queries drifted.[/red]"
    console.print(status)

    table = Table(title="Snapshot Check Results")
    table.add_column("Query")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status")
    for r in results.queries:
        table.add_row(
            r.query[:60],
            f"{r.overlap_ratio:.2f}",
            f"{threshold:.2f}",
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
        )
    console.print(table)

    if not all_pass:
        sys.exit(1)


@snapshot_group.command("update")
@click.argument("snapshot_path")
@click.option("--retriever", required=True, help="Python import path to retriever factory.")
def snapshot_update(snapshot_path: str, retriever: str):
    """Update a snapshot file with current retriever output."""
    import importlib

    try:
        from trace_ops.rag.snapshot import RetrieverSnapshot
    except ImportError:
        console.print("[red]Install trace_ops[rag] for snapshot support.[/red]")
        sys.exit(1)

    snap = RetrieverSnapshot.load(snapshot_path)
    module_path, _, factory_name = retriever.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
        retriever_obj = getattr(mod, factory_name)()
    except Exception as exc:
        console.print(f"[red]Failed to load retriever: {exc}[/red]")
        sys.exit(1)

    queries = [r.query for r in snap.check(retriever_obj).queries]
    RetrieverSnapshot.record(retriever_obj, queries, save_to=snapshot_path)
    console.print(f"[green]Snapshot updated: {snapshot_path}[/green]")


@main.command()
@click.argument("cassette_path")
@click.option(
    "--scorer",
    type=click.Choice(["ragas", "deepeval"]),
    default="ragas",
    show_default=True,
    help="Scoring framework to use.",
)
@click.option("--judge-model", default="gpt-4o-mini", show_default=True, help="LLM judge model.")
@click.option(
    "--metrics",
    default="faithfulness,answer_relevancy,context_precision",
    show_default=True,
    help="Comma-separated list of metrics to compute.",
)
@click.option("-o", "--output", default=None, help="Write JSON results to file.")
def rescore(cassette_path: str, scorer: str, judge_model: str, metrics: str, output: str | None):
    """Rescore a cassette's RAG quality using an LLM judge."""
    from trace_ops.cassette import load_cassette

    try:
        trace = load_cassette(cassette_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    try:
        if scorer == "ragas":
            from trace_ops.rag.scorers import BaseRAGScorer, RagasScorer
            scorer_obj: BaseRAGScorer = RagasScorer(judge_model=judge_model, metrics=metrics.split(","))
        else:
            from trace_ops.rag.scorers import BaseRAGScorer, DeepEvalScorer
            scorer_obj = DeepEvalScorer(judge_model=judge_model, metrics=metrics.split(","))
    except ImportError as exc:
        console.print(f"[red]Scorer import failed: {exc}[/red]")
        sys.exit(1)

    # Extract query/context/response from the first RAG retrieval + LLM pair
    from trace_ops._types import EventType
    retrieval = next((e for e in trace.events if e.event_type == EventType.RETRIEVAL), None)
    response_event = next((e for e in trace.events if e.event_type == EventType.LLM_RESPONSE), None)
    query = (retrieval.query if retrieval else "") or ""
    context_chunks = [str(c.get("text", "") if isinstance(c, dict) else c) for c in (retrieval.chunks or [])] if retrieval else []
    response_raw = response_event.response if response_event else None
    response: str = response_raw if isinstance(response_raw, str) else (str(response_raw) if response_raw else "")

    result = scorer_obj.score(query, context_chunks, response)
    console.print(Panel(
        "\n".join(f"[bold]{k}:[/bold] {v:.3f}" for k, v in result.scores.items()),
        title=f"RAG Scores ({scorer}) — {cassette_path}",
    ))

    if output:
        import json as _json
        scores_dict = {"scores": result.scores, "scorer": result.scorer, "judge_model": result.judge_model}
        Path(output).write_text(_json.dumps(scores_dict, indent=2), encoding="utf-8")
        console.print(f"[green]Scores written to {output}[/green]")


@main.command("export-rag")
@click.argument("cassette_dir")
@click.option(
    "--format", "fmt",
    type=click.Choice(["ragas", "deepeval", "csv"]),
    default="ragas",
    show_default=True,
    help="Export format.",
)
@click.option("-o", "--output", required=True, help="Output file path.")
def export_rag(cassette_dir: str, fmt: str, output: str):
    """Export RAG retrieval data from cassettes to a dataset file."""
    try:
        from trace_ops.rag.export import to_csv, to_deepeval_dataset, to_ragas_dataset
    except ImportError:
        console.print("[red]Install trace_ops[rag] for RAG export.[/red]")
        sys.exit(1)

    cassette_path = Path(cassette_dir)
    if not cassette_path.exists():
        console.print(f"[red]Path not found: {cassette_dir}[/red]")
        sys.exit(1)

    try:
        if fmt == "ragas":
            result = to_ragas_dataset(cassette_dir)
            import json as _json
            Path(output).write_text(_json.dumps(result, indent=2, default=str), encoding="utf-8")
        elif fmt == "deepeval":
            result = to_deepeval_dataset(cassette_dir)
            import json as _json
            Path(output).write_text(_json.dumps(result, indent=2, default=str), encoding="utf-8")
        else:
            to_csv(cassette_dir, output)
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[green]RAG dataset exported to {output}.[/green]")


@main.command("export-finetune")
@click.argument("cassette_dir")
@click.option(
    "--format", "fmt",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    show_default=True,
    help="Fine-tune format.",
)
@click.option("-o", "--output", required=True, help="Output JSONL file path.")
@click.option("--system", default=None, help="Override system prompt for all examples.")
def export_finetune(cassette_dir: str, fmt: str, output: str, system: str | None):
    """Export cassettes as fine-tuning data (JSONL)."""
    try:
        from trace_ops.export.finetune import to_anthropic_finetune, to_openai_finetune
    except ImportError:
        console.print("[red]Install trace_ops for fine-tune export.[/red]")
        sys.exit(1)

    if not Path(cassette_dir).exists():
        console.print(f"[red]Path not found: {cassette_dir}[/red]")
        sys.exit(1)

    kwargs: dict = {"output": output}
    if system is not None:
        kwargs["system_prompt"] = system

    try:
        if fmt == "openai":
            to_openai_finetune(cassette_dir, **kwargs)
        else:
            to_anthropic_finetune(cassette_dir, **kwargs)
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[green]Fine-tune data exported to {output}.[/green]")


# ── v0.6.0 Behavioral Analysis commands (inspired by agent-pr-replay) ──────


@main.command("analyze")
@click.argument("cassette_dir")
@click.option("--window", default=3, show_default=True, help="Tool-sequence window size for n-gram detection.")
@click.option("--top", default=10, show_default=True, help="Top N sequences to show.")
@click.option("-o", "--output", default=None, help="Write PatternReport as JSON to this file.")
@click.option("--skills", "gen_skills", default=None, help="Also write a AGENTS.md guidance file to this path.")
def analyze(cassette_dir: str, window: int, top: int, output: str | None, gen_skills: str | None):
    """Analyze behavioral patterns across a directory of cassettes.

    Finds recurring tool sequences, model usage, error patterns, and
    cost/token averages — inspired by agent-pr-replay's stats module.

    Example:

        traceops analyze cassettes/ --skills AGENTS.md
    """
    from trace_ops.analysis import PatternDetector, SkillsGenerator

    cassette_path = Path(cassette_dir)
    if not cassette_path.exists():
        console.print(f"[red]Directory not found: {cassette_dir}[/red]")
        sys.exit(1)

    with console.status("[bold green]Loading cassettes..."):
        detector = PatternDetector(window_size=window, top_n=top)
        report = detector.analyze_dir(cassette_path)

    if report.cassette_count == 0:
        console.print(f"[yellow]No cassettes found in {cassette_dir}[/yellow]")
        return

    console.print(Panel(
        f"[bold]Cassettes:[/bold] {report.cassette_count}\n"
        f"[bold]Total events:[/bold] {report.total_events}\n"
        f"[bold]Avg LLM calls:[/bold] {report.avg_llm_calls:.1f}\n"
        f"[bold]Avg tokens:[/bold] {report.avg_tokens:.0f}\n"
        f"[bold]Avg cost:[/bold] ${report.avg_cost_usd:.5f}",
        title=f"Pattern Analysis: {cassette_dir}",
    ))

    if report.top_tool_sequences:
        console.print("\n[bold]Top Tool Sequences:[/bold]")
        for i, seq in enumerate(report.top_tool_sequences[:5], 1):
            arrow = " → ".join(seq.sequence)
            console.print(f"  {i}. {arrow}  ({seq.count}×)")

    if report.model_usage:
        console.print("\n[bold]Model Usage:[/bold]")
        for m in report.model_usage[:5]:
            console.print(
                f"  {m.model}: {m.call_count} calls, "
                f"{m.total_tokens:,} tokens, ${m.total_cost_usd:.5f}"
            )

    if report.tool_frequency:
        console.print("\n[bold]Tool Heatmap:[/bold]")
        max_count = max(report.tool_frequency.values())
        for tool, count in list(report.tool_frequency.items())[:10]:
            bar_len = int(count / max_count * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            console.print(f"  {tool:20s} {bar} {count}")

    if output:
        Path(output).write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        console.print(f"[green]Report saved to {output}[/green]")

    if gen_skills:
        gen = SkillsGenerator()
        gen.from_pattern_report(report, output_path=Path(gen_skills))
        console.print(f"[green]Guidance written to {gen_skills}[/green]")


@main.command("gap-report")
@click.argument("golden_dir")
@click.argument("agent_dir")
@click.option("-o", "--output", default=None, help="Write GapReport as JSON to this file.")
@click.option("--skills", "gen_skills", default=None, help="Write AGENTS.md guidance to this path.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Print report as JSON.")
def gap_report(golden_dir: str, agent_dir: str, output: str | None, gen_skills: str | None, as_json: bool):
    """Compare agent traces to golden baselines and show behavioral gaps.

    GOLDEN_DIR: directory of golden / human-validated cassettes.
    AGENT_DIR:  directory of cassettes recorded by the agent under test.

    Inspired by agent-pr-replay's approach of comparing AI agent output
    to merged human PRs to find systematic behavioral divergences.

    Example:

        traceops gap-report cassettes/golden/ cassettes/agent/ --skills AGENTS.md
    """
    from trace_ops.analysis import GapAnalyzer, SkillsGenerator
    from trace_ops.cassette import load_cassette

    def _load_dir(d: str) -> list[tuple[str, object]]:
        p = Path(d)
        if not p.exists():
            console.print(f"[red]Directory not found: {d}[/red]")
            sys.exit(1)
        pairs: list[tuple[str, object]] = []
        for path in sorted(p.rglob("*.yaml")):
            with contextlib.suppress(Exception):
                pairs.append((path.name, load_cassette(path)))
        return pairs

    golden = _load_dir(golden_dir)
    agent = _load_dir(agent_dir)

    if not golden:
        console.print(f"[yellow]No golden cassettes found in {golden_dir}[/yellow]")
        return
    if not agent:
        console.print(f"[yellow]No agent cassettes found in {agent_dir}[/yellow]")
        return

    analyzer = GapAnalyzer()
    report = analyzer.compare(golden, agent)  # type: ignore[arg-type]

    if as_json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return

    if not report.gaps:
        console.print(f"[green]✅ {report.summary()}[/green]")
    else:
        severity_colors = {"critical": "red", "warning": "yellow", "info": "blue"}
        console.print(Panel(
            report.summary(),
            title="Behavioral Gap Report",
            border_style="red" if report.critical_count else "yellow",
        ))
        table = Table(title="Gaps Found")
        table.add_column("Severity", width=10)
        table.add_column("Category", width=20)
        table.add_column("Description")
        table.add_column("Frequency", justify="right", width=10)
        for gap in report.gaps:
            color = severity_colors.get(gap.severity, "white")
            table.add_row(
                f"[{color}]{gap.severity.upper()}[/{color}]",
                gap.category,
                gap.description,
                f"{gap.frequency*100:.0f}%",
            )
        console.print(table)

    if output:
        Path(output).write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        console.print(f"[green]Gap report saved to {output}[/green]")

    if gen_skills:
        gen = SkillsGenerator()
        gen.from_gap_report(report, output_path=Path(gen_skills))
        console.print(f"[green]Guidance written to {gen_skills}[/green]")

    if report.critical_count:
        sys.exit(1)


@main.command("pr-diff")
@click.argument("pr_url")
@click.option("--token", default=None, envvar="GITHUB_TOKEN", help="GitHub API token (or set GITHUB_TOKEN).")
@click.option("--task", is_flag=True, default=False, help="Print the reverse-engineered task prompt only.")
@click.option("--files", is_flag=True, default=False, help="List changed files with +/- stats.")
def pr_diff(pr_url: str, token: str | None, task: bool, files: bool):
    """Fetch a GitHub PR diff to use as a golden baseline.

    Inspired by agent-pr-replay's PR fetcher — pulls any public PR and
    reverse-engineers a task prompt from the diff so you can compare
    your agent's behaviour against the human solution.

    Example:

        traceops pr-diff https://github.com/owner/repo/pull/42 --task
    """
    from trace_ops.github import PRFetcher

    with console.status("[bold green]Fetching PR from GitHub..."):
        try:
            fetcher = PRFetcher(token=token)
            diff = fetcher.fetch(pr_url)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)

    if task:
        print(diff.extract_task_prompt())
        return

    console.print(Panel(
        f"[bold]PR:[/bold] #{diff.pr_number} — {diff.title}\n"
        f"[bold]Author:[/bold] {diff.author}\n"
        f"[bold]Merged:[/bold] {diff.merged_at or 'N/A'}\n"
        f"[bold]Files changed:[/bold] {len(diff.files)}\n"
        f"[bold]Lines:[/bold] +{diff.total_additions} -{diff.total_deletions}",
        title=f"GitHub PR: {pr_url}",
        border_style="cyan",
    ))

    if files:
        table = Table(title="Changed Files")
        table.add_column("File", style="cyan")
        table.add_column("Status", width=10)
        table.add_column("+", justify="right", style="green", width=6)
        table.add_column("-", justify="right", style="red", width=6)
        for f in diff.files:
            table.add_row(f.filename, f.status, str(f.additions), str(f.deletions))
        console.print(table)
    else:
        console.print("\n[bold]Task prompt (reverse-engineered):[/bold]")
        console.print(diff.extract_task_prompt())


if __name__ == "__main__":
    main()
