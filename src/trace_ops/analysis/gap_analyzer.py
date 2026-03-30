"""Behavioral gap analysis — compare agent traces to golden baselines.

Finds systematic divergences in token usage, tool selection, model choice,
and error rates. Directly inspired by agent-pr-replay's diff_comparison module
which compares Claude Code's output against human-authored PR diffs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .._types import EventType, Trace


@dataclass
class BehavioralGap:
    """A systematic difference between agent and golden/baseline behavior."""

    category: str  # "token_inflation", "cost_inflation", "missing_tool", etc.
    description: str
    severity: str  # "critical", "warning", "info"
    frequency: float  # fraction of agent traces affected (0..1)
    golden_value: Any = None
    agent_value: Any = None
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "frequency": round(self.frequency, 3),
            "golden_value": self.golden_value,
            "agent_value": self.agent_value,
            "examples": self.examples[:3],
        }


@dataclass
class GapReport:
    """Results of comparing agent traces to golden baselines."""

    golden_count: int
    agent_count: int
    gaps: list[BehavioralGap]

    @property
    def critical_count(self) -> int:
        return sum(1 for g in self.gaps if g.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for g in self.gaps if g.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "golden_count": self.golden_count,
            "agent_count": self.agent_count,
            "gap_count": len(self.gaps),
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "gaps": [g.to_dict() for g in self.gaps],
        }

    def summary(self) -> str:
        if not self.gaps:
            return (
                f"No significant behavioral gaps found "
                f"({self.golden_count} golden vs {self.agent_count} agent traces)."
            )
        return (
            f"Found {len(self.gaps)} behavioral gap(s): "
            f"{self.critical_count} critical, {self.warning_count} warnings "
            f"({self.golden_count} golden vs {self.agent_count} agent traces)."
        )


class GapAnalyzer:
    """Compare a set of agent traces to golden baselines and identify behavioral gaps.

    Inspired by agent-pr-replay's approach of comparing AI coding agent output
    against merged human PRs to find systematic divergences. Adapted here for
    general LLM agent traces recorded with TraceOps.

    Example::

        analyzer = GapAnalyzer()
        report = analyzer.compare(golden_traces, agent_traces)
        for gap in report.gaps:
            print(f"[{gap.severity.upper()}] {gap.description}")
    """

    #: Flag if agent uses this much more tokens/cost than golden
    INFLATION_THRESHOLD = 1.5
    #: Flag if agent error rate exceeds golden by this many percentage points
    ERROR_RATE_DELTA = 0.10
    #: Flag tool presence difference beyond this fraction
    TOOL_FREQ_THRESHOLD = 0.30

    def compare(
        self,
        golden: list[tuple[str, Trace]],
        agent: list[tuple[str, Trace]],
    ) -> GapReport:
        """Compare agent traces to golden traces and return a :class:`GapReport`."""
        if not golden or not agent:
            return GapReport(golden_count=len(golden), agent_count=len(agent), gaps=[])

        gaps: list[BehavioralGap] = []
        g_n, a_n = len(golden), len(agent)

        # ── Token inflation ──────────────────────────────────────────────
        g_tokens = [t.total_tokens for _, t in golden if t.total_tokens > 0]
        a_tokens = [t.total_tokens for _, t in agent if t.total_tokens > 0]
        if g_tokens and a_tokens:
            g_avg = sum(g_tokens) / len(g_tokens)
            a_avg = sum(a_tokens) / len(a_tokens)
            if g_avg > 0 and a_avg / g_avg >= self.INFLATION_THRESHOLD:
                ratio = a_avg / g_avg
                gaps.append(
                    BehavioralGap(
                        category="token_inflation",
                        description=(
                            f"Agent uses {ratio:.1f}× more tokens than golden baseline "
                            f"(avg {a_avg:.0f} vs {g_avg:.0f})"
                        ),
                        severity="warning" if ratio < 3 else "critical",
                        frequency=1.0,
                        golden_value=round(g_avg, 1),
                        agent_value=round(a_avg, 1),
                    )
                )

        # ── Cost inflation ───────────────────────────────────────────────
        g_costs = [t.total_cost_usd for _, t in golden if t.total_cost_usd > 0]
        a_costs = [t.total_cost_usd for _, t in agent if t.total_cost_usd > 0]
        if g_costs and a_costs:
            g_avg_c = sum(g_costs) / len(g_costs)
            a_avg_c = sum(a_costs) / len(a_costs)
            if g_avg_c > 0 and a_avg_c / g_avg_c >= self.INFLATION_THRESHOLD:
                ratio_c = a_avg_c / g_avg_c
                gaps.append(
                    BehavioralGap(
                        category="cost_inflation",
                        description=(
                            f"Agent costs {ratio_c:.1f}× more than golden baseline "
                            f"(avg ${a_avg_c:.5f} vs ${g_avg_c:.5f})"
                        ),
                        severity="warning" if ratio_c < 3 else "critical",
                        frequency=1.0,
                        golden_value=round(g_avg_c, 6),
                        agent_value=round(a_avg_c, 6),
                    )
                )

        # ── Tool presence differences ────────────────────────────────────
        def tool_presence(traces: list[tuple[str, Trace]]) -> dict[str, float]:
            counter: Counter[str] = Counter()
            for _, trace in traces:
                used = {
                    e.tool_name
                    for e in trace.events
                    if e.event_type == EventType.TOOL_CALL and e.tool_name
                }
                counter.update(used)
            total = len(traces)
            return {tool: cnt / total for tool, cnt in counter.items()}

        g_tf = tool_presence(golden)
        a_tf = tool_presence(agent)
        for tool in set(g_tf) | set(a_tf):
            g_freq = g_tf.get(tool, 0.0)
            a_freq = a_tf.get(tool, 0.0)
            delta = a_freq - g_freq
            if delta <= -self.TOOL_FREQ_THRESHOLD:
                pct = sum(
                    1 for _, t in agent
                    if not any(e.tool_name == tool for e in t.events)
                ) / a_n
                gaps.append(
                    BehavioralGap(
                        category="missing_tool",
                        description=(
                            f"Agent under-uses '{tool}': golden uses it in "
                            f"{g_freq*100:.0f}% of traces, agent in only {a_freq*100:.0f}%"
                        ),
                        severity="warning",
                        frequency=pct,
                        golden_value=round(g_freq, 3),
                        agent_value=round(a_freq, 3),
                    )
                )
            elif delta >= self.TOOL_FREQ_THRESHOLD:
                pct = sum(
                    1 for _, t in agent
                    if any(e.tool_name == tool for e in t.events)
                ) / a_n
                gaps.append(
                    BehavioralGap(
                        category="extra_tool",
                        description=(
                            f"Agent over-uses '{tool}': golden uses it in "
                            f"{g_freq*100:.0f}% of traces, agent in {a_freq*100:.0f}%"
                        ),
                        severity="info",
                        frequency=pct,
                        golden_value=round(g_freq, 3),
                        agent_value=round(a_freq, 3),
                    )
                )

        # ── Model mismatch ───────────────────────────────────────────────
        def top_model(traces: list[tuple[str, Trace]]) -> str | None:
            counter: Counter[str] = Counter()
            for _, trace in traces:
                for e in trace.events:
                    if e.model:
                        counter[e.model] += 1
            return counter.most_common(1)[0][0] if counter else None

        g_model, a_model = top_model(golden), top_model(agent)
        if g_model and a_model and g_model != a_model:
            gaps.append(
                BehavioralGap(
                    category="model_mismatch",
                    description=(
                        f"Agent primarily uses '{a_model}' but golden traces use '{g_model}'"
                    ),
                    severity="info",
                    frequency=1.0,
                    golden_value=g_model,
                    agent_value=a_model,
                )
            )

        # ── Error rate ───────────────────────────────────────────────────
        g_err = sum(
            1 for _, t in golden
            if any(e.event_type == EventType.ERROR for e in t.events)
        ) / g_n
        a_err = sum(
            1 for _, t in agent
            if any(e.event_type == EventType.ERROR for e in t.events)
        ) / a_n
        if a_err - g_err >= self.ERROR_RATE_DELTA:
            gaps.append(
                BehavioralGap(
                    category="error_rate",
                    description=(
                        f"Agent error rate {a_err*100:.0f}% vs golden {g_err*100:.0f}% "
                        f"(+{(a_err - g_err)*100:.0f}pp)"
                    ),
                    severity="critical",
                    frequency=a_err,
                    golden_value=round(g_err, 3),
                    agent_value=round(a_err, 3),
                )
            )

        # ── LLM call inflation ───────────────────────────────────────────
        g_avg_calls = sum(t.total_llm_calls for _, t in golden) / g_n
        a_avg_calls = sum(t.total_llm_calls for _, t in agent) / a_n
        if g_avg_calls > 0 and a_avg_calls / g_avg_calls >= self.INFLATION_THRESHOLD:
            ratio_c = a_avg_calls / g_avg_calls
            gaps.append(
                BehavioralGap(
                    category="llm_call_inflation",
                    description=(
                        f"Agent makes {ratio_c:.1f}× more LLM calls than golden "
                        f"(avg {a_avg_calls:.1f} vs {g_avg_calls:.1f})"
                    ),
                    severity="warning",
                    frequency=1.0,
                    golden_value=round(g_avg_calls, 2),
                    agent_value=round(a_avg_calls, 2),
                )
            )

        severity_order = {"critical": 0, "warning": 1, "info": 2}
        gaps.sort(key=lambda g: (severity_order.get(g.severity, 3), -g.frequency))
        return GapReport(golden_count=g_n, agent_count=a_n, gaps=gaps)
