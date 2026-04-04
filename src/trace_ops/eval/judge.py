"""LLM-as-judge evaluation for agent traces.

Evaluates a recorded :class:`~trace_ops._types.Trace` against named quality
criteria (e.g. *correctness*, *helpfulness*, *tool_efficiency*) by submitting
the trace transcript to an LLM judge and parsing a structured JSON response.

Scores are normalised to **0–1** regardless of the raw 1–5 scale.

Usage::

    from trace_ops.eval import LLMJudge

    judge = LLMJudge(model="gpt-4o-mini")
    result = judge.evaluate(trace)
    print(result.summary())

    # Specific criteria
    result = judge.evaluate(trace, extra_criteria=["goal_completion", "safety"])
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from trace_ops._types import EventType, Trace
from trace_ops.eval.rubrics import BUILTIN_CRITERIA, EvalCriteria

# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert AI evaluator. You will be given a transcript of an AI agent \
run — including user messages, assistant responses, and tool calls — and must \
evaluate it against the specified criteria.

For each criterion provide:
- A score from 1 to 5 (1 = very poor, 3 = acceptable, 5 = excellent)
- One concise sentence of reasoning

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "evaluations": [
    {"criterion": "<name>", "score": <integer 1-5>, "reasoning": "<sentence>"},
    ...
  ]
}

Do not add any text, markdown, or code blocks outside the JSON object.\
"""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion.

    Attributes:
        criterion: Criterion name (e.g. ``"correctness"``).
        score: Normalised score in ``[0.0, 1.0]``.
        raw_score: Raw integer score on the judge's 1–5 scale.
        reasoning: One-sentence rationale from the judge.
    """

    criterion: str
    score: float
    raw_score: int
    reasoning: str


@dataclass
class TraceEvaluation:
    """Evaluation of an agent trace across one or more criteria.

    Attributes:
        scores: Per-criterion scores.
        overall_score: Mean of all normalised scores.
        judge_model: Model name used for judging.
        judge_tokens: Total tokens consumed by the judge call.
        judge_cost_usd: Estimated cost in USD.
        judge_duration_ms: Wall-clock time for the judge call.
    """

    scores: list[CriterionScore] = field(default_factory=list)
    overall_score: float = 0.0
    judge_model: str = ""
    judge_tokens: int = 0
    judge_cost_usd: float = 0.0
    judge_duration_ms: float = 0.0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def score_for(self, criterion: str) -> CriterionScore | None:
        """Return the :class:`CriterionScore` for *criterion*, or ``None``."""
        return next((s for s in self.scores if s.criterion == criterion), None)

    def passes(self, criterion: str, min_score: float = 0.6) -> bool:
        """Return ``True`` if *criterion* is at or above *min_score*."""
        s = self.score_for(criterion)
        return s is not None and s.score >= min_score

    def summary(self) -> str:
        """Return a human-readable summary table of all scores."""
        lines = [f"Trace Evaluation  overall={self.overall_score:.2f}  "
                 f"model={self.judge_model}  tokens={self.judge_tokens}"]
        for s in self.scores:
            filled = round(s.score * 10)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(
                f"  {s.criterion:<22} {bar} {s.score:.2f}  {s.reasoning}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-serialisable)."""
        return {
            "overall_score": self.overall_score,
            "judge_model": self.judge_model,
            "judge_tokens": self.judge_tokens,
            "judge_cost_usd": self.judge_cost_usd,
            "judge_duration_ms": self.judge_duration_ms,
            "scores": [
                {
                    "criterion": s.criterion,
                    "score": s.score,
                    "raw_score": s.raw_score,
                    "reasoning": s.reasoning,
                }
                for s in self.scores
            ],
        }


# ---------------------------------------------------------------------------
# Trace → transcript
# ---------------------------------------------------------------------------


def _extract_text(content: Any, max_chars: int = 600) -> str:
    """Best-effort extraction of text from an OpenAI/Anthropic content value."""
    if isinstance(content, str):
        return content[:max_chars]
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or str(block.get("content", "")))
            else:
                parts.append(str(block))
        return " ".join(parts)[:max_chars]
    return str(content)[:max_chars]


def build_trace_summary(trace: Trace) -> str:  # noqa: N802 — public for tests
    """Convert a :class:`~trace_ops._types.Trace` to a human-readable transcript.

    The transcript is what the LLM judge reads when evaluating the trace.
    """
    lines: list[str] = []

    for event in trace.events:
        etype = event.event_type

        if etype == EventType.LLM_REQUEST and event.messages:
            for msg in event.messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user")).upper()
                content = _extract_text(msg.get("content", ""))
                if content.strip():
                    lines.append(f"[{role}] {content}")

        elif etype == EventType.LLM_RESPONSE and event.response is not None:
            resp = event.response
            text = ""
            if isinstance(resp, str):
                text = resp[:800]
            elif isinstance(resp, dict):
                # OpenAI format: choices[0].message.content
                choices = resp.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    text = _extract_text(msg.get("content", ""))
                # Anthropic format: content[0].text
                if not text:
                    ant_content = resp.get("content") or []
                    if ant_content and isinstance(ant_content[0], dict):
                        text = ant_content[0].get("text", "")[:800]
            if text.strip():
                lines.append(f"[ASSISTANT] {text}")

        elif etype == EventType.TOOL_CALL:
            inp = ""
            if event.tool_input:
                try:
                    inp = json.dumps(event.tool_input)[:200]
                except (TypeError, ValueError):
                    inp = str(event.tool_input)[:200]
            lines.append(f"[TOOL CALL] {event.tool_name or '?'}({inp})")

        elif etype == EventType.TOOL_RESULT:
            out = str(event.tool_output or "")[:300]
            lines.append(f"[TOOL RESULT] {event.tool_name or '?'}: {out}")

        elif etype == EventType.ERROR:
            lines.append(
                f"[ERROR] {event.error_type or 'error'}: {event.error_message or ''}"
            )

    return "\n".join(lines) if lines else "(empty trace)"


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

#: Callable type for the injectable LLM caller used in tests.
LLMCallerType = Callable[[str, str, str], tuple[str, int]]


class LLMJudge:
    """Evaluate agent traces using an LLM judge.

    Submits the trace transcript plus criterion descriptions to a chat LLM and
    parses the JSON response into :class:`CriterionScore` objects.

    Args:
        model: Judge model name.  Default ``"gpt-4o-mini"``.
        provider: ``"openai"`` or ``"anthropic"``.  Default ``"openai"``.
        criteria: Criteria to score — each item may be a built-in criterion
            name (see :data:`~trace_ops.eval.rubrics.BUILTIN_CRITERIA`) or a
            custom :class:`~trace_ops.eval.rubrics.EvalCriteria` object.
            Defaults to ``["correctness", "helpfulness", "tool_efficiency"]``.

    Example::

        from trace_ops.eval import LLMJudge

        judge = LLMJudge(model="gpt-4o-mini", criteria=["correctness", "safety"])
        result = judge.evaluate(trace)
        print(result.summary())

        # Override criteria per-call
        result = judge.evaluate(trace, extra_criteria=["tone"])
    """

    DEFAULT_CRITERIA: list[str] = ["correctness", "helpfulness", "tool_efficiency"]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        criteria: Sequence[str | EvalCriteria] | None = None,
        *,
        _llm_caller: LLMCallerType | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.criteria: list[str | EvalCriteria] = list(criteria) if criteria else list(self.DEFAULT_CRITERIA)
        self._llm_caller = _llm_caller  # injectable for unit tests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trace: Trace,
        *,
        extra_criteria: Sequence[str | EvalCriteria] | None = None,
    ) -> TraceEvaluation:
        """Evaluate *trace* and return scored results.

        Args:
            trace: The recorded :class:`~trace_ops._types.Trace` to evaluate.
            extra_criteria: Additional criteria appended to the judge's default
                list for this call only.

        Returns:
            A :class:`TraceEvaluation` with per-criterion scores, overall score,
            and judge metadata.

        Raises:
            ImportError: If the required LLM SDK is not installed.
            RuntimeError: If the judge response cannot be parsed as JSON.
        """
        all_criteria: list[str | EvalCriteria] = list(self.criteria)
        if extra_criteria:
            all_criteria.extend(extra_criteria)

        criteria_objs = self._resolve_criteria(all_criteria)
        user_prompt = self._build_prompt(trace, criteria_objs)

        t0 = time.perf_counter()
        raw_response, tokens = self._call_llm(_JUDGE_SYSTEM, user_prompt)
        duration_ms = (time.perf_counter() - t0) * 1000

        scores = self._parse_response(raw_response, criteria_objs)
        overall = sum(s.score for s in scores) / len(scores) if scores else 0.0

        return TraceEvaluation(
            scores=scores,
            overall_score=round(overall, 4),
            judge_model=self.model,
            judge_tokens=tokens,
            judge_cost_usd=self._estimate_cost(tokens),
            judge_duration_ms=round(duration_ms, 1),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_criteria(
        self, raw: Sequence[str | EvalCriteria]
    ) -> list[EvalCriteria]:
        result: list[EvalCriteria] = []
        for item in raw:
            if isinstance(item, EvalCriteria):
                result.append(item)
            elif item in BUILTIN_CRITERIA:
                result.append(BUILTIN_CRITERIA[item])
            else:
                # Unknown name → generic criterion
                result.append(
                    EvalCriteria(
                        name=item,
                        description=f"Evaluate the agent on: {item}",
                    )
                )
        return result

    def _build_prompt(self, trace: Trace, criteria: list[EvalCriteria]) -> str:
        transcript = build_trace_summary(trace)
        criteria_block = "\n".join(f"- {c.prompt_text()}" for c in criteria)
        return (
            f"## Agent Transcript\n\n{transcript}\n\n"
            f"## Evaluation Criteria\n\n{criteria_block}\n\n"
            "Evaluate the transcript against every criterion listed above and "
            "return only the JSON object."
        )

    def _call_llm(self, system: str, user: str) -> tuple[str, int]:
        """Dispatch to the configured provider (or injected mock)."""
        if self._llm_caller is not None:
            return self._llm_caller(system, user, self.model)
        if self.provider == "openai":
            return self._call_openai(system, user)
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        raise ValueError(
            f"Unknown provider {self.provider!r}. Use 'openai' or 'anthropic'."
        )

    def _call_openai(self, system: str, user: str) -> tuple[str, int]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required: pip install openai  "
                "or  pip install 'traceops[openai]'"
            ) from exc

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens

    def _call_anthropic(self, system: str, user: str) -> tuple[str, int]:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required: pip install anthropic  "
                "or  pip install 'traceops[anthropic]'"
            ) from exc

        client = Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""  # type: ignore[union-attr]
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        return text, tokens

    def _parse_response(
        self,
        raw: str,
        criteria: list[EvalCriteria],
    ) -> list[CriterionScore]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM judge returned invalid JSON: {raw[:300]!r}"
            ) from exc

        evals: list[dict[str, Any]] = data.get("evaluations") or []
        criteria_by_name = {c.name: c for c in criteria}
        scores: list[CriterionScore] = []

        for item in evals:
            if not isinstance(item, dict):
                continue
            name = str(item.get("criterion", ""))
            raw_score = int(item.get("score", 3))
            reasoning = str(item.get("reasoning", ""))

            criterion = criteria_by_name.get(name)
            scale_min = criterion.scale_min if criterion else 1
            scale_max = criterion.scale_max if criterion else 5
            span = max(scale_max - scale_min, 1)
            norm = max(0.0, min(1.0, (raw_score - scale_min) / span))

            scores.append(
                CriterionScore(
                    criterion=name,
                    score=round(norm, 4),
                    raw_score=raw_score,
                    reasoning=reasoning,
                )
            )

        # Fill in any criteria the judge silently skipped
        found = {s.criterion for s in scores}
        for c in criteria:
            if c.name not in found:
                scores.append(
                    CriterionScore(
                        criterion=c.name,
                        score=0.5,
                        raw_score=3,
                        reasoning="(score not returned by judge — defaulted to 0.5)",
                    )
                )

        return scores

    @staticmethod
    def _estimate_cost(tokens: int) -> float:
        """Rough cost estimate using gpt-4o-mini blended rate (~$0.30/1M tokens)."""
        return round(tokens * 0.30 / 1_000_000, 8)
