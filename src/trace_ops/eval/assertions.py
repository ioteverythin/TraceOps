"""Assertion helpers for LLM-judge evaluation scores.

These helpers integrate with pytest: a failing assertion raises
:class:`EvalAssertionError` (a subclass of :class:`AssertionError`), so it
shows up as a test failure with a descriptive diff message.

Usage::

    from trace_ops.eval.assertions import assert_eval_score, assert_passes_criteria
    from trace_ops.eval import LLMJudge

    judge = LLMJudge(model="gpt-4o-mini")

    # All default criteria must score >= 0.6
    assert_eval_score(trace, judge=judge)

    # Specific criteria must score >= 0.8
    assert_eval_score(trace, criteria=["correctness", "safety"], min_score=0.8)

    # Convenience: positional criteria list
    assert_passes_criteria(trace, ["correctness", "helpfulness"])
"""

from __future__ import annotations

from trace_ops._types import Trace
from trace_ops.eval.judge import LLMJudge, TraceEvaluation


class EvalAssertionError(AssertionError):
    """Raised when one or more evaluation criteria fall below the threshold.

    Subclasses :class:`AssertionError` so pytest reports it as a test failure.
    """


def assert_eval_score(
    trace: Trace,
    *,
    criteria: list[str] | None = None,
    min_score: float = 0.6,
    judge: LLMJudge | None = None,
) -> TraceEvaluation:
    """Assert that *trace* scores at or above *min_score* on all given criteria.

    Args:
        trace: The :class:`~trace_ops._types.Trace` to evaluate.
        criteria: Criterion names to check.  Defaults to the judge's own
            default criteria (``correctness``, ``helpfulness``,
            ``tool_efficiency``).
        min_score: Minimum normalised score in ``[0.0, 1.0]``.  Default
            ``0.6`` (equivalent to a 3.4/5 raw score).
        judge: A pre-configured :class:`~trace_ops.eval.judge.LLMJudge`.
            A default judge is created when ``None`` is passed.

    Returns:
        The :class:`~trace_ops.eval.judge.TraceEvaluation` so callers can
        inspect individual scores after the assertion.

    Raises:
        EvalAssertionError: If any criterion scores below *min_score*.

    Example::

        result = assert_eval_score(trace, criteria=["safety"], min_score=0.9)
        print(result.summary())
    """
    effective_criteria = criteria or LLMJudge.DEFAULT_CRITERIA

    if judge is None:
        judge = LLMJudge(criteria=effective_criteria)
    else:
        # Rebuild with the requested criteria, preserving provider/model/_llm_caller
        judge = LLMJudge(
            model=judge.model,
            provider=judge.provider,
            criteria=effective_criteria,
            _llm_caller=judge._llm_caller,
        )

    result = judge.evaluate(trace)

    failures = [s for s in result.scores if s.score < min_score]
    if failures:
        lines = [
            f"Eval assertion failed — "
            f"{len(failures)} criterion/criteria scored below {min_score:.0%}:"
        ]
        for s in failures:
            lines.append(
                f"  {s.criterion:<22} score={s.score:.2f} < {min_score:.2f}"
                f"   reason: {s.reasoning}"
            )
        raise EvalAssertionError("\n".join(lines))

    return result


def assert_passes_criteria(
    trace: Trace,
    criteria: list[str],
    *,
    min_score: float = 0.6,
    judge: LLMJudge | None = None,
) -> TraceEvaluation:
    """Assert *trace* passes all listed *criteria*.

    Convenience wrapper around :func:`assert_eval_score` with *criteria* as a
    positional argument.

    Args:
        trace: The trace to evaluate.
        criteria: List of criterion names that must all pass.
        min_score: Minimum normalised score per criterion.  Default ``0.6``.
        judge: Optional pre-configured judge.

    Returns:
        The :class:`~trace_ops.eval.judge.TraceEvaluation`.

    Raises:
        EvalAssertionError: If any criterion scores below *min_score*.
    """
    return assert_eval_score(
        trace, criteria=criteria, min_score=min_score, judge=judge
    )
