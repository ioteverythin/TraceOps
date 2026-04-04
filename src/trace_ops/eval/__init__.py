"""LLM-as-judge evaluation add-on for traceops.

Evaluate any agent trace against named quality criteria using an LLM judge.
Unlike the RAG-specific scorers, this works on *any* trace — retrieval-based
or not.

Install::

    pip install 'traceops[openai]'   # or anthropic

Quick start::

    from trace_ops.eval import LLMJudge, assert_eval_score

    judge = LLMJudge(model="gpt-4o-mini")
    result = judge.evaluate(trace)
    print(result.summary())

    # In a pytest test
    assert_eval_score(trace, criteria=["correctness", "safety"], min_score=0.8)

Built-in criteria
-----------------
``correctness``, ``helpfulness``, ``conciseness``, ``safety``,
``tool_efficiency``, ``goal_completion``, ``faithfulness``, ``tone``.

Custom criteria::

    from trace_ops.eval import EvalCriteria, LLMJudge

    my_criterion = EvalCriteria(
        name="brand_voice",
        description="Does the response match our brand voice guidelines?",
    )
    judge = LLMJudge(criteria=["correctness", my_criterion])
"""

from trace_ops.eval.assertions import EvalAssertionError, assert_eval_score, assert_passes_criteria
from trace_ops.eval.judge import (
    CriterionScore,
    LLMCallerType,
    LLMJudge,
    TraceEvaluation,
    build_trace_summary,
)
from trace_ops.eval.rubrics import BUILTIN_CRITERIA, EvalCriteria

__all__ = [
    # Rubrics
    "EvalCriteria",
    "BUILTIN_CRITERIA",
    # Judge
    "LLMJudge",
    "LLMCallerType",
    "CriterionScore",
    "TraceEvaluation",
    "build_trace_summary",
    # Assertions
    "assert_eval_score",
    "assert_passes_criteria",
    "EvalAssertionError",
]
