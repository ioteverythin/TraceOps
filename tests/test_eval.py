"""Tests for the trace_ops.eval LLM-as-judge module.

All tests use a mock ``_llm_caller`` so no real API calls are made.
"""

from __future__ import annotations

import json

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.eval import (
    BUILTIN_CRITERIA,
    CriterionScore,
    EvalAssertionError,
    EvalCriteria,
    LLMJudge,
    TraceEvaluation,
    assert_eval_score,
    assert_passes_criteria,
    build_trace_summary,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_trace(events: list[TraceEvent] | None = None) -> Trace:
    """Return a minimal Trace for testing."""
    t = Trace(trace_id="test-trace")
    if events:
        for e in events:
            t.add_event(e)
    t.finalize()
    return t


def _simple_trace() -> Trace:
    """A two-event trace: user asks, assistant responds."""
    return _make_trace([
        TraceEvent(
            event_type=EventType.LLM_REQUEST,
            provider="openai",
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
        ),
        TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            provider="openai",
            model="gpt-4o",
            response={
                "choices": [{"message": {"content": "The answer is 4."}}]
            },
            input_tokens=10,
            output_tokens=8,
        ),
    ])


def _tool_trace() -> Trace:
    """A trace with a tool call + result."""
    return _make_trace([
        TraceEvent(
            event_type=EventType.LLM_REQUEST,
            messages=[{"role": "user", "content": "Search for cats."}],
        ),
        TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name="web_search",
            tool_input={"query": "cats"},
        ),
        TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_name="web_search",
            tool_output="Cats are small domesticated mammals.",
        ),
        TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            response={
                "choices": [{"message": {"content": "Cats are small domesticated mammals."}}]
            },
        ),
    ])


def _error_trace() -> Trace:
    """A trace that includes an error event."""
    return _make_trace([
        TraceEvent(
            event_type=EventType.LLM_REQUEST,
            messages=[{"role": "user", "content": "Do something."}],
        ),
        TraceEvent(
            event_type=EventType.ERROR,
            error_type="RuntimeError",
            error_message="Something went wrong",
        ),
    ])


def _mock_caller(scores: dict[str, int]):
    """Return a _llm_caller that returns *scores* (raw 1-5) for each criterion."""
    def caller(system: str, user: str, model: str) -> tuple[str, int]:
        evals = [
            {"criterion": name, "score": val, "reasoning": f"Mock reasoning for {name}."}
            for name, val in scores.items()
        ]
        return json.dumps({"evaluations": evals}), 42
    return caller


def _high_scorer(_sys: str, _user: str, _model: str) -> tuple[str, int]:
    """A caller that gives 5/5 on correctness, helpfulness, tool_efficiency."""
    evals = [
        {"criterion": "correctness", "score": 5, "reasoning": "Perfect."},
        {"criterion": "helpfulness", "score": 5, "reasoning": "Very helpful."},
        {"criterion": "tool_efficiency", "score": 5, "reasoning": "Efficient."},
    ]
    return json.dumps({"evaluations": evals}), 100


def _low_scorer(_sys: str, _user: str, _model: str) -> tuple[str, int]:
    """A caller that gives 1/5 on all default criteria."""
    evals = [
        {"criterion": "correctness", "score": 1, "reasoning": "Wrong."},
        {"criterion": "helpfulness", "score": 1, "reasoning": "Not helpful."},
        {"criterion": "tool_efficiency", "score": 1, "reasoning": "Wasteful."},
    ]
    return json.dumps({"evaluations": evals}), 50


# ── EvalCriteria tests ───────────────────────────────────────────────


class TestEvalCriteria:
    def test_builtin_criteria_count(self):
        assert len(BUILTIN_CRITERIA) == 8

    def test_all_names_present(self):
        expected = {
            "correctness", "helpfulness", "conciseness", "safety",
            "tool_efficiency", "goal_completion", "faithfulness", "tone",
        }
        assert set(BUILTIN_CRITERIA.keys()) == expected

    def test_prompt_text(self):
        c = BUILTIN_CRITERIA["correctness"]
        text = c.prompt_text()
        assert "correctness" in text
        assert "1–5" in text

    def test_custom_criteria(self):
        c = EvalCriteria(name="brand", description="Brand voice?", scale_min=0, scale_max=10)
        assert c.scale_min == 0
        assert c.scale_max == 10
        assert "brand" in c.prompt_text()
        assert "0–10" in c.prompt_text()


# ── build_trace_summary tests ────────────────────────────────────────


class TestBuildTraceSummary:
    def test_simple_trace(self):
        summary = build_trace_summary(_simple_trace())
        assert "[USER] What is 2+2?" in summary
        assert "[ASSISTANT] The answer is 4." in summary

    def test_tool_trace(self):
        summary = build_trace_summary(_tool_trace())
        assert "[TOOL CALL] web_search" in summary
        assert "[TOOL RESULT] web_search" in summary

    def test_error_trace(self):
        summary = build_trace_summary(_error_trace())
        assert "[ERROR]" in summary
        assert "Something went wrong" in summary

    def test_empty_trace(self):
        summary = build_trace_summary(_make_trace())
        assert summary == "(empty trace)"

    def test_anthropic_response_format(self):
        """Trace with Anthropic-style response should be parsed."""
        t = _make_trace([
            TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                response={"content": [{"text": "Hello from Claude!"}]},
            ),
        ])
        summary = build_trace_summary(t)
        assert "Hello from Claude!" in summary

    def test_string_response(self):
        t = _make_trace([
            TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                response="Plain string response",
            ),
        ])
        summary = build_trace_summary(t)
        assert "Plain string response" in summary


# ── LLMJudge tests ──────────────────────────────────────────────────


class TestLLMJudge:
    def test_evaluate_default_criteria(self):
        judge = LLMJudge(_llm_caller=_high_scorer)
        result = judge.evaluate(_simple_trace())

        assert isinstance(result, TraceEvaluation)
        assert len(result.scores) == 3
        assert result.overall_score == 1.0
        assert result.judge_tokens == 100

    def test_evaluate_with_extra_criteria(self):
        judge = LLMJudge(_llm_caller=_mock_caller({
            "correctness": 5,
            "helpfulness": 4,
            "tool_efficiency": 3,
            "safety": 5,
        }))
        result = judge.evaluate(_simple_trace(), extra_criteria=["safety"])
        names = {s.criterion for s in result.scores}
        assert "safety" in names

    def test_custom_criteria_object(self):
        custom = EvalCriteria(name="brand_voice", description="Brand match?")
        judge = LLMJudge(
            criteria=[custom],
            _llm_caller=_mock_caller({"brand_voice": 4}),
        )
        result = judge.evaluate(_simple_trace())
        assert result.scores[0].criterion == "brand_voice"
        assert result.scores[0].raw_score == 4

    def test_unknown_criterion_name_creates_generic(self):
        judge = LLMJudge(
            criteria=["mystery_metric"],
            _llm_caller=_mock_caller({"mystery_metric": 3}),
        )
        result = judge.evaluate(_simple_trace())
        assert result.scores[0].criterion == "mystery_metric"
        assert result.scores[0].score == 0.5  # raw=3 → (3-1)/4 = 0.5

    def test_normalisation(self):
        """Scores normalise raw 1-5 to 0.0-1.0."""
        judge = LLMJudge(
            criteria=["correctness"],
            _llm_caller=_mock_caller({"correctness": 1}),
        )
        result = judge.evaluate(_simple_trace())
        assert result.scores[0].score == 0.0  # raw=1 → (1-1)/4 = 0

        judge2 = LLMJudge(
            criteria=["correctness"],
            _llm_caller=_mock_caller({"correctness": 5}),
        )
        result2 = judge2.evaluate(_simple_trace())
        assert result2.scores[0].score == 1.0  # raw=5 → (5-1)/4 = 1

    def test_missing_criterion_defaults(self):
        """If the judge omits a criterion from the response, it defaults to 0.5."""
        judge = LLMJudge(
            criteria=["correctness", "safety"],
            _llm_caller=_mock_caller({"correctness": 5}),  # safety missing
        )
        result = judge.evaluate(_simple_trace())
        safety = result.score_for("safety")
        assert safety is not None
        assert safety.score == 0.5

    def test_invalid_json_raises(self):
        def bad_caller(_s: str, _u: str, _m: str) -> tuple[str, int]:
            return "this is not json", 10

        judge = LLMJudge(_llm_caller=bad_caller)
        with pytest.raises(RuntimeError, match="invalid JSON"):
            judge.evaluate(_simple_trace())

    def test_judge_model_metadata(self):
        judge = LLMJudge(model="claude-3-haiku", provider="anthropic", _llm_caller=_high_scorer)
        result = judge.evaluate(_simple_trace())
        assert result.judge_model == "claude-3-haiku"
        assert result.judge_duration_ms >= 0

    def test_cost_estimate(self):
        result = LLMJudge(_llm_caller=_high_scorer).evaluate(_simple_trace())
        assert result.judge_cost_usd > 0

    def test_unknown_provider_raises(self):
        judge = LLMJudge(provider="gemini")
        with pytest.raises(ValueError, match="Unknown provider"):
            judge.evaluate(_simple_trace())


# ── TraceEvaluation tests ────────────────────────────────────────────


class TestTraceEvaluation:
    def test_score_for(self):
        result = LLMJudge(_llm_caller=_high_scorer).evaluate(_simple_trace())
        s = result.score_for("correctness")
        assert s is not None
        assert s.criterion == "correctness"

    def test_score_for_missing(self):
        result = LLMJudge(_llm_caller=_high_scorer).evaluate(_simple_trace())
        assert result.score_for("nonexistent") is None

    def test_passes(self):
        result = LLMJudge(_llm_caller=_high_scorer).evaluate(_simple_trace())
        assert result.passes("correctness", min_score=0.5) is True

    def test_passes_fails(self):
        result = LLMJudge(_llm_caller=_low_scorer).evaluate(_simple_trace())
        assert result.passes("correctness", min_score=0.5) is False

    def test_summary_string(self):
        result = LLMJudge(_llm_caller=_high_scorer).evaluate(_simple_trace())
        text = result.summary()
        assert "correctness" in text
        assert "overall=" in text

    def test_to_dict(self):
        result = LLMJudge(_llm_caller=_high_scorer).evaluate(_simple_trace())
        d = result.to_dict()
        assert "overall_score" in d
        assert "scores" in d
        assert isinstance(d["scores"], list)
        assert d["scores"][0]["criterion"] == "correctness"

    def test_overall_is_mean(self):
        judge = LLMJudge(
            criteria=["correctness", "helpfulness"],
            _llm_caller=_mock_caller({"correctness": 5, "helpfulness": 1}),
        )
        result = judge.evaluate(_simple_trace())
        # (1.0 + 0.0) / 2 = 0.5
        assert result.overall_score == 0.5


# ── assertion tests ──────────────────────────────────────────────────


class TestAssertions:
    def test_assert_eval_score_passes(self):
        judge = LLMJudge(_llm_caller=_high_scorer)
        result = assert_eval_score(_simple_trace(), min_score=0.5, judge=judge)
        assert isinstance(result, TraceEvaluation)

    def test_assert_eval_score_fails(self):
        judge = LLMJudge(_llm_caller=_low_scorer)
        with pytest.raises(EvalAssertionError, match="scored below"):
            assert_eval_score(_simple_trace(), min_score=0.5, judge=judge)

    def test_assert_eval_score_custom_criteria(self):
        judge = LLMJudge(_llm_caller=_mock_caller({"safety": 5}))
        result = assert_eval_score(
            _simple_trace(), criteria=["safety"], min_score=0.8, judge=judge
        )
        assert result.score_for("safety") is not None

    def test_assert_passes_criteria(self):
        judge = LLMJudge(_llm_caller=_mock_caller({"correctness": 5, "safety": 5}))
        result = assert_passes_criteria(
            _simple_trace(), ["correctness", "safety"], judge=judge
        )
        assert result.overall_score == 1.0

    def test_assert_passes_criteria_fails(self):
        judge = LLMJudge(_llm_caller=_mock_caller({"correctness": 1, "safety": 1}))
        with pytest.raises(EvalAssertionError):
            assert_passes_criteria(
                _simple_trace(), ["correctness", "safety"], judge=judge
            )

    def test_eval_assertion_is_assertion_error(self):
        """EvalAssertionError is a subclass of AssertionError."""
        assert issubclass(EvalAssertionError, AssertionError)


# ── CriterionScore tests ────────────────────────────────────────────


class TestCriterionScore:
    def test_fields(self):
        s = CriterionScore(criterion="test", score=0.8, raw_score=4, reasoning="Good.")
        assert s.criterion == "test"
        assert s.score == 0.8
        assert s.raw_score == 4
        assert s.reasoning == "Good."
