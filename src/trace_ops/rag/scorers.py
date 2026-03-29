"""RAG quality scorers — compute metrics and cache them in the cassette.

Scorers run during recording (not replay), call a judge LLM once, and write
the scores into the cassette. On replay the scores are read directly —
zero additional LLM calls, zero cost.

Usage::

    from trace_ops import Recorder
    from trace_ops.rag.scorers import RagasScorer

    with Recorder(save_to="cassettes/qa.yaml", rag_scorer=RagasScorer()) as rec:
        result = rag_chain.invoke({"question": "What is the refund policy?"})
    # cassette now contains cached faithfulness / context_precision / answer_relevancy
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringResult:
    """Result from a RAG scorer."""

    scores: dict[str, float]
    scorer: str
    judge_model: str
    judge_tokens: int = 0
    judge_cost_usd: float = 0.0
    judge_duration_ms: float = 0.0


class BaseRAGScorer(ABC):
    """Abstract base class for RAG scorers."""

    @abstractmethod
    def score(self, query: str, context_chunks: list[str], response: str) -> ScoringResult:
        """Compute RAG metrics for a query-context-response triple.

        Args:
            query: The user's question / retrieval query.
            context_chunks: Retrieved text chunks used as context.
            response: The LLM's final response.

        Returns:
            A :class:`ScoringResult` with named metric scores.
        """
        ...


class RagasScorer(BaseRAGScorer):
    """Compute RAG quality metrics using the RAGAS library.

    Requires ``pip install traceops[ragas]``.

    Args:
        metrics: List of metric names to compute.  Defaults to
            ``["faithfulness", "context_precision", "answer_relevancy"]``.
        judge_model: LLM model name for the RAGAS judge (default ``gpt-4o-mini``).
    """

    DEFAULT_METRICS = ["faithfulness", "context_precision", "answer_relevancy"]

    def __init__(
        self,
        metrics: list[str] | None = None,
        judge_model: str = "gpt-4o-mini",
    ) -> None:
        self.metric_names = metrics or self.DEFAULT_METRICS
        self.judge_model = judge_model

    def score(self, query: str, context_chunks: list[str], response: str) -> ScoringResult:
        """Run RAGAS evaluation and return scores."""
        try:
            from ragas import evaluate
            from ragas.dataset_schema import SingleTurnSample
        except ImportError as exc:
            raise ImportError(
                "Install ragas: pip install traceops[ragas]"
            ) from exc

        # Build metric objects
        metrics = self._build_metrics()

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=context_chunks,
            response=response,
        )

        t0 = time.perf_counter()
        try:
            result = evaluate(dataset=[sample], metrics=metrics)
        except Exception as exc:
            raise RuntimeError(f"RAGAS evaluation failed: {exc}") from exc
        duration_ms = (time.perf_counter() - t0) * 1000

        scores: dict[str, float] = {}
        for name in self.metric_names:
            val = result.get(name) if hasattr(result, "get") else None
            if val is not None:
                scores[name] = float(val)

        return ScoringResult(
            scores=scores,
            scorer="ragas",
            judge_model=self.judge_model,
            judge_duration_ms=duration_ms,
        )

    def _build_metrics(self) -> list[Any]:
        try:
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                ResponseRelevancy,
                ContextRecall,
            )
        except ImportError as exc:
            raise ImportError("Install ragas: pip install traceops[ragas]") from exc

        name_map: dict[str, Any] = {
            "faithfulness": Faithfulness,
            "context_precision": LLMContextPrecisionWithoutReference,
            "answer_relevancy": ResponseRelevancy,
            "context_recall": ContextRecall,
        }
        metrics = []
        for name in self.metric_names:
            cls = name_map.get(name)
            if cls is not None:
                metrics.append(cls())
        return metrics


class DeepEvalScorer(BaseRAGScorer):
    """Compute RAG quality metrics using the DeepEval library.

    Requires ``pip install traceops[deepeval]``.

    Args:
        metrics: List of metric names to compute.  Defaults to
            ``["faithfulness", "contextual_relevancy", "answer_relevancy"]``.
        judge_model: Model name for the DeepEval judge (default ``gpt-4o-mini``).
    """

    DEFAULT_METRICS = ["faithfulness", "contextual_relevancy", "answer_relevancy"]

    def __init__(
        self,
        metrics: list[str] | None = None,
        judge_model: str = "gpt-4o-mini",
    ) -> None:
        self.metric_names = metrics or self.DEFAULT_METRICS
        self.judge_model = judge_model

    def score(self, query: str, context_chunks: list[str], response: str) -> ScoringResult:
        """Run DeepEval evaluation and return scores."""
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:
            raise ImportError(
                "Install deepeval: pip install traceops[deepeval]"
            ) from exc

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=context_chunks,
        )

        scores: dict[str, float] = {}
        t0 = time.perf_counter()
        for name in self.metric_names:
            metric = self._build_metric(name)
            if metric is not None:
                metric.measure(test_case)
                scores[name] = float(getattr(metric, "score", 0.0))
        duration_ms = (time.perf_counter() - t0) * 1000

        return ScoringResult(
            scores=scores,
            scorer="deepeval",
            judge_model=self.judge_model,
            judge_duration_ms=duration_ms,
        )

    def _build_metric(self, name: str) -> Any | None:
        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                ContextualRelevancyMetric,
                AnswerRelevancyMetric,
            )
        except ImportError as exc:
            raise ImportError(
                "Install deepeval: pip install traceops[deepeval]"
            ) from exc

        name_map = {
            "faithfulness": lambda: FaithfulnessMetric(model=self.judge_model),
            "contextual_relevancy": lambda: ContextualRelevancyMetric(model=self.judge_model),
            "answer_relevancy": lambda: AnswerRelevancyMetric(model=self.judge_model),
        }
        factory = name_map.get(name)
        return factory() if factory else None
