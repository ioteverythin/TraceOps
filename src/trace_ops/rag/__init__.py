"""RAG add-on for traceops.

Provides RAG-aware recording, retrieval assertions, RAG diff, scorer
caching, context-window analysis, retriever snapshot testing, and
dataset export.

Quick start::

    from trace_ops import Recorder
    from trace_ops.rag.assertions import assert_chunk_count, assert_rag_scores
    from trace_ops.rag.scorers import RagasScorer

    # Record with auto-interception + quality scoring
    with Recorder(save_to="cassettes/qa.yaml", intercept_rag=True,
                  rag_scorer=RagasScorer()) as rec:
        result = rag_chain.invoke({"question": "What is the refund policy?"})

    assert_chunk_count(rec.trace, min_chunks=3, max_chunks=10)
    assert_rag_scores(rec.trace, min_faithfulness=0.8)
"""

from trace_ops.rag.assertions import (
    RAGAssertionError,
    assert_chunk_count,
    assert_context_window_usage,
    assert_min_relevance_score,
    assert_no_retrieval_drift,
    assert_rag_scores,
    assert_retrieval_latency,
)
from trace_ops.rag.context_analysis import ContextAnalysis, analyze_context_usage
from trace_ops.rag.diff import RAGDiffResult, RetrievalDiff, diff_rag
from trace_ops.rag.export import to_csv, to_deepeval_dataset, to_openai_finetune, to_ragas_dataset
from trace_ops.rag.recorder import Chunk, EmbeddingEvent, RetrievalEvent
from trace_ops.rag.scorers import BaseRAGScorer, DeepEvalScorer, RagasScorer, ScoringResult
from trace_ops.rag.snapshot import RetrieverSnapshot, SnapshotCheckResult

__all__ = [
    # Assertions
    "RAGAssertionError",
    "assert_chunk_count",
    "assert_retrieval_latency",
    "assert_min_relevance_score",
    "assert_context_window_usage",
    "assert_no_retrieval_drift",
    "assert_rag_scores",
    # Diff
    "diff_rag",
    "RAGDiffResult",
    "RetrievalDiff",
    # Scorers
    "BaseRAGScorer",
    "RagasScorer",
    "DeepEvalScorer",
    "ScoringResult",
    # Context analysis
    "analyze_context_usage",
    "ContextAnalysis",
    # Snapshot
    "RetrieverSnapshot",
    "SnapshotCheckResult",
    # Export
    "to_ragas_dataset",
    "to_deepeval_dataset",
    "to_csv",
    "to_openai_finetune",
    # Dataclasses
    "Chunk",
    "RetrievalEvent",
    "EmbeddingEvent",
]
