"""traceops — record and replay LLM agent traces for deterministic regression testing.

Framework-agnostic. Works with OpenAI, Anthropic, LiteLLM, LangChain, CrewAI,
or any custom agent. No LLM calls during replay — tests are fast, free, deterministic.

Quick start:
    from trace_ops import Recorder, Replayer

    # Record an agent run
    with Recorder(save_to="cassettes/test_math.yaml") as rec:
        result = agent.run("What is 2+2?")

    # Replay deterministically (zero API calls)
    with Replayer("cassettes/test_math.yaml"):
        result = agent.run("What is 2+2?")
        assert result == "4"

    # Also works async:
    async with Recorder(save_to="cassettes/test.yaml") as rec:
        result = await agent.arun("What is 2+2?")
"""

from trace_ops._types import (
    EventType,
    Trace,
    TraceEvent,
    TraceMetadata,
)
from trace_ops.assertions import (
    AgentLoopError,
    BudgetExceededError,
    assert_cost_under,
    assert_max_llm_calls,
    assert_no_loops,
    assert_tokens_under,
)
from trace_ops.cassette import (
    CassetteMismatchError,
    CassetteNotFoundError,
    load_cassette,
    save_cassette,
)
from trace_ops.diff import TraceDiff, assert_trace_unchanged, diff_traces
from trace_ops.normalize import (
    NormalizedResponse,
    NormalizedToolCall,
    normalize_for_comparison,
    normalize_response,
)
from trace_ops.recorder import Recorder
from trace_ops.replayer import Replayer
from trace_ops.reporters.cost_dashboard import CostDashboard, CostSummary

# RAG add-on (graceful degradation if not installed)
try:
    from trace_ops.rag.diff import RAGDiffResult, diff_rag
    from trace_ops.rag.assertions import (
        RAGAssertionError,
        assert_chunk_count,
        assert_retrieval_latency,
        assert_min_relevance_score,
        assert_no_retrieval_drift,
        assert_rag_scores,
    )
    from trace_ops.rag.scorers import RagasScorer, DeepEvalScorer
    from trace_ops.rag.snapshot import RetrieverSnapshot
    from trace_ops.rag.context_analysis import analyze_context_usage
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# MCP add-on
try:
    from trace_ops.mcp.diff import MCPDiffResult, diff_mcp
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# Semantic add-on
try:
    from trace_ops.semantic.similarity import SemanticDiffResult, semantic_similarity
    from trace_ops.semantic.assertions import SemanticRegressionError, assert_semantic_similarity
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

# Export add-on
try:
    from trace_ops.export.finetune import to_openai_finetune, to_anthropic_finetune
    _EXPORT_AVAILABLE = True
except ImportError:
    _EXPORT_AVAILABLE = False

__version__ = "0.5.0"

__all__ = [
    # Core
    "Recorder",
    "Replayer",
    # Types
    "Trace",
    "TraceEvent",
    "TraceMetadata",
    "EventType",
    # Cassette
    "save_cassette",
    "load_cassette",
    "CassetteNotFoundError",
    "CassetteMismatchError",
    # Diff
    "TraceDiff",
    "diff_traces",
    "assert_trace_unchanged",
    # Normalization
    "NormalizedToolCall",
    "NormalizedResponse",
    "normalize_response",
    "normalize_for_comparison",
    # Assertions
    "assert_cost_under",
    "assert_tokens_under",
    "assert_max_llm_calls",
    "assert_no_loops",
    "BudgetExceededError",
    "AgentLoopError",
    # Reporters
    "CostDashboard",
    "CostSummary",
    # RAG (available when trace_ops[rag] installed)
    "diff_rag",
    "RAGDiffResult",
    "RAGAssertionError",
    "assert_chunk_count",
    "assert_retrieval_latency",
    "assert_min_relevance_score",
    "assert_no_retrieval_drift",
    "assert_rag_scores",
    "RagasScorer",
    "DeepEvalScorer",
    "RetrieverSnapshot",
    "analyze_context_usage",
    # MCP
    "diff_mcp",
    "MCPDiffResult",
    # Semantic
    "semantic_similarity",
    "SemanticDiffResult",
    "SemanticRegressionError",
    "assert_semantic_similarity",
    # Export / fine-tune
    "to_openai_finetune",
    "to_anthropic_finetune",
]
