"""Core data model for agent execution traces.

A Trace is a complete recording of an agent run. It contains a sequence of
Events — each event is either an LLM call, a tool invocation, or an agent
decision. The trace captures everything needed to deterministically replay
the agent's execution without making real API calls.

Key design decision: we record at the SDK level (intercepting openai.chat.completions.create,
anthropic.messages.create, etc.) rather than at the HTTP level (like VCR.py). This gives us
semantic understanding of what happened — we know "this was a tool call" vs "this was a
completion" — which HTTP-level recording can't distinguish.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    """Types of events in an agent trace."""

    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_DECISION = "agent_decision"
    ERROR = "error"

    # RAG events
    RETRIEVAL = "retrieval"
    EMBEDDING_CALL = "embedding_call"
    RAG_SCORES = "rag_scores"

    # MCP events
    MCP_SERVER_CONNECT = "mcp_server_connect"
    MCP_TOOL_CALL = "mcp_tool_call"
    MCP_TOOL_RESULT = "mcp_tool_result"


@dataclass
class TraceEvent:
    """A single event in an agent execution trace.

    Events are the atoms of a trace. Each event records one interaction:
    an LLM call, a tool invocation, or an agent-level decision.
    """

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])

    # LLM-specific fields
    provider: str | None = None  # "openai", "anthropic", "litellm"
    model: str | None = None
    messages: list[dict[str, Any]] | None = None  # input messages
    response: dict[str, Any] | None = None  # full response object
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None  # tool definitions sent to LLM

    # Tool-specific fields
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any = None

    # Agent decision fields
    decision: str | None = None  # e.g., "delegate_to_agent_b", "select_tool_search"
    reasoning: str | None = None

    # Error fields
    error_type: str | None = None
    error_message: str | None = None

    # Cost tracking
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    # Timing
    duration_ms: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # RAG-specific fields
    query: str | None = None                          # retrieval query text
    chunks: list[dict[str, Any]] | None = None        # retrieved chunks [{id, text, score, metadata}]
    vector_store: str | None = None                   # "chromadb", "pinecone", etc.
    collection: str | None = None                     # collection / index name
    top_k: int | None = None
    total_chunks_searched: int | None = None
    dimensions: int | None = None                     # embedding dimensions
    scores: dict[str, float] | None = None            # RAG quality scores

    # MCP-specific fields
    server_name: str | None = None
    server_url: str | None = None
    capabilities: list[str] | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    is_error: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict, dropping None fields for compact storage."""
        d: dict[str, Any] = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }
        for key in [
            "provider", "model", "messages", "response", "temperature",
            "max_tokens", "tools", "tool_name", "tool_input", "tool_output",
            "decision", "reasoning", "error_type", "error_message",
            "input_tokens", "output_tokens", "cost_usd", "duration_ms",
            # RAG fields
            "query", "chunks", "vector_store", "collection", "top_k",
            "total_chunks_searched", "dimensions", "scores",
            # MCP fields
            "server_name", "server_url", "capabilities", "arguments", "result", "is_error",
        ]:
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        """Deserialize from a dict."""
        data = dict(data)
        data["event_type"] = EventType(data["event_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TraceMetadata:
    """Metadata about the trace recording environment."""

    recorded_at: float = field(default_factory=time.time)
    trace_ops_version: str = "0.5.0"
    python_version: str = ""
    framework: str | None = None  # "langchain", "crewai", "openai-agents-sdk", "custom"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "recorded_at": self.recorded_at,
            "trace_ops_version": self.trace_ops_version,
        }
        if self.python_version:
            d["python_version"] = self.python_version
        if self.framework:
            d["framework"] = self.framework
        if self.description:
            d["description"] = self.description
        if self.tags:
            d["tags"] = self.tags
        if self.env:
            d["env"] = self.env
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Trace:
    """A complete recording of an agent execution.

    This is the top-level object that gets saved to a cassette file.
    It contains all events from a single agent run, plus metadata
    about the recording environment.
    """

    trace_id: str = field(default_factory=lambda: uuid4().hex[:16])
    events: list[TraceEvent] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=TraceMetadata)

    # Summary stats (computed after recording)
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0

    # Thread safety — protects events list and stats
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def add_event(self, event: TraceEvent) -> None:
        """Add an event to the trace (thread-safe)."""
        with self._lock:
            self.events.append(event)
            self._update_stats(event)

    def _update_stats(self, event: TraceEvent) -> None:
        """Update summary statistics after adding an event."""
        if event.event_type == EventType.LLM_RESPONSE:
            self.total_llm_calls += 1
            if event.input_tokens:
                self.total_tokens += event.input_tokens
            if event.output_tokens:
                self.total_tokens += event.output_tokens
            if event.cost_usd:
                self.total_cost_usd += event.cost_usd
        elif event.event_type == EventType.TOOL_RESULT:
            self.total_tool_calls += 1
        elif event.event_type == EventType.EMBEDDING_CALL:
            if event.cost_usd:
                self.total_cost_usd += event.cost_usd
        if event.duration_ms:
            self.total_duration_ms += event.duration_ms

    def finalize(self) -> None:
        """Compute final stats after recording is complete."""
        self.total_llm_calls = sum(
            1 for e in self.events if e.event_type == EventType.LLM_RESPONSE
        )
        self.total_tool_calls = sum(
            1 for e in self.events if e.event_type == EventType.TOOL_RESULT
        )
        self.total_tokens = sum(
            (e.input_tokens or 0) + (e.output_tokens or 0)
            for e in self.events
            if e.event_type == EventType.LLM_RESPONSE
        )
        self.total_cost_usd = sum(
            e.cost_usd or 0.0
            for e in self.events
            if e.cost_usd is not None
        )
        self.total_duration_ms = sum(
            e.duration_ms or 0.0 for e in self.events if e.duration_ms
        )

    @property
    def llm_events(self) -> list[TraceEvent]:
        """Get only LLM request/response events."""
        return [
            e for e in self.events
            if e.event_type in (EventType.LLM_REQUEST, EventType.LLM_RESPONSE)
        ]

    @property
    def tool_events(self) -> list[TraceEvent]:
        """Get only tool call/result events."""
        return [
            e for e in self.events
            if e.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
        ]

    @property
    def retrieval_events(self) -> list[TraceEvent]:
        """Get all retrieval events (RAG vector store queries)."""
        return [e for e in self.events if e.event_type == EventType.RETRIEVAL]

    @property
    def embedding_events(self) -> list[TraceEvent]:
        """Get all embedding call events."""
        return [e for e in self.events if e.event_type == EventType.EMBEDDING_CALL]

    @property
    def mcp_events(self) -> list[TraceEvent]:
        """Get all MCP-related events."""
        return [
            e for e in self.events
            if e.event_type in (
                EventType.MCP_SERVER_CONNECT,
                EventType.MCP_TOOL_CALL,
                EventType.MCP_TOOL_RESULT,
            )
        ]

    @property
    def rag_scores(self) -> dict[str, float] | None:
        """Get cached RAG quality scores from the cassette, if any."""
        for e in self.events:
            if e.event_type == EventType.RAG_SCORES and e.scores:
                return e.scores
        return None

    @property
    def trajectory(self) -> list[str]:
        """Get the high-level trajectory as a list of step descriptions.

        Returns something like:
            ["llm_call:gpt-4o", "tool:search_files", "llm_call:gpt-4o", "tool:read_file"]
        """
        steps = []
        for event in self.events:
            if event.event_type == EventType.LLM_REQUEST:
                steps.append(f"llm_call:{event.model or 'unknown'}")
            elif event.event_type == EventType.TOOL_CALL:
                steps.append(f"tool:{event.tool_name or 'unknown'}")
            elif event.event_type == EventType.AGENT_DECISION:
                steps.append(f"decision:{event.decision or 'unknown'}")
            elif event.event_type == EventType.ERROR:
                steps.append(f"error:{event.error_type or 'unknown'}")
            elif event.event_type == EventType.RETRIEVAL:
                steps.append(f"retrieval:{event.vector_store or 'unknown'}")
            elif event.event_type == EventType.EMBEDDING_CALL:
                steps.append(f"embedding:{event.model or 'unknown'}")
            elif event.event_type == EventType.MCP_TOOL_CALL:
                steps.append(f"mcp:{event.server_name or 'unknown'}.{event.tool_name or 'unknown'}")
        return steps

    def fingerprint(self) -> str:
        """Generate a hash fingerprint of the trajectory.

        Two traces with the same fingerprint took the same path
        (same sequence of LLM calls, tool calls, and decisions).
        """
        trajectory_str = "|".join(self.trajectory)
        return hashlib.sha256(trajectory_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trace to a dict."""
        return {
            "version": "1",
            "trace_id": self.trace_id,
            "metadata": self.metadata.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "summary": {
                "total_llm_calls": self.total_llm_calls,
                "total_tool_calls": self.total_tool_calls,
                "total_tokens": self.total_tokens,
                "total_cost_usd": self.total_cost_usd,
                "total_duration_ms": self.total_duration_ms,
                "trajectory": self.trajectory,
                "fingerprint": self.fingerprint(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Deserialize a trace from a dict."""
        trace = cls(
            trace_id=data.get("trace_id", uuid4().hex[:16]),
            metadata=TraceMetadata.from_dict(data.get("metadata", {})),
        )
        for event_data in data.get("events", []):
            trace.events.append(TraceEvent.from_dict(event_data))
        trace.finalize()
        return trace
