"""Replayer — injects recorded LLM responses for deterministic replay.

The replayer monkey-patches LLM SDK methods just like the recorder, but
instead of forwarding calls to real APIs and recording responses, it
intercepts the calls and returns the pre-recorded responses from a cassette.

If the agent's execution diverges from the recording (different prompt,
different tool call, different sequence), the replay fails explicitly
with a CassetteMismatchError.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.cassette import CassetteMismatchError, load_cassette


class Replayer:
    """Replays agent traces from a cassette file.

    Intercepts LLM SDK calls and returns pre-recorded responses instead
    of making real API calls. Zero API cost, millisecond execution,
    fully deterministic.

    Usage as context manager:
        with Replayer("cassettes/test_math.yaml") as replayer:
            # Agent code runs here but all LLM calls return recorded responses
            result = agent.run("What is 2+2?")

        # If the agent took a different path, Replayer raises CassetteMismatchError

    Usage as decorator:
        @Replayer.replay("cassettes/test_math.yaml")
        def test_my_agent():
            result = agent.run("What is 2+2?")
    """

    def __init__(
        self,
        cassette_path: str,
        *,
        strict: bool = True,
        allow_new_calls: bool = False,
        intercept_openai: bool = True,
        intercept_anthropic: bool = True,
        intercept_litellm: bool = True,
        intercept_langchain: bool = True,
        intercept_langgraph: bool = True,
        intercept_crewai: bool = True,
        intercept_rag: bool = False,
        intercept_mcp: bool = False,
    ) -> None:
        """
        Args:
            cassette_path: Path to the YAML cassette file.
            strict: If True, fail on any divergence from the recording.
                    If False, log warnings but continue.
            allow_new_calls: If True, pass through unrecorded calls to real APIs.
                            Useful for extending cassettes incrementally.
            intercept_rag: If True, replay recorded RAG retrieval events so tests
                           run without a live vector store.
            intercept_mcp: If True, replay recorded MCP tool call events so tests
                           run without a live MCP server.
        """
        self.cassette_path = cassette_path
        self.strict = strict
        self.allow_new_calls = allow_new_calls
        self._trace: Trace | None = None
        self._response_queue: list[TraceEvent] = []
        self._call_index = 0
        self._patches: list[Any] = []
        self._intercept_openai = intercept_openai
        self._intercept_anthropic = intercept_anthropic
        self._intercept_litellm = intercept_litellm
        self._intercept_langchain = intercept_langchain
        self._intercept_langgraph = intercept_langgraph
        self._intercept_crewai = intercept_crewai
        self._intercept_rag = intercept_rag
        self._intercept_mcp = intercept_mcp
        self._replay_trace: Trace | None = None  # what actually happened during replay
        self._lock = threading.Lock()
        # RAG replay state
        self._retrieval_queue: list[TraceEvent] = []
        self._retrieval_index = 0
        self._retrieval_lock = threading.Lock()
        # MCP replay state
        self._mcp_queue: list[TraceEvent] = []
        self._mcp_index = 0
        self._mcp_lock = threading.Lock()
        # RAG/MCP patch restore tuples: (name, original, cls, attr)
        self._rag_replay_patches: list[tuple[str, Any, Any, str]] = []
        self._mcp_replay_patches: list[tuple[str, Any, Any, str]] = []

    @property
    def recorded_trace(self) -> Trace | None:
        """The original recorded trace from the cassette."""
        return self._trace

    @property
    def replay_trace(self) -> Trace | None:
        """Trace of what happened during replay (for comparison)."""
        return self._replay_trace

    def _load(self) -> None:
        """Load the cassette and prepare the response queue."""
        self._trace = load_cassette(self.cassette_path)
        self._replay_trace = Trace()

        # Build queue of LLM responses in order
        self._response_queue = [
            e for e in self._trace.events
            if e.event_type == EventType.LLM_RESPONSE
        ]
        self._call_index = 0

        # Build queues for RAG/MCP replay
        self._retrieval_queue = [
            e for e in self._trace.events
            if e.event_type == EventType.RETRIEVAL
        ]
        self._retrieval_index = 0

        self._mcp_queue = [
            e for e in self._trace.events
            if e.event_type == EventType.MCP_TOOL_RESULT
        ]
        self._mcp_index = 0

    def _get_next_response(self, provider: str, model: str) -> dict[str, Any]:
        """Get the next recorded LLM response from the queue (thread-safe).

        Raises CassetteMismatchError if we've exhausted all responses.
        """
        with self._lock:
            if self._call_index >= len(self._response_queue):
                msg = (
                    f"Agent made more LLM calls than recorded.\n"
                    f"Expected {len(self._response_queue)} calls, "
                    f"but got call #{self._call_index + 1} "
                    f"(provider={provider}, model={model}).\n"
                    f"Re-record the cassette to capture the new behavior."
                )
                if self.strict:
                    raise CassetteMismatchError(msg)
                return {}

            event = self._response_queue[self._call_index]
            self._call_index += 1

            # Verify the call matches (provider + model)
            if self.strict:
                if event.provider and event.provider != provider:
                    raise CassetteMismatchError(
                        f"Call #{self._call_index}: expected provider "
                        f"'{event.provider}', got '{provider}'.",
                        expected_event=event.to_dict(),
                    )
                if event.model and event.model != model:
                    raise CassetteMismatchError(
                        f"Call #{self._call_index}: expected model "
                        f"'{event.model}', got '{model}'.",
                        expected_event=event.to_dict(),
                    )

            return event.response or {}

    def _get_next_retrieval(self) -> TraceEvent | None:
        """Get the next recorded retrieval event from the queue (thread-safe)."""
        with self._retrieval_lock:
            if self._retrieval_index >= len(self._retrieval_queue):
                return None
            event = self._retrieval_queue[self._retrieval_index]
            self._retrieval_index += 1
            return event

    def _get_next_mcp_result(self, tool_name: str) -> TraceEvent | None:
        """Get the next recorded MCP tool result event (thread-safe)."""
        with self._mcp_lock:
            if self._mcp_index >= len(self._mcp_queue):
                return None
            event = self._mcp_queue[self._mcp_index]
            self._mcp_index += 1
            return event

    def _install_rag_replay_patches(self) -> None:
        """Patch RAG retriever methods to return recorded chunks."""
        replayer = self

        # LangChain VectorStoreRetriever replay
        try:
            from langchain_core.vectorstores import VectorStoreRetriever

            original = VectorStoreRetriever._get_relevant_documents

            def patched_get_docs(self_inner: Any, query: str, **kwargs: Any) -> list[Any]:
                event = replayer._get_next_retrieval()
                if event is None or not event.chunks:
                    return []
                from types import SimpleNamespace
                return [
                    SimpleNamespace(
                        page_content=c.get("content", ""),
                        metadata=c.get("metadata", {}),
                    )
                    for c in event.chunks
                ]

            self._rag_replay_patches.append(
                ("langchain_retriever", original, VectorStoreRetriever, "_get_relevant_documents")
            )
            VectorStoreRetriever._get_relevant_documents = patched_get_docs  # type: ignore[assignment]
        except ImportError:
            pass

        # LlamaIndex BaseRetriever replay
        try:
            from llama_index.core.retrievers import BaseRetriever

            original_retrieve = BaseRetriever.retrieve

            def patched_retrieve(self_inner: Any, query: Any, **kwargs: Any) -> list[Any]:
                event = replayer._get_next_retrieval()
                if event is None or not event.chunks:
                    return []
                from types import SimpleNamespace
                return [
                    SimpleNamespace(
                        node=SimpleNamespace(
                            text=c.get("content", ""),
                            metadata=c.get("metadata", {}),
                            id_=c.get("id", ""),
                        ),
                        score=c.get("score", 1.0),
                    )
                    for c in event.chunks
                ]

            self._rag_replay_patches.append(
                ("llamaindex_retriever", original_retrieve, BaseRetriever, "retrieve")
            )
            BaseRetriever.retrieve = patched_retrieve  # type: ignore[assignment]
        except ImportError:
            pass

        # ChromaDB Collection.query replay
        try:
            from chromadb import Collection

            original_query = Collection.query

            def patched_chroma_query(self_inner: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
                event = replayer._get_next_retrieval()
                if event is None or not event.chunks:
                    return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
                return {
                    "documents": [[c.get("content", "") for c in event.chunks]],
                    "metadatas": [[c.get("metadata", {}) for c in event.chunks]],
                    "distances": [[1.0 - c.get("score", 1.0) for c in event.chunks]],
                    "ids": [[c.get("id", "") for c in event.chunks]],
                }

            self._rag_replay_patches.append(
                ("chromadb_collection", original_query, Collection, "query")
            )
            Collection.query = patched_chroma_query  # type: ignore[assignment]
        except ImportError:
            pass

    def _install_mcp_replay_patches(self) -> None:
        """Patch MCP ClientSession.call_tool to return recorded results."""
        replayer = self
        try:
            from mcp import ClientSession

            original_call_tool = ClientSession.call_tool

            async def patched_call_tool(
                self_inner: Any, name: str, arguments: dict[str, Any] | None = None
            ) -> Any:
                event = replayer._get_next_mcp_result(name)
                if event is None:
                    if replayer.allow_new_calls:
                        return await original_call_tool(self_inner, name, arguments)
                    from types import SimpleNamespace
                    return SimpleNamespace(content=[], isError=False)
                from types import SimpleNamespace
                return SimpleNamespace(
                    content=event.result if event.result is not None else [],
                    isError=bool(event.is_error),
                )

            self._mcp_replay_patches.append(
                ("mcp_client_session", original_call_tool, ClientSession, "call_tool")
            )
            ClientSession.call_tool = patched_call_tool  # type: ignore[assignment]
        except ImportError:
            pass

    # ── Patching Logic ──

    def _install_patches(self) -> None:
        """Install replay interceptors on LLM SDK methods."""
        if self._intercept_openai:
            self._try_patch_openai()
            self._try_patch_openai_async()
        if self._intercept_anthropic:
            self._try_patch_anthropic()
            self._try_patch_anthropic_async()
        if self._intercept_litellm:
            self._try_patch_litellm()
            self._try_patch_litellm_async()
        if self._intercept_langchain:
            self._try_patch_langchain()
        if self._intercept_langgraph:
            self._try_patch_langgraph()
        if self._intercept_crewai:
            self._try_patch_crewai()
        if self._intercept_rag:
            self._install_rag_replay_patches()
        if self._intercept_mcp:
            self._install_mcp_replay_patches()

    def _remove_patches(self) -> None:
        """Remove all interceptors and restore original methods."""
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        # Restore RAG replay patches
        for _name, original, cls, attr in reversed(self._rag_replay_patches):
            setattr(cls, attr, original)
        self._rag_replay_patches.clear()
        # Restore MCP replay patches
        for _name, original, cls, attr in reversed(self._mcp_replay_patches):
            setattr(cls, attr, original)
        self._mcp_replay_patches.clear()

    def _try_patch_openai(self) -> None:
        """Patch OpenAI to return recorded responses (sync)."""
        try:
            from openai.resources.chat.completions import Completions

            from trace_ops.recorder import _Patch

            original = Completions.create
            replayer = self

            def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                model = kwargs.get("model", "unknown")
                response_dict = replayer._get_next_response("openai", model)

                if not response_dict and replayer.allow_new_calls:
                    return original(self_inner, *args, **kwargs)

                if is_streaming:
                    from trace_ops.streaming import StreamReplay
                    return StreamReplay(response_dict, "openai")

                return _dict_to_openai_response(response_dict)

            self._patches.append(_Patch(Completions, "create", original, patched_create))
            Completions.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_openai_async(self) -> None:
        """Patch OpenAI async to return recorded responses."""
        try:
            from openai.resources.chat.completions import AsyncCompletions

            from trace_ops.recorder import _Patch

            original = AsyncCompletions.create
            replayer = self

            async def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                model = kwargs.get("model", "unknown")
                response_dict = replayer._get_next_response("openai", model)

                if not response_dict and replayer.allow_new_calls:
                    return await original(self_inner, *args, **kwargs)

                if is_streaming:
                    from trace_ops.streaming import AsyncStreamReplay
                    return AsyncStreamReplay(response_dict, "openai")

                return _dict_to_openai_response(response_dict)

            self._patches.append(_Patch(AsyncCompletions, "create", original, patched_create))
            AsyncCompletions.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_anthropic(self) -> None:
        """Patch Anthropic to return recorded responses (sync)."""
        try:
            from anthropic.resources.messages import Messages

            from trace_ops.recorder import _Patch

            original = Messages.create
            replayer = self

            def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                model = kwargs.get("model", "unknown")
                response_dict = replayer._get_next_response("anthropic", model)

                if not response_dict and replayer.allow_new_calls:
                    return original(self_inner, *args, **kwargs)

                if is_streaming:
                    from trace_ops.streaming import StreamReplay
                    return StreamReplay(response_dict, "anthropic")

                return _dict_to_anthropic_response(response_dict)

            self._patches.append(_Patch(Messages, "create", original, patched_create))
            Messages.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_anthropic_async(self) -> None:
        """Patch Anthropic async to return recorded responses."""
        try:
            from anthropic.resources.messages import AsyncMessages

            from trace_ops.recorder import _Patch

            original = AsyncMessages.create
            replayer = self

            async def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                model = kwargs.get("model", "unknown")
                response_dict = replayer._get_next_response("anthropic", model)

                if not response_dict and replayer.allow_new_calls:
                    return await original(self_inner, *args, **kwargs)

                if is_streaming:
                    from trace_ops.streaming import AsyncStreamReplay
                    return AsyncStreamReplay(response_dict, "anthropic")

                return _dict_to_anthropic_response(response_dict)

            self._patches.append(_Patch(AsyncMessages, "create", original, patched_create))
            AsyncMessages.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_litellm(self) -> None:
        """Patch litellm to return recorded responses (sync)."""
        try:
            import litellm

            from trace_ops.recorder import _Patch

            original = litellm.completion
            replayer = self

            def patched_completion(*args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                model = str(kwargs.get("model", args[0] if args else "unknown"))
                response_dict = replayer._get_next_response("litellm", model)

                if not response_dict and replayer.allow_new_calls:
                    return original(*args, **kwargs)

                if is_streaming:
                    from trace_ops.streaming import StreamReplay
                    return StreamReplay(response_dict, "litellm")

                return _dict_to_openai_response(response_dict)

            self._patches.append(_Patch(litellm, "completion", original, patched_completion))
            litellm.completion = patched_completion  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_litellm_async(self) -> None:
        """Patch litellm.acompletion to return recorded responses."""
        try:
            import litellm

            from trace_ops.recorder import _Patch

            original = litellm.acompletion
            replayer = self

            async def patched_acompletion(*args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                model = str(kwargs.get("model", args[0] if args else "unknown"))
                response_dict = replayer._get_next_response("litellm", model)

                if not response_dict and replayer.allow_new_calls:
                    return await original(*args, **kwargs)

                if is_streaming:
                    from trace_ops.streaming import AsyncStreamReplay
                    return AsyncStreamReplay(response_dict, "litellm")

                return _dict_to_openai_response(response_dict)

            self._patches.append(_Patch(litellm, "acompletion", original, patched_acompletion))
            litellm.acompletion = patched_acompletion  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_langchain(self) -> None:
        """Patch LangChain to return recorded responses."""
        try:
            from trace_ops.interceptors.langchain import install_langchain_replay_patches
            install_langchain_replay_patches(self, self._patches)
        except ImportError:
            pass

    def _try_patch_langgraph(self) -> None:
        """Patch LangGraph for replay (no-op — LLM calls handled by LangChain interceptor)."""
        try:
            from trace_ops.interceptors.langgraph import install_langgraph_replay_patches
            install_langgraph_replay_patches(self, self._patches)
        except ImportError:
            pass

    def _try_patch_crewai(self) -> None:
        """Patch CrewAI for replay (primarily delegates to underlying LLM patches)."""
        try:
            from trace_ops.interceptors.crewai import install_crewai_replay_patches
            install_crewai_replay_patches(self, self._patches)
        except ImportError:
            pass

    # ── Context Manager ──

    def __enter__(self) -> Replayer:
        self._load()
        self._install_patches()
        return self

    def __exit__(self, *args: Any) -> None:
        self._remove_patches()

        # Check if all recorded responses were consumed
        if self.strict and self._call_index < len(self._response_queue):
            unconsumed = len(self._response_queue) - self._call_index
            raise CassetteMismatchError(
                f"Agent made fewer LLM calls than recorded.\n"
                f"{unconsumed} recorded responses were never consumed.\n"
                f"Expected {len(self._response_queue)} calls, "
                f"got {self._call_index}.\n"
                f"The agent's behavior has changed — re-record the cassette."
            )

    async def __aenter__(self) -> Replayer:
        """Async context manager entry — loads cassette and installs patches."""
        self._load()
        self._install_patches()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit — removes patches and validates."""
        self._remove_patches()

        if self.strict and self._call_index < len(self._response_queue):
            unconsumed = len(self._response_queue) - self._call_index
            raise CassetteMismatchError(
                f"Agent made fewer LLM calls than recorded.\n"
                f"{unconsumed} recorded responses were never consumed.\n"
                f"Expected {len(self._response_queue)} calls, "
                f"got {self._call_index}.\n"
                f"The agent's behavior has changed — re-record the cassette."
            )

    # ── Decorator ──

    @staticmethod
    def replay(cassette_path: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator that replays a cassette for the decorated function."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            import functools

            @functools.wraps(func)
            def wrapper(*args: Any, **fn_kwargs: Any) -> Any:
                with Replayer(cassette_path, **kwargs):
                    return func(*args, **fn_kwargs)

            return wrapper
        return decorator


def _dict_to_openai_response(data: dict[str, Any]) -> Any:
    """Convert a dict back to an OpenAI-like response object.

    We use a SimpleNamespace to create a duck-typed object that
    behaves like an OpenAI response for attribute access.
    """
    from types import SimpleNamespace

    def _to_ns(d: Any) -> Any:
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_to_ns(item) for item in d]
        return d

    return _to_ns(data)


def _dict_to_anthropic_response(data: dict[str, Any]) -> Any:
    """Convert a dict back to an Anthropic-like response object."""
    from types import SimpleNamespace

    def _to_ns(d: Any) -> Any:
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_to_ns(item) for item in d]
        return d

    return _to_ns(data)
