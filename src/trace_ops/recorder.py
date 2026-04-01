"""Recorder — intercepts LLM and tool calls to record agent execution traces.

The recorder works by monkey-patching SDK client methods at runtime.
When you enter a recording context, it wraps openai.chat.completions.create,
anthropic.messages.create, etc. with interceptors that capture the full
request/response cycle. When you exit the context, the patches are removed.

This is the same approach used by VCR.py for HTTP, but operating at the
SDK level so we get semantic understanding of what happened.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from trace_ops._types import EventType, Trace, TraceEvent, TraceMetadata
from trace_ops.cassette import save_cassette


class Recorder:
    """Records agent execution traces by intercepting LLM SDK calls.

    Usage as context manager:
        with Recorder() as recorder:
            # Run your agent code here — all LLM calls are recorded
            agent.run("What is 2+2?")

        trace = recorder.trace
        save_cassette(trace, "cassettes/test_math.yaml")

    Usage as decorator:
        @Recorder.record("cassettes/test_math.yaml")
        def test_my_agent():
            agent.run("What is 2+2?")
    """

    def __init__(
        self,
        *,
        description: str = "",
        tags: list[str] | None = None,
        save_to: str | None = None,
        intercept_openai: bool = True,
        intercept_anthropic: bool = True,
        intercept_litellm: bool = True,
        intercept_langchain: bool = True,
        intercept_langgraph: bool = True,
        intercept_crewai: bool = True,
        # RAG options
        intercept_rag: bool = False,
        rag_scorer: Any | None = None,
        # MCP options
        intercept_mcp: bool = False,
    ) -> None:
        self.save_to = save_to
        self._trace = Trace(
            metadata=TraceMetadata(
                description=description,
                tags=tags or [],
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            )
        )
        self._patches: list[_Patch] = []
        self._intercept_openai = intercept_openai
        self._intercept_anthropic = intercept_anthropic
        self._intercept_litellm = intercept_litellm
        self._intercept_langchain = intercept_langchain
        self._intercept_langgraph = intercept_langgraph
        self._intercept_crewai = intercept_crewai
        self._intercept_rag = intercept_rag
        self._rag_scorer = rag_scorer
        self._intercept_mcp = intercept_mcp
        # Storage for RAG / MCP monkey-patch metadata: (name, original, cls, attr)
        self._rag_patches: list[tuple[str, Any, Any, str]] = []
        self._mcp_patches: list[tuple[str, Any, Any, str]] = []

    @property
    def trace(self) -> Trace:
        """Get the recorded trace."""
        return self._trace

    def add_event(self, event: TraceEvent) -> None:
        """Manually add an event to the trace (for custom tool calls)."""
        self._trace.add_event(event)

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        duration_ms: float | None = None,
    ) -> None:
        """Convenience method to record a tool call + result pair."""
        self._trace.add_event(TraceEvent(
            event_type=EventType.TOOL_CALL,
            tool_name=tool_name,
            tool_input=tool_input,
        ))
        self._trace.add_event(TraceEvent(
            event_type=EventType.TOOL_RESULT,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
        ))

    def record_decision(self, decision: str, reasoning: str | None = None) -> None:
        """Record an agent-level decision."""
        self._trace.add_event(TraceEvent(
            event_type=EventType.AGENT_DECISION,
            decision=decision,
            reasoning=reasoning,
        ))

    # ── Patching Logic ──

    def _install_patches(self) -> None:
        """Install interceptors on LLM SDK methods."""
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
            self._install_rag_patches()
        if self._intercept_mcp:
            self._install_mcp_patches()

    def _remove_patches(self) -> None:
        """Remove all interceptors and restore original methods."""
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        # Restore RAG patches
        for _name, original, cls, attr in reversed(self._rag_patches):
            with contextlib.suppress(Exception):
                setattr(cls, attr, original)
        self._rag_patches.clear()
        # Restore MCP patches
        for _name, original, cls, attr in reversed(self._mcp_patches):
            with contextlib.suppress(Exception):
                setattr(cls, attr, original)
        self._mcp_patches.clear()

    def _install_rag_patches(self) -> None:
        """Install RAG vector-store interceptors."""
        from trace_ops.rag.interceptors import (
            patch_chromadb,
            patch_langchain_retriever,
            patch_llamaindex,
            patch_openai_embeddings,
            patch_pinecone,
            patch_qdrant,
        )
        for fn in (
            patch_chromadb,
            patch_langchain_retriever,
            patch_llamaindex,
            patch_pinecone,
            patch_qdrant,
            patch_openai_embeddings,
        ):
            with contextlib.suppress(Exception):
                fn(self)

    def _install_mcp_patches(self) -> None:
        """Install MCP ClientSession interceptor."""
        from trace_ops.mcp.interceptor import patch_mcp
        with contextlib.suppress(Exception):
            patch_mcp(self)

    def record_retrieval(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        vector_store: str = "custom",
        collection: str = "",
        top_k: int = 0,
        embedding_model: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """Manually record a retrieval event.

        Use this when you have a custom retriever that isn't auto-patched.

        Args:
            query: The search query text.
            retrieved_chunks: List of dicts with ``id``, ``text``, ``score``,
                and optionally ``metadata`` keys.
            vector_store: Name of the vector store (e.g. ``"chromadb"``).
            collection: Collection / index name.
            top_k: Number of results requested.
            embedding_model: Embedding model used for the query (if known).
            duration_ms: Retrieval latency in milliseconds.
        """
        self._trace.add_event(TraceEvent(
            event_type=EventType.RETRIEVAL,
            query=query,
            chunks=retrieved_chunks,
            vector_store=vector_store,
            collection=collection or None,
            top_k=top_k or None,
            duration_ms=duration_ms or None,
            metadata={"embedding_model": embedding_model} if embedding_model else {},
        ))

    def _record_retrieval_event(self, event: Any) -> None:
        """Called by RAG interceptors to add a retrieval event to the trace."""
        self._trace.add_event(TraceEvent(
            event_type=EventType.RETRIEVAL,
            query=event.query,
            chunks=[c.to_dict() if hasattr(c, "to_dict") else c for c in (event.chunks or [])],
            vector_store=event.vector_store,
            collection=event.collection or None,
            top_k=event.top_k or None,
            duration_ms=event.duration_ms or None,
        ))

    def _run_rag_scorer(self) -> None:
        """Run the configured RAG scorer and cache scores in the trace."""
        if self._rag_scorer is None:
            return

        retrieval_events = self._trace.retrieval_events
        if not retrieval_events:
            return

        # Get the last LLM response text
        from trace_ops._types import EventType
        llm_responses = [e for e in self._trace.events if e.event_type == EventType.LLM_RESPONSE]
        if not llm_responses:
            return

        resp = llm_responses[-1].response or {}
        choices = resp.get("choices") or []
        response_text = ""
        if choices:
            response_text = (choices[0].get("message") or {}).get("content", "") or ""

        # Use the first retrieval for scoring
        retrieval = retrieval_events[0]
        context_chunks = [
            (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
            for c in (retrieval.chunks or [])
        ]

        try:
            result = self._rag_scorer.score(
                query=retrieval.query or "",
                context_chunks=context_chunks,
                response=response_text,
            )
            self._trace.add_event(TraceEvent(
                event_type=EventType.RAG_SCORES,
                scores=result.scores,
                metadata={
                    "scorer": result.scorer,
                    "judge_model": result.judge_model,
                    "judge_tokens": result.judge_tokens,
                    "judge_cost_usd": result.judge_cost_usd,
                    "judge_duration_ms": result.judge_duration_ms,
                },
            ))
        except Exception as exc:
            import warnings
            warnings.warn(f"RAG scorer failed: {exc}", stacklevel=2)

    def _try_patch_openai(self) -> None:
        """Attempt to patch the OpenAI SDK (sync)."""
        try:
            from openai.resources.chat.completions import Completions

            original = Completions.create
            recorder = self

            def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                messages = kwargs.get("messages", args[0] if args else [])
                model = kwargs.get("model", "unknown")
                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="openai",
                    model=model,
                    messages=_safe_serialize(messages),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    tools=_safe_serialize(kwargs.get("tools")),
                    metadata={"stream": True} if is_streaming else {},
                ))

                start = time.monotonic()
                try:
                    response = original(self_inner, *args, **kwargs)
                except Exception as exc:
                    recorder._trace.add_event(TraceEvent(
                        event_type=EventType.ERROR,
                        provider="openai",
                        model=model,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ))
                    raise

                if is_streaming:
                    from trace_ops.streaming import StreamCapture

                    def on_complete(assembled: dict[str, Any]) -> None:
                        elapsed = (time.monotonic() - start) * 1000
                        usage = assembled.get("usage", {})
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.LLM_RESPONSE,
                            provider="openai",
                            model=model,
                            response=assembled,
                            duration_ms=elapsed,
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"),
                            metadata={"streamed": True},
                        ))
                        _record_tool_calls_from_response(recorder, assembled, "openai")

                    return StreamCapture(response, "openai", on_complete)

                elapsed = (time.monotonic() - start) * 1000

                # Extract usage info
                resp_dict = _response_to_dict(response)
                usage = getattr(response, "usage", None)

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    provider="openai",
                    model=model,
                    response=resp_dict,
                    duration_ms=elapsed,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                ))

                # Record any tool calls in the response
                _record_tool_calls_from_openai(recorder, response)

                return response

            self._patches.append(_Patch(Completions, "create", original, patched_create))
            Completions.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_openai_async(self) -> None:
        """Patch the OpenAI async SDK."""
        try:
            from openai.resources.chat.completions import AsyncCompletions

            original = AsyncCompletions.create
            recorder = self

            async def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                messages = kwargs.get("messages", args[0] if args else [])
                model = kwargs.get("model", "unknown")
                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="openai",
                    model=model,
                    messages=_safe_serialize(messages),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    tools=_safe_serialize(kwargs.get("tools")),
                    metadata={"stream": True} if is_streaming else {},
                ))

                start = time.monotonic()
                try:
                    response = await original(self_inner, *args, **kwargs)
                except Exception as exc:
                    recorder._trace.add_event(TraceEvent(
                        event_type=EventType.ERROR,
                        provider="openai",
                        model=model,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ))
                    raise

                if is_streaming:
                    from trace_ops.streaming import AsyncStreamCapture

                    def on_complete(assembled: dict[str, Any]) -> None:
                        elapsed = (time.monotonic() - start) * 1000
                        usage = assembled.get("usage", {})
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.LLM_RESPONSE,
                            provider="openai",
                            model=model,
                            response=assembled,
                            duration_ms=elapsed,
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"),
                            metadata={"streamed": True},
                        ))
                        _record_tool_calls_from_response(recorder, assembled, "openai")

                    return AsyncStreamCapture(response, "openai", on_complete)

                elapsed = (time.monotonic() - start) * 1000
                resp_dict = _response_to_dict(response)
                usage = getattr(response, "usage", None)

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    provider="openai",
                    model=model,
                    response=resp_dict,
                    duration_ms=elapsed,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                ))
                _record_tool_calls_from_openai(recorder, response)

                return response

            self._patches.append(_Patch(AsyncCompletions, "create", original, patched_create))
            AsyncCompletions.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_anthropic(self) -> None:
        """Attempt to patch the Anthropic SDK (sync)."""
        try:
            from anthropic.resources.messages import Messages

            original = Messages.create
            recorder = self

            def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                messages = kwargs.get("messages", [])
                model = kwargs.get("model", "unknown")
                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="anthropic",
                    model=model,
                    messages=_safe_serialize(messages),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    tools=_safe_serialize(kwargs.get("tools")),
                    metadata={"stream": True} if is_streaming else {},
                ))

                start = time.monotonic()
                try:
                    response = original(self_inner, *args, **kwargs)
                except Exception as exc:
                    recorder._trace.add_event(TraceEvent(
                        event_type=EventType.ERROR,
                        provider="anthropic",
                        model=model,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ))
                    raise

                if is_streaming:
                    from trace_ops.streaming import StreamCapture

                    def on_complete(assembled: dict[str, Any]) -> None:
                        elapsed = (time.monotonic() - start) * 1000
                        usage = assembled.get("usage", {})
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.LLM_RESPONSE,
                            provider="anthropic",
                            model=model,
                            response=assembled,
                            duration_ms=elapsed,
                            input_tokens=usage.get("input_tokens"),
                            output_tokens=usage.get("output_tokens"),
                            metadata={"streamed": True},
                        ))
                        _record_tool_calls_from_anthropic_dict(recorder, assembled)

                    return StreamCapture(response, "anthropic", on_complete)

                elapsed = (time.monotonic() - start) * 1000
                resp_dict = _response_to_dict(response)
                usage = getattr(response, "usage", None)

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    provider="anthropic",
                    model=model,
                    response=resp_dict,
                    duration_ms=elapsed,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                ))

                # Record tool use blocks
                content = getattr(response, "content", [])
                for block in content:
                    if getattr(block, "type", "") == "tool_use":
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.TOOL_CALL,
                            tool_name=getattr(block, "name", "unknown"),
                            tool_input=getattr(block, "input", {}),
                            metadata={"tool_use_id": getattr(block, "id", "")},
                        ))

                return response

            self._patches.append(_Patch(Messages, "create", original, patched_create))
            Messages.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_anthropic_async(self) -> None:
        """Patch the Anthropic async SDK."""
        try:
            from anthropic.resources.messages import AsyncMessages

            original = AsyncMessages.create
            recorder = self

            async def patched_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                messages = kwargs.get("messages", [])
                model = kwargs.get("model", "unknown")
                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="anthropic",
                    model=model,
                    messages=_safe_serialize(messages),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    tools=_safe_serialize(kwargs.get("tools")),
                    metadata={"stream": True} if is_streaming else {},
                ))

                start = time.monotonic()
                try:
                    response = await original(self_inner, *args, **kwargs)
                except Exception as exc:
                    recorder._trace.add_event(TraceEvent(
                        event_type=EventType.ERROR,
                        provider="anthropic",
                        model=model,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ))
                    raise

                if is_streaming:
                    from trace_ops.streaming import AsyncStreamCapture

                    def on_complete(assembled: dict[str, Any]) -> None:
                        elapsed = (time.monotonic() - start) * 1000
                        usage = assembled.get("usage", {})
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.LLM_RESPONSE,
                            provider="anthropic",
                            model=model,
                            response=assembled,
                            duration_ms=elapsed,
                            input_tokens=usage.get("input_tokens"),
                            output_tokens=usage.get("output_tokens"),
                            metadata={"streamed": True},
                        ))
                        _record_tool_calls_from_anthropic_dict(recorder, assembled)

                    return AsyncStreamCapture(response, "anthropic", on_complete)

                elapsed = (time.monotonic() - start) * 1000
                resp_dict = _response_to_dict(response)
                usage = getattr(response, "usage", None)

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    provider="anthropic",
                    model=model,
                    response=resp_dict,
                    duration_ms=elapsed,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                ))

                content = getattr(response, "content", [])
                for block in content:
                    if getattr(block, "type", "") == "tool_use":
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.TOOL_CALL,
                            tool_name=getattr(block, "name", "unknown"),
                            tool_input=getattr(block, "input", {}),
                            metadata={"tool_use_id": getattr(block, "id", "")},
                        ))

                return response

            self._patches.append(_Patch(AsyncMessages, "create", original, patched_create))
            AsyncMessages.create = patched_create  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_litellm(self) -> None:
        """Attempt to patch litellm.completion (sync)."""
        try:
            import litellm

            original = litellm.completion
            recorder = self

            def patched_completion(*args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                model = kwargs.get("model", args[0] if args else "unknown")

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="litellm",
                    model=str(model),
                    messages=_safe_serialize(messages),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    tools=_safe_serialize(kwargs.get("tools")),
                    metadata={"stream": True} if is_streaming else {},
                ))

                start = time.monotonic()
                try:
                    response = original(*args, **kwargs)
                except Exception as exc:
                    recorder._trace.add_event(TraceEvent(
                        event_type=EventType.ERROR,
                        provider="litellm",
                        model=str(model),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ))
                    raise

                if is_streaming:
                    from trace_ops.streaming import StreamCapture

                    def on_complete(assembled: dict[str, Any]) -> None:
                        elapsed = (time.monotonic() - start) * 1000
                        usage = assembled.get("usage", {})
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.LLM_RESPONSE,
                            provider="litellm",
                            model=str(model),
                            response=assembled,
                            duration_ms=elapsed,
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"),
                            metadata={"streamed": True},
                        ))

                    return StreamCapture(response, "litellm", on_complete)

                elapsed = (time.monotonic() - start) * 1000
                resp_dict = _response_to_dict(response)
                usage = getattr(response, "usage", None)

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    provider="litellm",
                    model=str(model),
                    response=resp_dict,
                    duration_ms=elapsed,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                ))

                return response

            self._patches.append(_Patch(litellm, "completion", original, patched_completion))
            litellm.completion = patched_completion  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_litellm_async(self) -> None:
        """Patch litellm.acompletion (async)."""
        try:
            import litellm

            original = litellm.acompletion
            recorder = self

            async def patched_acompletion(*args: Any, **kwargs: Any) -> Any:
                is_streaming = kwargs.get("stream", False)
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                model = kwargs.get("model", args[0] if args else "unknown")

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_REQUEST,
                    provider="litellm",
                    model=str(model),
                    messages=_safe_serialize(messages),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    tools=_safe_serialize(kwargs.get("tools")),
                    metadata={"stream": True} if is_streaming else {},
                ))

                start = time.monotonic()
                try:
                    response = await original(*args, **kwargs)
                except Exception as exc:
                    recorder._trace.add_event(TraceEvent(
                        event_type=EventType.ERROR,
                        provider="litellm",
                        model=str(model),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ))
                    raise

                if is_streaming:
                    from trace_ops.streaming import AsyncStreamCapture

                    def on_complete(assembled: dict[str, Any]) -> None:
                        elapsed = (time.monotonic() - start) * 1000
                        usage = assembled.get("usage", {})
                        recorder._trace.add_event(TraceEvent(
                            event_type=EventType.LLM_RESPONSE,
                            provider="litellm",
                            model=str(model),
                            response=assembled,
                            duration_ms=elapsed,
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"),
                            metadata={"streamed": True},
                        ))

                    return AsyncStreamCapture(response, "litellm", on_complete)

                elapsed = (time.monotonic() - start) * 1000
                resp_dict = _response_to_dict(response)
                usage = getattr(response, "usage", None)

                recorder._trace.add_event(TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    provider="litellm",
                    model=str(model),
                    response=resp_dict,
                    duration_ms=elapsed,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                ))

                return response

            self._patches.append(_Patch(litellm, "acompletion", original, patched_acompletion))
            litellm.acompletion = patched_acompletion  # type: ignore[assignment]

        except ImportError:
            pass

    def _try_patch_langchain(self) -> None:
        """Attempt to patch LangChain SDK."""
        try:
            from trace_ops.interceptors.langchain import install_langchain_record_patches

            install_langchain_record_patches(self, self._patches)
        except ImportError:
            pass

    def _try_patch_langgraph(self) -> None:
        """Attempt to patch LangGraph (Pregel) SDK."""
        try:
            from trace_ops.interceptors.langgraph import install_langgraph_record_patches

            install_langgraph_record_patches(self, self._patches)
        except ImportError:
            pass

    def _try_patch_crewai(self) -> None:
        """Attempt to patch CrewAI SDK."""
        try:
            from trace_ops.interceptors.crewai import install_crewai_record_patches

            install_crewai_record_patches(self, self._patches)
        except ImportError:
            pass

    # ── Context Manager ──

    def __enter__(self) -> Recorder:
        self._install_patches()
        return self

    def __exit__(self, *args: Any) -> None:
        self._remove_patches()
        self._run_rag_scorer()
        self._trace.finalize()

        if self.save_to:
            save_cassette(self._trace, self.save_to)

    async def __aenter__(self) -> Recorder:
        """Async context manager entry — installs the same patches."""
        self._install_patches()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit — removes patches and finalises trace."""
        self._remove_patches()
        self._run_rag_scorer()
        self._trace.finalize()

        if self.save_to:
            save_cassette(self._trace, self.save_to)

    # ── Decorator ──

    @staticmethod
    def record(
        cassette_path: str | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Decorator that records an agent trace for the decorated function.

        Usage:
            @Recorder.record("cassettes/test_math.yaml")
            def test_my_agent():
                agent.run("What is 2+2?")
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            import functools

            @functools.wraps(func)
            def wrapper(*args: Any, **fn_kwargs: Any) -> Any:
                with Recorder(save_to=cassette_path, **kwargs):
                    return func(*args, **fn_kwargs)

            return wrapper
        return decorator


class _Patch:
    """Stores info needed to restore a monkey-patched method."""

    def __init__(self, obj: Any, attr: str, original: Any, replacement: Any) -> None:
        self.obj = obj
        self.attr = attr
        self.original = original
        self.replacement = replacement

    def restore(self) -> None:
        setattr(self.obj, self.attr, self.original)


def _safe_serialize(obj: Any) -> Any:
    """Safely convert SDK objects to JSON-serializable dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Convert an LLM response object to a serializable dict."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return _safe_serialize(response)


def _record_tool_calls_from_openai(recorder: Any, response: Any) -> None:
    """Extract and record tool calls from an OpenAI SDK response object."""
    choices = getattr(response, "choices", [])
    if not choices:
        return
    message = getattr(choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn:
            try:
                tc_args = json.loads(getattr(fn, "arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tc_args = {"_raw": getattr(fn, "arguments", "")}
            recorder._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL,
                tool_name=getattr(fn, "name", "unknown"),
                tool_input=tc_args,
                metadata={"tool_call_id": getattr(tc, "id", "")},
            ))


def _record_tool_calls_from_response(
    recorder: Any, assembled: dict[str, Any], provider: str
) -> None:
    """Record tool calls from an assembled response dict (used after streaming)."""
    if provider in ("openai", "litellm"):
        choices = assembled.get("choices", [])
        if not choices:
            return
        message = choices[0].get("message", {})
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                tc_args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tc_args = {"_raw": fn.get("arguments", "")}
            recorder._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL,
                tool_name=fn.get("name", "unknown"),
                tool_input=tc_args,
                metadata={"tool_call_id": tc.get("id", "")},
            ))
    elif provider == "anthropic":
        _record_tool_calls_from_anthropic_dict(recorder, assembled)


def _record_tool_calls_from_anthropic_dict(
    recorder: Any, assembled: dict[str, Any]
) -> None:
    """Record tool calls from an assembled Anthropic response dict."""
    for block in assembled.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            recorder._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL,
                tool_name=block.get("name", "unknown"),
                tool_input=block.get("input", {}),
                metadata={"tool_use_id": block.get("id", "")},
            ))
