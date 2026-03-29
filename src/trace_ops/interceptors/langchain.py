"""LangChain / LangGraph interceptor for traceops.

Patches ``BaseChatModel.invoke()`` / ``ainvoke()`` and
``BaseTool.invoke()`` / ``ainvoke()`` so that LangChain-based agents
are automatically recorded and replayed.

This module is **optional** — it only activates when
``langchain-core`` is installed.
"""

from __future__ import annotations

import time
from typing import Any


def install_langchain_record_patches(
    recorder: Any,
    patches: list[Any],
) -> None:
    """Install LangChain recording interceptors.

    Args:
        recorder: The :class:`~trace_ops.recorder.Recorder` instance.
        patches: The recorder's ``_patches`` list to append to.
    """
    _patch_chat_model_invoke(recorder, patches)
    _patch_chat_model_ainvoke(recorder, patches)
    _patch_base_tool_invoke(recorder, patches)
    _patch_base_tool_ainvoke(recorder, patches)


def install_langchain_replay_patches(
    replayer: Any,
    patches: list[Any],
) -> None:
    """Install LangChain replay interceptors.

    Args:
        replayer: The :class:`~trace_ops.replayer.Replayer` instance.
        patches: The replayer's ``_patches`` list to append to.
    """
    _patch_chat_model_invoke_replay(replayer, patches)
    _patch_chat_model_ainvoke_replay(replayer, patches)


# ── Recording patches ───────────────────────────────────────────────


def _patch_chat_model_invoke(recorder: Any, patches: list[Any]) -> None:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch, _response_to_dict, _safe_serialize

        original = BaseChatModel.invoke
        rec = recorder

        def patched_invoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            model_name = getattr(self_inner, "model_name", None) or type(self_inner).__name__

            rec._trace.add_event(TraceEvent(
                event_type=EventType.LLM_REQUEST,
                provider="langchain",
                model=model_name,
                messages=_safe_serialize(input),
                metadata={"framework": "langchain"},
            ))

            start = time.monotonic()
            try:
                response = original(self_inner, input, config, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langchain",
                    model=model_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            resp_dict = _response_to_dict(response)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                provider="langchain",
                model=model_name,
                response=resp_dict,
                duration_ms=elapsed,
                metadata={"framework": "langchain"},
            ))

            # Record tool calls from AIMessage
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    rec._trace.add_event(TraceEvent(
                        event_type=EventType.TOOL_CALL,
                        tool_name=tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                        tool_input=tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                        metadata={"framework": "langchain"},
                    ))

            return response

        patches.append(_Patch(BaseChatModel, "invoke", original, patched_invoke))
        BaseChatModel.invoke = patched_invoke  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_chat_model_ainvoke(recorder: Any, patches: list[Any]) -> None:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch, _response_to_dict, _safe_serialize

        original = BaseChatModel.ainvoke
        rec = recorder

        async def patched_ainvoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            model_name = getattr(self_inner, "model_name", None) or type(self_inner).__name__

            rec._trace.add_event(TraceEvent(
                event_type=EventType.LLM_REQUEST,
                provider="langchain",
                model=model_name,
                messages=_safe_serialize(input),
                metadata={"framework": "langchain"},
            ))

            start = time.monotonic()
            try:
                response = await original(self_inner, input, config, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langchain",
                    model=model_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            resp_dict = _response_to_dict(response)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                provider="langchain",
                model=model_name,
                response=resp_dict,
                duration_ms=elapsed,
                metadata={"framework": "langchain"},
            ))

            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    rec._trace.add_event(TraceEvent(
                        event_type=EventType.TOOL_CALL,
                        tool_name=tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                        tool_input=tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                        metadata={"framework": "langchain"},
                    ))

            return response

        patches.append(_Patch(BaseChatModel, "ainvoke", original, patched_ainvoke))
        BaseChatModel.ainvoke = patched_ainvoke  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_base_tool_invoke(recorder: Any, patches: list[Any]) -> None:
    try:
        from langchain_core.tools import BaseTool

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch, _safe_serialize

        original = BaseTool.invoke
        rec = recorder

        def patched_invoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            tool_name = getattr(self_inner, "name", type(self_inner).__name__)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL,
                tool_name=tool_name,
                tool_input=_safe_serialize(input),
                metadata={"framework": "langchain"},
            ))

            start = time.monotonic()
            try:
                result = original(self_inner, input, config, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langchain",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    tool_name=tool_name,
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            rec._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_RESULT,
                tool_name=tool_name,
                tool_output=_safe_serialize(result),
                duration_ms=elapsed,
                metadata={"framework": "langchain"},
            ))
            return result

        patches.append(_Patch(BaseTool, "invoke", original, patched_invoke))
        BaseTool.invoke = patched_invoke  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_base_tool_ainvoke(recorder: Any, patches: list[Any]) -> None:
    try:
        from langchain_core.tools import BaseTool

        from trace_ops._types import EventType, TraceEvent
        from trace_ops.recorder import _Patch, _safe_serialize

        original = BaseTool.ainvoke
        rec = recorder

        async def patched_ainvoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            tool_name = getattr(self_inner, "name", type(self_inner).__name__)

            rec._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_CALL,
                tool_name=tool_name,
                tool_input=_safe_serialize(input),
                metadata={"framework": "langchain"},
            ))

            start = time.monotonic()
            try:
                result = await original(self_inner, input, config, **kwargs)
            except Exception as exc:
                rec._trace.add_event(TraceEvent(
                    event_type=EventType.ERROR,
                    provider="langchain",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    tool_name=tool_name,
                ))
                raise

            elapsed = (time.monotonic() - start) * 1000
            rec._trace.add_event(TraceEvent(
                event_type=EventType.TOOL_RESULT,
                tool_name=tool_name,
                tool_output=_safe_serialize(result),
                duration_ms=elapsed,
                metadata={"framework": "langchain"},
            ))
            return result

        patches.append(_Patch(BaseTool, "ainvoke", original, patched_ainvoke))
        BaseTool.ainvoke = patched_ainvoke  # type: ignore[assignment]

    except ImportError:
        pass


# ── Replay patches ──────────────────────────────────────────────────


def _patch_chat_model_invoke_replay(replayer: Any, patches: list[Any]) -> None:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        from trace_ops.recorder import _Patch

        original = BaseChatModel.invoke
        rep = replayer

        def patched_invoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            model_name = getattr(self_inner, "model_name", None) or type(self_inner).__name__
            response_dict = rep._get_next_response("langchain", model_name)

            if not response_dict and rep.allow_new_calls:
                return original(self_inner, input, config, **kwargs)

            return _dict_to_langchain_response(response_dict)

        patches.append(_Patch(BaseChatModel, "invoke", original, patched_invoke))
        BaseChatModel.invoke = patched_invoke  # type: ignore[assignment]

    except ImportError:
        pass


def _patch_chat_model_ainvoke_replay(replayer: Any, patches: list[Any]) -> None:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        from trace_ops.recorder import _Patch

        original = BaseChatModel.ainvoke
        rep = replayer

        async def patched_ainvoke(self_inner: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            model_name = getattr(self_inner, "model_name", None) or type(self_inner).__name__
            response_dict = rep._get_next_response("langchain", model_name)

            if not response_dict and rep.allow_new_calls:
                return await original(self_inner, input, config, **kwargs)

            return _dict_to_langchain_response(response_dict)

        patches.append(_Patch(BaseChatModel, "ainvoke", original, patched_ainvoke))
        BaseChatModel.ainvoke = patched_ainvoke  # type: ignore[assignment]

    except ImportError:
        pass


# ── Helpers ─────────────────────────────────────────────────────────


def _dict_to_langchain_response(data: dict[str, Any]) -> Any:
    """Convert a response dict back into a LangChain-compatible AIMessage.

    Attempts to use the real ``AIMessage`` class so downstream code
    that type-checks the response still works.  Falls back to
    SimpleNamespace if ``langchain-core`` isn't available.
    """
    try:
        from langchain_core.messages import AIMessage

        content = data.get("content", "")
        # Handle nested message structures from model_dump
        if isinstance(content, dict):
            content = content.get("content", str(content))

        tool_calls = []
        raw_tcs = data.get("tool_calls", [])
        for tc in raw_tcs:
            tool_calls.append({
                "name": tc.get("name", ""),
                "args": tc.get("args", tc.get("arguments", {})),
                "id": tc.get("id", ""),
            })

        return AIMessage(
            content=content if isinstance(content, str) else str(content),
            tool_calls=tool_calls if tool_calls else [],
        )
    except Exception:
        from types import SimpleNamespace

        def _to_ns(d: Any) -> Any:
            if isinstance(d, dict):
                return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
            if isinstance(d, list):
                return [_to_ns(i) for i in d]
            return d

        return _to_ns(data)
