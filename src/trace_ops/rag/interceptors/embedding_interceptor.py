"""OpenAI / Cohere embedding call interceptor."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_ops.recorder import Recorder


def patch_openai_embeddings(recorder: Recorder) -> None:
    """Patch ``openai.resources.Embeddings.create()`` to record embedding events."""
    try:
        from openai.resources.embeddings import Embeddings
    except ImportError:
        return

    original = Embeddings.create

    @functools.wraps(original)
    def patched(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self_inner, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        from trace_ops._types import EventType, TraceEvent

        input_val = kwargs.get("input", args[0] if args else "")
        input_text = (
            input_val[0] if isinstance(input_val, list) and input_val else str(input_val or "")
        )
        model = kwargs.get("model", "unknown")
        usage = getattr(result, "usage", None)
        input_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        dims = 0
        if result and getattr(result, "data", None):
            emb = result.data[0]
            dims = len(getattr(emb, "embedding", []) or [])

        recorder._trace.add_event(TraceEvent(
            event_type=EventType.EMBEDDING_CALL,
            provider="openai",
            model=model,
            query=input_text,
            input_tokens=input_tokens,
            dimensions=dims,
            duration_ms=duration_ms,
        ))
        return result

    Embeddings.create = patched  # type: ignore[method-assign]
    recorder._rag_patches.append(("openai.resources.Embeddings.create", original, Embeddings, "create"))
