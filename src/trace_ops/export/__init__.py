"""Dataset export utilities for traceops.

Export recorded cassettes to multiple formats for fine-tuning and evaluation.

Usage::

    from trace_ops.export import to_openai_finetune, to_anthropic_finetune

    to_openai_finetune("cassettes/", output="training_data.jsonl")
    to_anthropic_finetune("cassettes/", output="training_data_anthropic.jsonl")

For RAG-specific exports (RAGAS, DeepEval, CSV) see ``trace_ops.rag.export``.
"""

from trace_ops.export.finetune import to_anthropic_finetune, to_openai_finetune

__all__ = [
    "to_openai_finetune",
    "to_anthropic_finetune",
]
