"""RAG-aware trace diffing.

Compares two traces at the retrieval and generation levels separately,
enabling precise root-cause analysis: did the retriever drift, or did the
generator change given the same context?

Usage::

    from trace_ops import load_cassette
    from trace_ops.rag.diff import diff_rag

    old = load_cassette("cassettes/golden.yaml")
    new = load_cassette("cassettes/current.yaml")
    result = diff_rag(old, new)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trace_ops._types import Trace


@dataclass
class ChunkDiff:
    """Diff of a single chunk between two retrieval events."""

    status: str  # "added" | "removed" | "kept" | "score_changed"
    chunk_id: str
    old_score: float | None = None
    new_score: float | None = None
    text_preview: str = ""


@dataclass
class RetrievalDiff:
    """Diff of one retrieval event between two traces."""

    query: str
    chunk_diffs: list[ChunkDiff] = field(default_factory=list)
    old_chunk_count: int = 0
    new_chunk_count: int = 0
    overlap_ratio: float = 1.0
    mean_score_delta: float = 0.0

    @property
    def has_drift(self) -> bool:
        return any(d.status in ("added", "removed") for d in self.chunk_diffs)


@dataclass
class RAGDiffResult:
    """Complete RAG-aware diff result."""

    retrieval_diffs: list[RetrievalDiff] = field(default_factory=list)
    generator_changed: bool = False
    retriever_changed: bool = False
    diagnosis: str = "no_change"
    # Number of retrieval events added (positive) or removed (negative)
    total_retrievals_delta: int = 0

    @property
    def has_changes(self) -> bool:
        """True if retriever drifted or generator output changed."""
        return self.retriever_changed or self.generator_changed

    @property
    def chunks_changed(self) -> int:
        """Total number of chunks added or removed across all retrieval events."""
        count = 0
        for rd in self.retrieval_diffs:
            count += sum(1 for d in rd.chunk_diffs if d.status in ("added", "removed"))
        return count

    def summary(self) -> str:
        lines = ["RAG Comparison:"]
        lines.append("  RETRIEVER:")

        if not self.retrieval_diffs:
            lines.append("    (no retrieval events found)")
        else:
            for rd in self.retrieval_diffs:
                q = rd.query[:60]
                if rd.has_drift:
                    added = [d for d in rd.chunk_diffs if d.status == "added"]
                    removed = [d for d in rd.chunk_diffs if d.status == "removed"]
                    lines.append(f"    ⚠ RETRIEVAL DRIFT for '{q}':")
                    lines.append(
                        f"      {len(removed)} chunks dropped, {len(added)} chunks added"
                    )
                    lines.append(f"      Overlap: {rd.overlap_ratio:.0%}")
                    lines.append(f"      Mean score delta: {rd.mean_score_delta:+.3f}")
                    for d in removed:
                        lines.append(
                            f"      - Dropped: {d.chunk_id} (score: {d.old_score:.2f})"
                        )
                    for d in added:
                        lines.append(
                            f"      + Added:   {d.chunk_id} (score: {d.new_score:.2f})"
                        )
                else:
                    lines.append(f"    ✅ No drift for '{q}'")

        lines.append("")
        lines.append("  GENERATOR:")
        if self.generator_changed:
            lines.append("    ⚠ RESPONSE CHANGED")
        else:
            lines.append("    ✅ Response unchanged")

        lines.append("")
        lines.append(f"  DIAGNOSIS: {self.diagnosis}")
        return "\n".join(lines)


def diff_rag(old_trace: "Trace", new_trace: "Trace") -> RAGDiffResult:
    """Compare RAG-specific events between two traces.

    Args:
        old_trace: Baseline (golden) trace.
        new_trace: Current trace to compare.

    Returns:
        A :class:`RAGDiffResult` describing retriever drift and generator changes.
    """
    old_retrievals = old_trace.retrieval_events
    new_retrievals = new_trace.retrieval_events

    retrieval_diffs: list[RetrievalDiff] = []
    retriever_changed = False

    for old_r, new_r in zip(old_retrievals, new_retrievals):
        def _chunk_map(event):
            m = {}
            for c in (event.chunks or []):
                if isinstance(c, dict):
                    m[c.get("id", "")] = c
                else:
                    m[getattr(c, "id", "")] = c
            return m

        old_chunks = _chunk_map(old_r)
        new_chunks = _chunk_map(new_r)
        all_ids = old_chunks.keys() | new_chunks.keys()

        def _score(mapping, cid):
            c = mapping.get(cid)
            if c is None:
                return None
            return float(c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0))

        def _preview(mapping, cid):
            c = mapping.get(cid)
            if c is None:
                return ""
            t = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            return (t or "")[:80]

        chunk_diffs: list[ChunkDiff] = []
        for cid in all_ids:
            if cid in old_chunks and cid not in new_chunks:
                chunk_diffs.append(
                    ChunkDiff("removed", cid, old_score=_score(old_chunks, cid),
                              text_preview=_preview(old_chunks, cid))
                )
            elif cid not in old_chunks and cid in new_chunks:
                chunk_diffs.append(
                    ChunkDiff("added", cid, new_score=_score(new_chunks, cid),
                              text_preview=_preview(new_chunks, cid))
                )
            else:
                old_s = _score(old_chunks, cid) or 0.0
                new_s = _score(new_chunks, cid) or 0.0
                status = "score_changed" if abs(old_s - new_s) > 0.01 else "kept"
                chunk_diffs.append(ChunkDiff(status, cid, old_score=old_s, new_score=new_s))

        overlap_ids = old_chunks.keys() & new_chunks.keys()
        overlap_ratio = len(overlap_ids) / max(len(all_ids), 1)
        score_deltas = [
            (_score(new_chunks, cid) or 0.0) - (_score(old_chunks, cid) or 0.0)
            for cid in overlap_ids
        ]
        mean_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0

        rd = RetrievalDiff(
            query=old_r.query or "",
            chunk_diffs=chunk_diffs,
            old_chunk_count=len(old_chunks),
            new_chunk_count=len(new_chunks),
            overlap_ratio=overlap_ratio,
            mean_score_delta=mean_delta,
        )
        retrieval_diffs.append(rd)
        if rd.has_drift:
            retriever_changed = True

    # Check if any retrieval events differ in count
    if len(old_retrievals) != len(new_retrievals):
        retriever_changed = True

    # Check generator changes via LLM response content
    from trace_ops._types import EventType

    def _responses(trace):
        return [
            e.response for e in trace.events
            if e.event_type == EventType.LLM_RESPONSE and e.response is not None
        ]

    generator_changed = _responses(old_trace) != _responses(new_trace)

    # Diagnosis
    if retriever_changed and generator_changed:
        diagnosis = "Retriever drift likely caused response change"
    elif retriever_changed and not generator_changed:
        diagnosis = "Retriever drifted but generator produced same output"
    elif not retriever_changed and generator_changed:
        diagnosis = "Same context but generator produced different output (model change?)"
    else:
        diagnosis = "No RAG changes detected"

    return RAGDiffResult(
        retrieval_diffs=retrieval_diffs,
        generator_changed=generator_changed,
        retriever_changed=retriever_changed,
        diagnosis=diagnosis,
        total_retrievals_delta=len(new_retrievals) - len(old_retrievals),
    )
