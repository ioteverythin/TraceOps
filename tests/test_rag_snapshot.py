"""Tests for RetrieverSnapshot — record, load, check, update, check_trace."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_ops._types import EventType, Trace, TraceEvent
from trace_ops.rag.snapshot import QueryResult, RetrieverSnapshot, SnapshotCheckResult

# ── Mock retriever ─────────────────────────────────────────────────────────


def _mock_retriever_fn(chunks_by_query: dict[str, list[dict]]):
    """Return a retriever_fn that returns preset chunks for each query."""
    def fn(query: str, top_k: int = 5) -> list[dict]:
        return chunks_by_query.get(query, [])
    return fn


def _chunk(id: str, score: float = 0.9, text: str = "text") -> dict:
    return {"id": id, "score": score, "text": text}


# ── QueryResult / SnapshotCheckResult ──────────────────────────────────────


def test_query_result_fields():
    qr = QueryResult(query="q", overlap_ratio=0.85, score_delta=0.01)
    assert qr.query == "q"
    assert qr.overlap_ratio == 0.85
    assert qr.score == 0.85  # alias property
    assert qr.passed is True  # not has_drift


def test_query_result_has_drift():
    qr = QueryResult(query="q", overlap_ratio=0.3, score_delta=0.5, has_drift=True)
    assert qr.has_drift is True
    assert qr.passed is False


def test_snapshot_check_result_has_drift_any():
    result = SnapshotCheckResult(queries=[
        QueryResult("q1", 0.9, 0.0),
        QueryResult("q2", 0.2, 0.5, has_drift=True),
    ])
    assert result.has_drift is True


def test_snapshot_check_result_no_drift():
    result = SnapshotCheckResult(queries=[
        QueryResult("q1", 0.95, 0.0),
        QueryResult("q2", 1.0, 0.0),
    ])
    assert result.has_drift is False


def test_snapshot_check_result_min_overlap():
    result = SnapshotCheckResult(queries=[
        QueryResult("q1", 0.9, 0.0),
        QueryResult("q2", 0.7, 0.0),
    ])
    assert result.min_overlap(0.6) is True
    assert result.min_overlap(0.8) is False


def test_snapshot_check_result_summary():
    result = SnapshotCheckResult(queries=[
        QueryResult("q1", 0.9, 0.0, has_drift=False),
        QueryResult("q2", 0.3, 0.5, added_chunks=["c_new"], removed_chunks=["c_old"], has_drift=True),
    ])
    summary = result.summary()
    assert "DRIFT" in summary
    assert "OK" in summary


# ── RetrieverSnapshot.record() ─────────────────────────────────────────────


def test_snapshot_record_saves_yaml(tmp_path: Path):
    queries = ["refund policy", "shipping times"]
    chunks_map = {
        "refund policy": [_chunk("c1"), _chunk("c2")],
        "shipping times": [_chunk("c3")],
    }
    fn = _mock_retriever_fn(chunks_map)
    save_to = tmp_path / "snap.yaml"
    snap = RetrieverSnapshot.record(
        retriever=None,
        queries=queries,
        save_to=str(save_to),
        retriever_fn=fn,
    )
    assert save_to.exists()
    assert len(snap.data["queries"]) == 2


def test_snapshot_record_all_queries_present(tmp_path: Path):
    queries = ["q1", "q2", "q3"]
    fn = _mock_retriever_fn({"q1": [_chunk("a")], "q2": [_chunk("b")], "q3": []})
    snap = RetrieverSnapshot.record(None, queries, save_to=tmp_path / "s.yaml", retriever_fn=fn)
    assert set(snap.data["queries"].keys()) == set(queries)


# ── RetrieverSnapshot.load() ───────────────────────────────────────────────


def test_snapshot_load_roundtrip(tmp_path: Path):
    queries = ["hello"]
    fn = _mock_retriever_fn({"hello": [_chunk("c1", score=0.95)]})
    RetrieverSnapshot.record(None, queries, save_to=tmp_path / "s.yaml", retriever_fn=fn)
    loaded = RetrieverSnapshot.load(tmp_path / "s.yaml")
    assert loaded.data["queries"]["hello"][0]["id"] == "c1"
    assert loaded.data["queries"]["hello"][0]["score"] == pytest.approx(0.95)


def test_snapshot_load_missing_file():
    with pytest.raises(FileNotFoundError):
        RetrieverSnapshot.load("/nonexistent/snap.yaml")


# ── RetrieverSnapshot.check() ─────────────────────────────────────────────


def test_snapshot_check_no_drift(tmp_path: Path):
    queries = ["q1"]
    chunks_map = {"q1": [_chunk("c1"), _chunk("c2")]}
    fn = _mock_retriever_fn(chunks_map)
    snap = RetrieverSnapshot.record(None, queries, save_to=tmp_path / "s.yaml", retriever_fn=fn)
    result = snap.check(None, retriever_fn=fn, threshold=0.8)
    assert not result.has_drift


def test_snapshot_check_detects_drift(tmp_path: Path):
    old_fn = _mock_retriever_fn({"q1": [_chunk("c1"), _chunk("c2"), _chunk("c3")]})
    new_fn = _mock_retriever_fn({"q1": [_chunk("c4"), _chunk("c5"), _chunk("c6")]})
    snap = RetrieverSnapshot.record(None, ["q1"], save_to=tmp_path / "s.yaml", retriever_fn=old_fn)
    result = snap.check(None, retriever_fn=new_fn, threshold=0.8)
    assert result.has_drift


def test_snapshot_check_threshold_respected(tmp_path: Path):
    old_fn = _mock_retriever_fn({"q1": [_chunk("c1"), _chunk("c2"), _chunk("c3"), _chunk("c4")]})
    # Jaccard overlap = |{c1,c2}| / |{c1,c2,c3,c4,c5,c6}| = 2/6 ≈ 0.333
    new_fn = _mock_retriever_fn({"q1": [_chunk("c1"), _chunk("c2"), _chunk("c5"), _chunk("c6")]})
    snap = RetrieverSnapshot.record(None, ["q1"], save_to=tmp_path / "s.yaml", retriever_fn=old_fn)
    result_high = snap.check(None, retriever_fn=new_fn, threshold=0.8)
    result_low = snap.check(None, retriever_fn=new_fn, threshold=0.3)
    assert result_high.has_drift
    assert not result_low.has_drift


# ── RetrieverSnapshot.update() ────────────────────────────────────────────


def test_snapshot_update_overwrites(tmp_path: Path):
    old_fn = _mock_retriever_fn({"q1": [_chunk("c1")]})
    new_fn = _mock_retriever_fn({"q1": [_chunk("c99")]})
    snap = RetrieverSnapshot.record(None, ["q1"], save_to=tmp_path / "s.yaml", retriever_fn=old_fn)
    snap.update(None, save_to=tmp_path / "s.yaml", retriever_fn=new_fn)
    reloaded = RetrieverSnapshot.load(tmp_path / "s.yaml")
    assert reloaded.data["queries"]["q1"][0]["id"] == "c99"


def test_snapshot_update_requires_save_to(tmp_path: Path):
    fn = _mock_retriever_fn({"q": []})
    snap = RetrieverSnapshot.record(None, ["q"], save_to=tmp_path / "s.yaml", retriever_fn=fn)
    with pytest.raises(ValueError, match="save_to is required"):
        snap.update(None, retriever_fn=fn)


# ── RetrieverSnapshot.check_trace() ───────────────────────────────────────


def _trace_with_retrieval(query: str, chunk_ids: list[str]) -> Trace:
    trace = Trace()
    trace.add_event(TraceEvent(
        event_type=EventType.RETRIEVAL,
        query=query,
        chunks=[{"id": cid, "text": "t", "score": 0.9} for cid in chunk_ids],
    ))
    return trace


def test_snapshot_check_trace_passes(tmp_path: Path):
    fn = _mock_retriever_fn({"my query": [_chunk("c1"), _chunk("c2")]})
    snap = RetrieverSnapshot.record(None, ["my query"], save_to=tmp_path / "s.yaml", retriever_fn=fn)
    trace = _trace_with_retrieval("my query", ["c1", "c2"])
    results = snap.check_trace(trace, threshold=0.8)
    assert len(results) == 1
    assert results[0].passed


def test_snapshot_check_trace_detects_drift(tmp_path: Path):
    fn = _mock_retriever_fn({"my query": [_chunk("c1"), _chunk("c2"), _chunk("c3")]})
    snap = RetrieverSnapshot.record(None, ["my query"], save_to=tmp_path / "s.yaml", retriever_fn=fn)
    trace = _trace_with_retrieval("my query", ["c4", "c5", "c6"])
    results = snap.check_trace(trace, threshold=0.8)
    assert len(results) == 1
    assert not results[0].passed


def test_snapshot_check_trace_skips_unmatched_queries(tmp_path: Path):
    fn = _mock_retriever_fn({"known query": [_chunk("c1")]})
    snap = RetrieverSnapshot.record(None, ["known query"], save_to=tmp_path / "s.yaml", retriever_fn=fn)
    trace = _trace_with_retrieval("unknown query", ["c1"])
    results = snap.check_trace(trace, threshold=0.8)
    assert len(results) == 0  # no match in snapshot


# ── RetrieverSnapshot.results property ────────────────────────────────────


def test_snapshot_results_property(tmp_path: Path):
    fn = _mock_retriever_fn({"q1": [_chunk("c1")], "q2": [_chunk("c2")]})
    snap = RetrieverSnapshot.record(None, ["q1", "q2"], save_to=tmp_path / "s.yaml", retriever_fn=fn)
    results = snap.results
    assert isinstance(results, list)
    assert len(results) == 2
    queries = {r["query"] for r in results}
    assert "q1" in queries
    assert "q2" in queries
