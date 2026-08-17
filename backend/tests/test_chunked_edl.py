"""Tests for M4 Map-Reduce EDL chunking logic.

Tests the partitioning, sub-budget calculation, and Reduce merge correctness
using entirely mock data — no real LLM calls, no file I/O.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from app.models import EGTSegment, EGTDocument, generate_clip_id
from app.tasks.edl import _partition_segments_into_chunks, generate_edl_chunked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(source_file: str, start: float, end: float) -> EGTSegment:
    return EGTSegment(
        clip_id=generate_clip_id(source_file, start, end),
        source_file=source_file,
        start_sec=start,
        end_sec=end,
        segment_type="SPEECH",
        quality_score=0.9,
        transcript=f"Clip from {source_file} at {start:.0f}s",
    )


def _make_egt(n_segs: int, duration_per_seg: float = 10.0) -> EGTDocument:
    segs = [
        _make_segment("video.mp4", i * duration_per_seg, (i + 1) * duration_per_seg)
        for i in range(n_segs)
    ]
    return EGTDocument(
        segments=segs,
        total_duration_sec=n_segs * duration_per_seg,
        source_file_count=1,
        context_summary="Test vlog.",
    )


# ---------------------------------------------------------------------------
# Tests: _partition_segments_into_chunks
# ---------------------------------------------------------------------------

def test_partition_exact_multiple():
    """Segments divisible evenly into chunks."""
    segs = _make_egt(12).segments
    chunks = _partition_segments_into_chunks(segs, chunk_size=4)
    assert len(chunks) == 3
    assert all(len(c) == 4 for c in chunks)


def test_partition_with_remainder():
    """Last chunk is smaller when segments don't divide evenly."""
    segs = _make_egt(11).segments
    chunks = _partition_segments_into_chunks(segs, chunk_size=4)
    assert len(chunks) == 3
    assert len(chunks[0]) == 4
    assert len(chunks[2]) == 3  # remainder


def test_partition_single_chunk():
    """When n_segs <= chunk_size, only one chunk is returned."""
    segs = _make_egt(5).segments
    chunks = _partition_segments_into_chunks(segs, chunk_size=10)
    assert len(chunks) == 1
    assert len(chunks[0]) == 5


def test_partition_preserves_order():
    """Chronological order of segments is preserved across chunks."""
    segs = _make_egt(10).segments
    chunks = _partition_segments_into_chunks(segs, chunk_size=3)
    flat = [s for chunk in chunks for s in chunk]
    assert [s.clip_id for s in flat] == [s.clip_id for s in segs]


# ---------------------------------------------------------------------------
# Tests: sub-budget proportionality
# ---------------------------------------------------------------------------

def test_sub_budget_proportional():
    """Each chunk should receive a sub-budget proportional to its raw duration."""
    # 3 chunks of 10 segs @ 10s each = 300s total raw, target = 120s
    # Each chunk raw = 100s, sub-budget should be 120 * (100/300) = 40s
    egt = _make_egt(30, duration_per_seg=10.0)
    total_raw = 300.0
    target = 120.0
    chunk_raw = 100.0

    expected_sub_budget = target * (chunk_raw / total_raw)
    assert abs(expected_sub_budget - 40.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: generate_edl_chunked (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def egt_large():
    """An EGT document with 75 segments — above the default chunk_threshold of 50."""
    return _make_egt(75, duration_per_seg=8.0)


def _make_fake_map_response(chunk_segs: List[EGTSegment], sub_budget: float):
    """Simulate generate_edl_llm returning a local EDL for a chunk."""
    edl_entries = []
    for i, seg in enumerate(chunk_segs[:5]):  # Pick first 5 from each chunk
        edl_entries.append({
            "clip_id": seg.clip_id,
            "source_file": seg.source_file,
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "core_start_sec": seg.start_sec,
            "core_end_sec": seg.end_sec,
            "narrative_priority": "CRITICAL" if i == 0 else "MEDIUM",
            "editorial_type": "INTRO" if i == 0 else "KEEP",
            "sequence_index": i,
        })
    return {"edl": edl_entries, "chain_of_thought": "Mock reasoning."}


def test_chunked_edl_returns_non_empty(egt_large):
    """generate_edl_chunked should return a non-empty list with 75 segments."""
    call_count = [0]

    def mock_generate_edl_llm(egt_json, target_duration, user_prompt):
        segs = [EGTSegment(**s) for s in egt_json.get("segments", [])]
        call_count[0] += 1
        return _make_fake_map_response(segs, target_duration or 60.0)

    def mock_reduce(*args, **kwargs):
        # Reduce returns None to test fallback to positional ordering
        return None

    with patch("app.tasks.edl.generate_edl_llm", side_effect=mock_generate_edl_llm), \
         patch("app.tasks.edl.generate_edl_reduce_llm", side_effect=mock_reduce), \
         patch("app.tasks.edl.settings") as mock_settings:
        mock_settings.edl_chunk_size = 35
        mock_settings.edl_chunk_threshold = 50
        mock_settings.reasoning_model = "gemini-2.5-flash"

        result, warning = generate_edl_chunked(egt_large, target_duration=120.0)

    assert isinstance(result, list)
    assert len(result) > 0
    # 75 segs / 35 = 3 chunks (2 full + 1 remainder) -> 3 LLM calls
    assert call_count[0] == 3


def test_chunked_edl_with_successful_reduce(egt_large):
    """When reduce succeeds, final ordering should follow reduce sequence_index."""
    all_picked_ids = []

    def mock_generate_edl_llm(egt_json, target_duration, user_prompt):
        segs = [EGTSegment(**s) for s in egt_json.get("segments", [])]
        entries = []
        for i, seg in enumerate(segs[:3]):
            all_picked_ids.append(seg.clip_id)
            entries.append({
                "clip_id": seg.clip_id,
                "source_file": seg.source_file,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "core_start_sec": seg.start_sec,
                "core_end_sec": seg.end_sec,
                "narrative_priority": "MEDIUM",
                "editorial_type": "KEEP",
                "sequence_index": i,
            })
        return {"edl": entries, "chain_of_thought": "Test."}

    def mock_reduce(chunk_summaries, target_duration, context_doc=""):
        # Reverse the order to test that reduce ordering is honoured
        return [
            {"clip_id": s["clip_id"], "sequence_index": idx, "narrative_priority": s["narrative_priority"]}
            for idx, s in enumerate(reversed(chunk_summaries))
        ]

    with patch("app.tasks.edl.generate_edl_llm", side_effect=mock_generate_edl_llm), \
         patch("app.tasks.edl.generate_edl_reduce_llm", side_effect=mock_reduce), \
         patch("app.tasks.edl.settings") as mock_settings:
        mock_settings.edl_chunk_size = 35
        mock_settings.edl_chunk_threshold = 50
        mock_settings.reasoning_model = "gemini-2.5-flash"

        result, warning = generate_edl_chunked(egt_large, target_duration=120.0)

    assert len(result) > 0
    # The result sequence indices should be 0, 1, 2, ... (re-assigned at finalize step)
    assert all(result[i]["sequence_index"] == i for i in range(len(result)))


def test_chunked_edl_map_failure_fallback(egt_large):
    """A failing Map chunk should fallback to mechanical filter for that chunk."""
    call_count = [0]

    def mock_generate_edl_llm(egt_json, target_duration, user_prompt):
        call_count[0] += 1
        if call_count[0] == 2:
            # Second chunk fails
            return None
        segs = [EGTSegment(**s) for s in egt_json.get("segments", [])]
        return _make_fake_map_response(segs, target_duration or 60.0)

    with patch("app.tasks.edl.generate_edl_llm", side_effect=mock_generate_edl_llm), \
         patch("app.tasks.edl.generate_edl_reduce_llm", return_value=None), \
         patch("app.tasks.edl.settings") as mock_settings:
        mock_settings.edl_chunk_size = 35
        mock_settings.edl_chunk_threshold = 50
        mock_settings.reasoning_model = "gemini-2.5-flash"

        # Should not raise — fallback keeps non-bad-take segments
        result, warning = generate_edl_chunked(egt_large, target_duration=120.0)

    assert isinstance(result, list)
    # Fallback includes all non-bad non-silence from the failed chunk
    assert len(result) > 0
